//! The job queue. This runs 1 or more worker threads and has them run compilation jobs as
//! appropriate.

use crate::mt::{MT, TraceId};
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::{
    cmp,
    collections::VecDeque,
    env,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

/// A job for the job queue.
pub(crate) struct Job {
    /// The function we expect to run for this job when it is ready.
    main: Box<dyn FnOnce() + Send>,
    /// If `Some`, wait until [TraceID] has compiled before running this job.
    connector_tid: Option<TraceId>,
    /// If `connector_tid` failed to compile, this function will be run.
    connector_failed: Box<dyn FnOnce() + Send>,
}

impl Job {
    /// Create a new job with a `main` method. If `connector_tid` is `Some`, `main` will only be
    /// run when `connector_tid` has compiled. If `connector_tid` fails to compile, then
    /// `connector_failed` will be run (and `main` will not be run).
    pub(crate) fn new(
        main: Box<dyn FnOnce() + Send>,
        connector_tid: Option<TraceId>,
        connector_failed: Box<dyn FnOnce() + Send>,
    ) -> Self {
        Self {
            main,
            connector_tid,
            connector_failed,
        }
    }
}

pub(crate) struct JobQueue {
    /// The hard cap on the number of worker threads.
    max_worker_threads: AtomicUsize,
    /// How many worker threads are waiting for work?
    idle_worker_threads: AtomicUsize,
    /// [JoinHandle]s to each worker thread so that when an [MT] value is dropped, we can try
    /// joining each worker thread and see if it caused an error or not. If it did, we can
    /// percolate the error upwards, making it more likely that the main thread exits with an
    /// error. In other words, this [Vec] makes it harder for errors to be missed.
    worker_threads: Mutex<Vec<JoinHandle<()>>>,
    /// The ordered queue of compilation worker functions, each a pair `(Option<connector_tid>,
    /// job)`. Before `job` is run, `connector_tid`, if it is `Some`, must be present in
    /// [MT::compiled_traces].
    queue: Arc<(Condvar, Mutex<VecDeque<Job>>)>,
}

impl JobQueue {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Arc::new((Condvar::new(), Mutex::new(VecDeque::new()))),
            max_worker_threads: AtomicUsize::new(cmp::max(
                1,
                match env::var("YK_JOBS") {
                    Ok(x) => x
                        .parse::<usize>()
                        .unwrap_or_else(|x| panic!("Invalid value for YK_JOBS: {x}")),
                    _ => num_cpus::get() - 1,
                },
            )),
            worker_threads: Mutex::new(Vec::new()),
            idle_worker_threads: AtomicUsize::new(0),
        })
    }

    /// Shut the queue down to the extent possible (running jobs cannot be cancelled and will
    /// continue running). The main utility of this function is that it can detect worker threads
    /// that have failed and, if we're lucky, get them to output what went wrong.
    pub(crate) fn shutdown(&self) {
        let mut lk = self.worker_threads.lock();
        for hdl in lk.drain(..) {
            if hdl.is_finished()
                && let Err(e) = hdl.join()
            {
                // Despite the name `resume_unwind` will abort if the unwind strategy in
                // Rust is set to `abort`.
                eprintln!("yk worker thread error");
                std::panic::resume_unwind(e);
            }
        }
    }

    /// Check the integrity of the job queue: if any job queue thread has panicked, this function
    /// will itself panic. This should only be used for testing purposes.
    #[cfg(feature = "yk_testing")]
    pub(super) fn check_integrity(&self) {
        let mut lk = self.worker_threads.lock();
        let mut i = 0;
        while i < lk.len() {
            if lk[i].is_finished() {
                if let Err(e) = lk.remove(i).join() {
                    // Despite the name `resume_unwind` will abort if the unwind strategy in
                    // Rust is set to `abort`.
                    eprintln!("yk worker thread error");
                    std::panic::resume_unwind(e);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Queue `job` to be run on a worker thread.
    pub(crate) fn push(self: &Arc<Self>, mt: &Arc<MT>, job: Job) {
        #[cfg(feature = "yk_testing")]
        if let Ok(true) = env::var("YKD_SERIALISE_COMPILATION").map(|x| x.as_str() == "1") {
            // To ensure that we properly test that compilation can occur in another thread, we
            // spin up a new thread for each compilation. This is only acceptable because a)
            // `SERIALISE_COMPILATION` is an internal yk testing feature b) when we use it we're
            // checking correctness, not performance.
            if let Some(tid) = job.connector_tid
                && !mt.compiled_traces.lock().contains_key(&tid)
            {
                self.queue.1.lock().push_back(job);
                return;
            }
            thread::spawn(job.main).join().unwrap();
            loop {
                let mut lk = self.queue.1.lock();
                let cnd = {
                    let ct_lk = mt.compiled_traces.lock();
                    lk.iter().position(|x| match &x.connector_tid {
                        Some(x) => ct_lk.contains_key(x),
                        None => true,
                    })
                };
                match cnd {
                    Some(x) => {
                        let job = lk.remove(x).unwrap();
                        drop(lk);
                        thread::spawn(job.main).join().unwrap();
                    }
                    None => break,
                }
            }
            return;
        }

        // Push the job onto the queue.
        let (cv, mtx) = &*self.queue;
        mtx.lock().push_back(job);
        cv.notify_one();

        // Is there an idle worker thread that can take the job on?
        if self.idle_worker_threads.load(Ordering::Relaxed) == 0 {
            // Do we have enough active worker threads? If not, spin another up.
            let mut lk = self.worker_threads.lock();
            if lk.len() < self.max_worker_threads.load(Ordering::Relaxed) {
                self.idle_worker_threads.fetch_add(1, Ordering::Relaxed);
                // We only keep a weak reference alive to `MT`, as otherwise an active compiler job
                // causes it to never be dropped.
                let mt_wk = Arc::downgrade(mt);
                let self_cl = Arc::clone(self);
                let jq = Arc::clone(&self.queue);
                let hdl = thread::spawn(move || {
                    let (cv, mtx) = &*jq;
                    let mut lk = mtx.lock();
                    // If the strong count for `mt` is 0 then it has been dropped and there is no
                    // point trying to do further work, even if there is work in the queue.
                    while let Some(mt_st) = mt_wk.upgrade() {
                        // Search through the queue looking for the first job we can compile (i.e.
                        // there is no connector trace ID, or the connector trade ID has been
                        // compiled).
                        self_cl.idle_worker_threads.fetch_sub(1, Ordering::Relaxed);
                        let cnd = {
                            let ct_lk = mt_st.compiled_traces.lock();
                            lk.iter().position(|x| match &x.connector_tid {
                                Some(x) => ct_lk.contains_key(x),
                                None => true,
                            })
                        };
                        match cnd {
                            Some(x) => {
                                let job = lk.remove(x).unwrap();
                                MutexGuard::unlocked(&mut lk, job.main);
                                self_cl.idle_worker_threads.fetch_add(1, Ordering::Relaxed);
                            }
                            None => {
                                self_cl.idle_worker_threads.fetch_add(1, Ordering::Relaxed);
                                cv.wait(&mut lk);
                            }
                        }
                    }
                });
                lk.push(hdl);
            }
        }
    }

    /// Notify the queue that `trid` has successfully completed. If there are other jobs waiting on
    /// `trid`, this function will try to have them run.
    pub(crate) fn notify_success(&self, trid: TraceId) {
        #[cfg(feature = "yk_testing")]
        if let Ok(true) = env::var("YKD_SERIALISE_COMPILATION").map(|x| x.as_str() == "1") {
            return;
        }
        // Since waking worker threads up is quite disruptive for the system, only send a wake-up
        // to other threads if necessary. Note: the most common outcomes are that 0 or 1 jobs are
        // waiting on us.
        let cnt = {
            let lk = self.queue.1.lock();
            lk.iter()
                .filter(|job| match &job.connector_tid {
                    Some(x) => *x == trid,
                    None => false,
                })
                .count()
        };
        if cnt == 1 {
            self.queue.0.notify_one();
        } else if cnt > 1 {
            self.queue.0.notify_all();
        }
    }

    /// Notify the queue that `trid` has failed. If there are other jobs waiting on `trid`, this
    /// function will inform them that they cannot run.
    pub(crate) fn notify_failure(self: &Arc<Self>, _mt: &Arc<MT>, trid: TraceId) {
        let mut removed = Vec::new();
        let mut i = 0;
        let mut lk = self.queue.1.lock();
        while i < lk.len() {
            if lk[i].connector_tid == Some(trid) {
                removed.push(lk.remove(i).unwrap());
            } else {
                i += 1;
            }
        }
        drop(lk);
        for x in removed {
            (x.connector_failed)();
        }
    }
}
