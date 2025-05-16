//! The job queue. This runs 1 or more worker threads and has them run compilation jobs as
//! appropriate.

use crate::mt::{TraceId, MT};
use parking_lot::{Condvar, Mutex, MutexGuard};
#[cfg(feature = "yk_testing")]
use std::env;
use std::{
    cmp,
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

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
    queue: Arc<(
        Condvar,
        Mutex<VecDeque<(Option<TraceId>, Box<dyn FnOnce() + Send>)>>,
    )>,
}

impl JobQueue {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Arc::new((Condvar::new(), Mutex::new(VecDeque::new()))),
            max_worker_threads: AtomicUsize::new(cmp::max(1, num_cpus::get() - 1)),
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

    /// Queue `job` to be run on a worker thread. If `connector_tid` is `Some`, wait for the
    /// relevant trace to be compiled before running `job`.
    pub(crate) fn push(
        self: &Arc<Self>,
        mt: &Arc<MT>,
        connector_tid: Option<TraceId>,
        job: Box<dyn FnOnce() + Send>,
    ) {
        #[cfg(feature = "yk_testing")]
        if let Ok(true) = env::var("YKD_SERIALISE_COMPILATION").map(|x| x.as_str() == "1") {
            // To ensure that we properly test that compilation can occur in another thread, we
            // spin up a new thread for each compilation. This is only acceptable because a)
            // `SERIALISE_COMPILATION` is an internal yk testing feature b) when we use it we're
            // checking correctness, not performance.
            thread::spawn(job).join().unwrap();
            return;
        }

        // Push the job onto the queue.
        let (cv, mtx) = &*self.queue;
        mtx.lock().push_back((connector_tid, job));
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
                            lk.iter()
                                .position(|(connector_tid, _)| match connector_tid {
                                    Some(x) => ct_lk.contains_key(x),
                                    None => true,
                                })
                        };
                        match cnd {
                            Some(x) => {
                                let (_, ctr) = lk.remove(x).unwrap();
                                MutexGuard::unlocked(&mut lk, ctr);
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
        // Since waking worker threads up is quite disruptive for the system, only send a wake-up
        // to other threads if necessary. Note: the most common outcomes are that 0 or 1 jobs are
        // waiting on us.
        let cnt = {
            let lk = self.queue.1.lock();
            lk.iter()
                .filter(|(ref connector_tid, _)| match connector_tid {
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
}
