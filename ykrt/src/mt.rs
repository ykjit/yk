// Copyright (c) 2017 King's College London
// created by the Software Development Team <http://soft-dev.org/>
//
// The Universal Permissive License (UPL), Version 1.0
//
// Subject to the condition set forth below, permission is hereby granted to any person obtaining a
// copy of this software, associated documentation and/or data (collectively the "Software"), free
// of charge and under any and all copyright rights in the Software, and any and all patent rights
// owned or freely licensable by each licensor hereunder covering either (i) the unmodified
// Software as contributed to or provided by such licensor, or (ii) the Larger Works (as defined
// below), to deal in both
//
// (a) the Software, and
// (b) any piece of software and/or hardware listed in the lrgrwrks.txt file
// if one is included with the Software (each a "Larger Work" to which the Software is contributed
// by such licensors),
//
// without restriction, including without limitation the rights to copy, create derivative works
// of, display, perform, and distribute the Software and make, use, sell, offer for sale, import,
// export, have made, and have sold the Software and the Larger Work(s), and to sublicense the
// foregoing rights on either these or other terms.
//
// This license is subject to the following condition: The above copyright notice and either this
// complete permission notice or at a minimum a reference to the UPL must be included in all copies
// or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#[cfg(test)]
use std::time::Duration;
use std::{
    io,
    panic::{catch_unwind, resume_unwind, UnwindSafe},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

pub type HotThreshold = u32;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;

// The current meta-tracing phase of a given location in the end-user's code. Consists of a tag and
// (optionally) a value. The tags are in the high order bits since we expect the most common tag is
// PHASE_COMPILED which (one day) will have an index associated with it. By also making that tag
// 0b00, we allow that index to be accessed without any further operations after the initial
// tag check.
const PHASE_TAG: u32 = 0b11 << 30; // All of the other PHASE_ tags must fit in this.
const PHASE_COMPILED: u32 = 0b00 << 30;
const PHASE_TRACING: u32 = 0b01 << 30;
const PHASE_COUNTING: u32 = 0b10 << 30; // The value specifies the current hot count.

/// A `Location` uniquely identifies a control point position in the end-user's program (and is
/// used by the `MT` to store data about that location). In other words, every position
/// that can be a control point also needs to have one `Location` value associated with it, and
/// that same `Location` value must always be used to identify that control point.
///
/// As this may suggest, program positions that can't be control points don't need an associated
/// `Location`. For interpreters that can't (or don't want) to be as selective, a simple (if
/// moderately wasteful) mechanism is for every bytecode or AST node to have its own `Location`
/// (even for bytecodes or nodes that can't be control points).
pub struct Location {
    pack: AtomicU32,
}

impl Location {
    /// Create a fresh Location suitable for passing to `MT::control_point`.
    pub fn new() -> Self {
        Self {
            pack: AtomicU32::new(PHASE_COUNTING),
        }
    }
}

/// Configure a meta-tracer. Note that a process can only have one meta-tracer active at one point.
pub struct MTBuilder {
    hot_threshold: HotThreshold,
}

impl MTBuilder {
    /// Create a meta-tracer with default parameters.
    pub fn new() -> Self {
        Self {
            hot_threshold: DEFAULT_HOT_THRESHOLD,
        }
    }

    /// Consume the `MTBuilder` and create a meta-tracer, returning the
    /// [`MTThread`](struct.MTThread.html) representing the current thread.
    pub fn init(self) -> MTThread {
        MTInner::init(self.hot_threshold)
    }

    /// Change this meta-tracer builder's `hot_threshold` value.
    pub fn hot_threshold(mut self, hot_threshold: HotThreshold) -> Self {
        self.hot_threshold = hot_threshold;
        self
    }
}

#[derive(Clone)]
/// A meta-tracer. Note that this is conceptually a "front-end" to the actual meta-tracer akin to
/// an `Rc`: this struct can be freely `clone()`d without duplicating the underlying meta-tracer.
pub struct MT {
    inner: Arc<MTInner>,
}

impl MT {
    /// Return this meta-tracer's hot threshold.
    pub fn hot_threshold(&self) -> HotThreshold {
        self.inner.hot_threshold.load(Ordering::Relaxed)
    }

    /// Create a new thread that can be used in the meta-tracer: the new thread that is created is
    /// handed a [`MTThread`](struct.MTThread.html) from which the `MT` itself can be accessed.
    pub fn spawn<F, T>(&self, f: F) -> io::Result<JoinHandle<T>>
    where
        F: FnOnce(MTThread) -> T,
        F: Send + UnwindSafe + 'static,
        T: Send + 'static,
    {
        let mt_cl = self.clone();
        thread::Builder::new().spawn(move || {
            mt_cl.inner.active_threads.fetch_add(1, Ordering::Relaxed);
            let r = catch_unwind(|| f(MTThreadInner::init(mt_cl.clone())));
            mt_cl.inner.active_threads.fetch_sub(1, Ordering::Relaxed);
            match r {
                Ok(r) => r,
                Err(e) => resume_unwind(e),
            }
        })
    }
}

impl Drop for MT {
    fn drop(&mut self) {
        MT_ACTIVE.store(false, Ordering::Relaxed);
    }
}

/// The innards of a meta-tracer.
struct MTInner {
    hot_threshold: AtomicU32,
    active_threads: AtomicUsize,
}

/// It's only safe to have one `MT` instance active at a time.
static MT_ACTIVE: AtomicBool = AtomicBool::new(false);

impl MTInner {
    /// Create a new `MT`, wrapped immediately in an [`MTThread`](struct.MTThread.html).
    fn init(hot_threshold: HotThreshold) -> MTThread {
        // A process can only have a single MT instance.

        // In non-testing, we panic if the user calls this method while an MT instance is active.
        #[cfg(not(test))]
        {
            if MT_ACTIVE.swap(true, Ordering::Relaxed) {
                panic!("Only one MT can be active at once.");
            }
        }

        // In testing, we simply sleep until the other MT instance has gone away: this has the
        // effect of serialising tests which use MT (but allowing other tests to run in parallel).
        #[cfg(test)]
        {
            loop {
                if !MT_ACTIVE.swap(true, Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }

        let mtc = Self {
            hot_threshold: AtomicU32::new(hot_threshold),
            active_threads: AtomicUsize::new(1),
        };
        let mt = MT {
            inner: Arc::new(mtc),
        };
        MTThreadInner::init(mt)
    }
}

/// A meta-tracer aware thread. Note that this is conceptually a "front-end" to the actual
/// meta-tracer thread akin to an `Rc`: this struct can be freely `clone()`d without duplicating
/// the underlying meta-tracer thread. Note that this struct is neither `Send` nor `Sync`: it
/// can only be accessed from within a single thread.
#[derive(Clone)]
pub struct MTThread {
    inner: Rc<MTThreadInner>,
}

impl MTThread {
    /// Return a meta-tracer [`MT`](struct.MT.html) struct.
    pub fn mt(&self) -> &MT {
        &self.inner.mt
    }

    /// Attempt to execute a compiled trace for location `loc`.
    pub fn control_point(&self, loc: &Location) {
        // Since we don't hold an explicit lock, updating a Location is tricky: we might read a
        // Location, work out what we'd like to update it to, and try updating it, only to find
        // that another thread interrupted us part way through. We therefore use compare_and_swap
        // to update values, allowing us to detect if we were interrupted. If we were interrupted,
        // we simply retry the whole operation.
        loop {
            let pack = &loc.pack;
            // We need Acquire ordering, as PHASE_COMPILED will need to read information written to
            // external data as a result of the PHASE_TRACING -> PHASE_COMPILED transition.
            let lp = pack.load(Ordering::Acquire);
            match lp & PHASE_TAG {
                PHASE_COUNTING => {
                    let count = lp & !PHASE_TAG;
                    let new_pack;
                    if count >= self.inner.hot_threshold {
                        new_pack = PHASE_TRACING;
                    } else {
                        new_pack = PHASE_COUNTING | (count + 1);
                    }
                    if pack.compare_and_swap(lp, new_pack, Ordering::Release) == lp {
                        break;
                    }
                }
                PHASE_TRACING => {
                    if pack.compare_and_swap(lp, PHASE_COMPILED, Ordering::Release) == lp {
                        break;
                    }
                }
                PHASE_COMPILED => break,
                _ => unreachable!(),
            }
        }
    }
}

/// The innards of a meta-tracer thread.
struct MTThreadInner {
    mt: MT,
    hot_threshold: HotThreshold,
}

impl MTThreadInner {
    fn init(mt: MT) -> MTThread {
        let hot_threshold = mt.hot_threshold();
        let inner = MTThreadInner { mt, hot_threshold };
        MTThread {
            inner: Rc::new(inner),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::{black_box, Bencher};
    use super::*;
    use std::sync::Arc;

    #[test]
    fn threshold_passed() {
        let hot_thrsh = 1500;
        let mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let lp = Location::new();
        for i in 0..hot_thrsh {
            mtt.control_point(&lp);
            assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_COUNTING | (i + 1));
        }
        mtt.control_point(&lp);
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_TRACING);
        mtt.control_point(&lp);
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_COMPILED);
    }

    #[test]
    fn threaded_threshold_passed() {
        let hot_thrsh = 4000;
        let mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let l_arc = Arc::new(Location::new());
        let mut thrs = vec![];
        for _ in 0..hot_thrsh / 4 {
            let l_arc_cl = l_arc.clone();
            let t = mtt
                .mt()
                .spawn(move |mtt| {
                    mtt.control_point(&*l_arc_cl);
                    let c1 = l_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c1 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(&*l_arc_cl);
                    let c2 = l_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c2 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(&*l_arc_cl);
                    let c3 = l_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c3 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(&*l_arc_cl);
                    let c4 = l_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c4 & PHASE_TAG, PHASE_COUNTING);
                    assert!(c4 > c3);
                    assert!(c3 > c2);
                    assert!(c2 > c1);
                })
                .unwrap();
            thrs.push(t);
        }
        for t in thrs {
            t.join().unwrap();
        }
        {
            mtt.control_point(&l_arc);
            assert_eq!(l_arc.pack.load(Ordering::Relaxed), PHASE_TRACING);
            mtt.control_point(&l_arc);
            assert_eq!(l_arc.pack.load(Ordering::Relaxed), PHASE_COMPILED);
        }
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mtt = MTBuilder::new().init();
        let lp = Location::new();
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mtt.control_point(&lp));
            }
        });
    }

    #[bench]
    fn bench_multi_threaded_control_point(b: &mut Bencher) {
        let mtt = MTBuilder::new().init();
        let l_arc = Arc::new(Location::new());
        b.iter(|| {
            let mut thrs = vec![];
            for _ in 0..4 {
                let l_arc_cl = Arc::clone(&l_arc);
                let t = mtt
                    .mt()
                    .spawn(move |mtt| {
                        for _ in 0..100000 {
                            black_box(mtt.control_point(&*l_arc_cl));
                        }
                    })
                    .unwrap();
                thrs.push(t);
            }
            for t in thrs {
                t.join().unwrap();
            }
        });
    }
}
