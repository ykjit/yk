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

use std::sync::atomic::{AtomicU32, Ordering};

pub type HotThreshold = u32;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;

// The current meta-tracing phase of a given location in the end-user's code. Consists of a tag and
// (optionally) a value. The tags are in the high order bits since we expect the most common tag is
// PHASE_COMPILED which (one day) will have an index associated with it. By also making that tag
// 0b00, we allow that index to be accessed without any further operations after the initial
// tag check.
const PHASE_TAG     : u32 = 0b11 << 30; // All of the other PHASE_ tags must fit in this.
const PHASE_COMPILED: u32 = 0b00 << 30;
const PHASE_TRACING : u32 = 0b01 << 30;
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
    pack: AtomicU32
}

impl Location {
    /// Create a fresh Location suitable for passing to `MT::control_point`.
    pub fn new() -> Self {
        Self { pack: AtomicU32::new(PHASE_COUNTING) }
    }
}

/// A meta-tracer.
pub struct MT {
    hot_threshold: HotThreshold
}

impl MT {
    /// Create a new `MT` with default settings.
    pub fn new() -> Self {
        Self::new_with_hot_threshold(DEFAULT_HOT_THRESHOLD)
    }

    /// Create a new `MT` with a specific hot threshold.
    pub fn new_with_hot_threshold(hot_threshold: HotThreshold) -> Self {
        Self {
            hot_threshold: hot_threshold
        }
    }

    /// Attempt to execute a compiled trace for location `loc`.
    pub fn control_point(&self, loc: &Location)
    {
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
                    if count >= self.hot_threshold {
                        new_pack = PHASE_TRACING;
                    } else {
                        new_pack = PHASE_COUNTING | (count + 1);
                    }
                    if pack.compare_and_swap(lp, new_pack, Ordering::Release) == lp {
                        break;
                    }
                },
                PHASE_TRACING => {
                    if pack.compare_and_swap(lp, PHASE_COMPILED, Ordering::Release) == lp {
                        break;
                    }
                },
                PHASE_COMPILED => break,
                _ => unreachable!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use test::{Bencher, black_box};
    use super::*;

    #[test]
    fn threshold_passed() {
        let hot_thrsh = 1500;
        let mt = MT::new_with_hot_threshold(hot_thrsh);
        let lp = Location::new();
        for i in 0..hot_thrsh {
            mt.control_point(&lp);
            assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_COUNTING | (i + 1));
        }
        mt.control_point(&lp);
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_TRACING);
        mt.control_point(&lp);
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_COMPILED);
    }

    #[test]
    fn threaded_threshold_passed() {
        let hot_thrsh = 4000;
        let mt_arc;
        {
            let mt = MT::new_with_hot_threshold(hot_thrsh);
            mt_arc = Arc::new(mt);
        }
        let lp_arc = Arc::new(Location::new());
        let mut thrs = vec![];
        for _ in 0..hot_thrsh / 4 {
            let mt_arc_cl = Arc::clone(&mt_arc);
            let lp_arc_cl = Arc::clone(&lp_arc);
            let t = thread::Builder::new()
                .spawn(move || {
                    mt_arc_cl.control_point(&*lp_arc_cl);
                    let c1 = lp_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c1 & PHASE_TAG, PHASE_COUNTING);
                    mt_arc_cl.control_point(&*lp_arc_cl);
                    let c2 = lp_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c2 & PHASE_TAG, PHASE_COUNTING);
                    mt_arc_cl.control_point(&*lp_arc_cl);
                    let c3 = lp_arc_cl.pack.load(Ordering::Relaxed);
                    assert_eq!(c3 & PHASE_TAG, PHASE_COUNTING);
                    mt_arc_cl.control_point(&*lp_arc_cl);
                    let c4 = lp_arc_cl.pack.load(Ordering::Relaxed);
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
        mt_arc.control_point(&lp_arc);
        assert_eq!(lp_arc.pack.load(Ordering::Relaxed), PHASE_TRACING);
        mt_arc.control_point(&lp_arc);
        assert_eq!(lp_arc.pack.load(Ordering::Relaxed), PHASE_COMPILED);
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mt = MT::new();
        let lp = Location::new();
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mt.control_point(&lp));
            }
        });
    }

    #[bench]
    fn bench_multi_threaded_control_point(b: &mut Bencher) {
        let mt_arc = Arc::new(MT::new());
        let lp_arc = Arc::new(Location::new());
        b.iter(|| {
            let mut thrs = vec![];
            for _ in 0..4 {
                let mt_arc_cl = Arc::clone(&mt_arc);
                let lp_arc_cl = Arc::clone(&lp_arc);
                let t = thread::Builder::new()
                    .spawn(move || {
                        for _ in 0..100000 {
                            black_box(mt_arc_cl.control_point(&*lp_arc_cl));
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
