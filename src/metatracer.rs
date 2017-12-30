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

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::sync::Mutex;

const HOT_THRESHOLD: u16 = 10;

/// The current meta-tracing state of a given location in the end-user's code.
#[derive(Debug, PartialEq)]
enum LocPhase {
    /// Below `HOT_THRESHOLD`, but counting up towards it.
    Counting(u16),
    /// A trace is being collected.
    Tracing,
    /// A compiled trace is available.
    Compiled
}

/// A meta-tracer.
pub struct MetaTracer<Loc> {
    counts: Mutex<HashMap<Loc, LocPhase>>
}

impl<Loc: Eq + Hash> MetaTracer<Loc> {
    pub fn new() -> Self {
        Self {
            counts: Mutex::new(HashMap::new())
        }
    }

    /// Attempt to execute a compiled trace for location `loc`: return a `ControlOutcome` to allow
    /// the end user to determine whether a trace was executed or not.
    pub fn control_point(&self, loc: Loc) {
        let mut cnts = self.counts.lock().unwrap();
        match cnts.entry(loc) {
            Entry::Occupied(mut v) => {
                match *v.get() {
                    LocPhase::Counting(x) => {
                        if x >= HOT_THRESHOLD {
                            v.insert(LocPhase::Tracing);
                        } else {
                            v.insert(LocPhase::Counting(x + 1));
                        }
                    },
                    LocPhase::Tracing => {
                        v.insert(LocPhase::Compiled);
                    },
                    LocPhase::Compiled => ()
                }
            },
            Entry::Vacant(v) => {
                v.insert(LocPhase::Counting(1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use super::*;

    #[test]
    fn threshold_passed() {
        let mt = MetaTracer::new();
        for i in 0..HOT_THRESHOLD {
            mt.control_point(0);
            assert_eq!(*mt.counts.lock().unwrap().get(&0).unwrap(), LocPhase::Counting(i + 1));
        }
        mt.control_point(0);
        assert_eq!(*mt.counts.lock().unwrap().get(&0).unwrap(), LocPhase::Tracing);
        mt.control_point(0);
        assert_eq!(*mt.counts.lock().unwrap().get(&0).unwrap(), LocPhase::Compiled);
    }

    #[test]
    fn threaded_threshold_passed() {
        let mt = Arc::new(MetaTracer::new());
        let mut thrs = vec![];
        for _ in 0..HOT_THRESHOLD {
            let mt_cl = Arc::clone(&mt);
            let t = thread::Builder::new()
                .spawn(move || {
                    mt_cl.control_point(0);
                    let l = mt_cl.counts.lock().unwrap();
                    let v = l.get(&0).unwrap(); 
                    match v {
                        &LocPhase::Counting(_) => (),
                        _ => panic!("Read {:?} instead of ControlOutcome::Counting(_)", v)
                    }
                })
                .unwrap();
            thrs.push(t);
        }
        for t in thrs {
            t.join().unwrap();
        }
        mt.control_point(0);
        assert_eq!(*mt.counts.lock().unwrap().get(&0).unwrap(), LocPhase::Tracing);
        mt.control_point(0);
        assert_eq!(*mt.counts.lock().unwrap().get(&0).unwrap(), LocPhase::Compiled);
    }
}
