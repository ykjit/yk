//! Conceptually this module takes an ordered collection of SIR block locations and converts it
//! into a Tracing IR (TIR) Trace using the SIR found in the `.yk_sir` section of the currently
//! running executable.

use super::SirTrace;
use crate::errors::InvalidTraceError;
use elf;
use fallible_iterator::FallibleIterator;
use std::{collections::HashMap, convert::TryFrom, env, fmt, io::Cursor};
use ykpack::{bodyflags, Body, Decoder, DefId, Pack, Terminator};
pub use ykpack::{
    BinOp, Constant, ConstantInt, Local, LocalIndex, Operand, Place, PlaceBase, PlaceProjection,
    Rvalue, SignedInt, Statement, UnsignedInt
};

lazy_static! {
    pub static ref SIR: Sir = {
        let ef = elf::File::open_path(env::current_exe().unwrap()).unwrap();
        let sec = ef.get_section(".yk_sir").expect("Can't find SIR section");
        let mut curs = Cursor::new(&sec.data);
        let mut dec = Decoder::from(&mut curs);

        let mut bodies = HashMap::new();
        let mut trace_heads = Vec::new();
        let mut trace_tails = Vec::new();
        while let Some(pack) = dec.next().unwrap() {
            match pack {
                Pack::Body(body) => {
                    // Cache some locations that we need quick access to.
                    if body.flags & bodyflags::TRACE_HEAD != 0 {
                        trace_heads.push(body.def_id.clone());
                    }

                    if body.flags & bodyflags::TRACE_TAIL != 0 {
                        trace_tails.push(body.def_id.clone());
                    }

                    let old = bodies.insert(body.def_id.clone(), body);
                    debug_assert!(old.is_none()); // should be no duplicates.
                },
                Pack::Debug(_) => (),
            }
        }

        assert_eq!(trace_heads.is_empty(), false);
        assert_eq!(trace_tails.is_empty(), false);
        let markers = SirMarkers { trace_heads, trace_tails };

        Sir {bodies, markers}
    };
}

/// The serialised IR loaded in from disk. One of these structures is generated in the above
/// `lazy_static` and is shared immutably for all threads.
pub struct Sir {
    /// Lets us map a DefId from a trace to a SIR body.
    pub bodies: HashMap<DefId, Body>,
    // Interesting locations that we need quick access to.
    pub markers: SirMarkers
}

/// Contains the DefIds of interesting locations required for trace manipulation.
pub struct SirMarkers {
    /// Functions which start tracing and whose suffix gets trimmed off the top of traces.
    /// Although you'd expect only one such function, (i.e. `yktrace::start_tracing`), in fact
    /// the location which appears in the trace can vary according to how Rust compiles the
    /// program (this happens even if `yktracer::start_tracing()` is marked `#[inline(never)]`).
    /// For this reason, we mark few different places as potential heads.
    ///
    /// We will only see the suffix of these functions in traces, as trace recording will start
    /// somewhere in the middle of them.
    ///
    /// The compiler is made aware of this location by the `#[trace_head]` annotation.
    pub trace_heads: Vec<DefId>,
    /// Similar to `trace_heads`, functions which stop tracing and whose prefix gets trimmed off
    /// the bottom of traces.
    ///
    /// The compiler is made aware of these locations by the `#[trace_tail]` annotation.
    pub trace_tails: Vec<DefId>
}

/// A TIR trace is conceptually a straight-line path through the SIR with guarded speculation.
#[derive(Debug)]
pub struct TirTrace {
    ops: Vec<TirOp>
}

impl TirTrace {
    /// Create a TirTrace from a SirTrace, trimming remnants of the code which starts/stops the
    /// tracer. Returns a TIR trace and the bounds the SIR trace was trimmed to, or Err if a DefId
    /// is encountered for which no SIR is available.
    pub fn new<'s>(trace: &'s dyn SirTrace) -> Result<Self, InvalidTraceError> {
        let mut ops = Vec::new();
        let mut itr = trace.into_iter().peekable();
        while let Some(loc) = itr.next() {
            let body = match SIR.bodies.get(&DefId::from_sir_loc(&loc)) {
                Some(b) => b,
                None => {
                    let def_id = DefId::from_sir_loc(&loc);
                    return Err(InvalidTraceError::no_sir(&def_id));
                }
            };

            // When adding statements to the trace, we clone them (rather than referencing the
            // statements in the SIR) so that we have the freedom to mutate them later.
            let user_bb_idx_usize = usize::try_from(loc.bb_idx()).unwrap();
            ops.extend(
                body.blocks[user_bb_idx_usize]
                    .stmts
                    .iter()
                    .cloned()
                    .map(TirOp::Statement)
            );

            // Convert the block terminator to a guard if necessary.
            let guard = match body.blocks[user_bb_idx_usize].term {
                Terminator::Goto(_)
                | Terminator::Return
                | Terminator::Drop { .. }
                | Terminator::DropAndReplace { .. }
                | Terminator::Call { .. }
                | Terminator::Unimplemented(_) => None,
                Terminator::Unreachable => panic!("Traced unreachable code"),
                Terminator::SwitchInt {
                    ref discr,
                    ref values,
                    ref target_bbs,
                    otherwise_bb
                } => {
                    // Peek at the next block in the trace to see which outgoing edge was taken and
                    // infer which value we must guard upon. We are working on the assumption that
                    // a trace can't end on a SwitchInt. i.e. that another block follows.
                    let next_blk = itr.peek().expect("no block to peek at").bb_idx();
                    let edge_idx = target_bbs.iter().position(|e| *e == next_blk);
                    match edge_idx {
                        Some(idx) => Some(Guard {
                            val: discr.clone(),
                            kind: GuardKind::Integer(values[idx].val())
                        }),
                        None => {
                            debug_assert!(next_blk == otherwise_bb);
                            Some(Guard {
                                val: discr.clone(),
                                kind: GuardKind::OtherInteger(
                                    values.iter().map(|v| v.val()).collect()
                                )
                            })
                        }
                    }
                }
                Terminator::Assert {
                    ref cond,
                    ref expected,
                    ..
                } => Some(Guard {
                    val: cond.clone(),
                    kind: GuardKind::Boolean(*expected)
                })
            };

            if guard.is_some() {
                ops.push(TirOp::Guard(guard.unwrap()));
            }
        }

        Ok(Self { ops })
    }

    /// Return the TIR operation at index `idx` in the trace.
    /// The index must not be out of bounds.
    pub fn op(&self, idx: usize) -> &TirOp {
        debug_assert!(idx <= self.ops.len() - 1, "bogus trace index");
        unsafe { &self.ops.get_unchecked(idx) }
    }

    /// Return the length of the trace measure in operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }
}

/// A guard states the assumptions from its position in a trace onward.
#[derive(Debug)]
pub struct Guard {
    /// The value to be checked if the guard is to pass.
    pub val: Place,
    /// The requirement upon `val` for the guard to pass.
    pub kind: GuardKind
}

/// A guard states the assumptions from its position in a trace onward.
#[derive(Debug)]
pub enum GuardKind {
    /// The value must be equal to an integer constant.
    Integer(u128),
    /// The value must not be a member of the specified collection of integers. This is necessary
    /// due to the "otherwise" semantics of the `SwitchInt` terminator in SIR.
    OtherInteger(Vec<u128>),
    /// The value must equal a Boolean constant.
    Boolean(bool)
}

impl fmt::Display for Guard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "guard({}, {})", self.val, self.kind)
    }
}

impl fmt::Display for GuardKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Integer(u128v) => write!(f, "integer({})", u128v),
            Self::OtherInteger(u128vs) => write!(f, "other_integer({:?})", u128vs),
            Self::Boolean(expect) => write!(f, "bool({})", expect)
        }
    }
}

/// A TIR operation. A collection of these makes a TIR trace.
#[derive(Debug)]
pub enum TirOp {
    Statement(Statement),
    Guard(Guard)
}

impl fmt::Display for TirOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TirOp::Statement(st) => write!(f, "{}", st),
            TirOp::Guard(gd) => write!(f, "{}", gd)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TirTrace;
    use crate::{start_tracing, TracingKind};
    use test::black_box;

    // Some work to trace.
    #[inline(never)]
    fn work(x: usize, y: usize) -> usize {
        let mut res = 0;
        while res < y {
            res += x;
        }
        res
    }

    #[test]
    fn nonempty_tir_trace() {
        #[cfg(tracermode = "sw")]
        let mut tracer = start_tracing(Some(TracingKind::SoftwareTracing));
        #[cfg(tracermode = "hw")]
        let mut tracer = start_tracing(Some(TracingKind::HardwareTracing));
        let res = black_box(work(black_box(3), black_box(13)));
        let sir_trace = tracer.t_impl.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        assert_eq!(res, 15);
        assert!(tir_trace.len() > 0);
    }
}
