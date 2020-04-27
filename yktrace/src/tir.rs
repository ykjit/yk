//! Conceptually this module takes an ordered collection of SIR block locations and converts it
//! into a Tracing IR (TIR) Trace using the SIR found in the `.yk_sir` section of the currently
//! running executable.

use super::SirTrace;
use crate::errors::InvalidTraceError;
use elf;
use fallible_iterator::FallibleIterator;
use std::{
    collections::HashMap,
    convert::{From, TryFrom},
    env,
    fmt::{self, Display},
    io::Cursor
};
use ykpack::{bodyflags, Body, Decoder, Pack, Terminator};
pub use ykpack::{
    BinOp, CallOperand, Constant, ConstantInt, Local, LocalIndex, Operand, Place, PlaceBase,
    PlaceElem, Rvalue, SignedInt, Statement, UnsignedInt
};

lazy_static! {
    pub static ref SIR: Sir = {
        let ef = elf::File::open_path(env::current_exe().unwrap()).unwrap();

        // We iterate over ELF sections, looking for ones which contain SIR and loading it into
        // memory.
        let mut bodies = HashMap::new();
        let mut trace_heads = Vec::new();
        let mut trace_tails = Vec::new();
        for sec in &ef.sections {
            if sec.shdr.name.starts_with(".yksir_") {
                let mut curs = Cursor::new(&sec.data);
                let mut dec = Decoder::from(&mut curs);

                while let Some(pack) = dec.next().unwrap() {
                    match pack {
                        Pack::Body(body) => {
                            // Cache some locations that we need quick access to.
                            if body.flags & bodyflags::TRACE_HEAD != 0 {
                                trace_heads.push(body.symbol_name.clone());
                            }

                            if body.flags & bodyflags::TRACE_TAIL != 0 {
                                trace_tails.push(body.symbol_name.clone());
                            }

                            // Due to the way Rust compiles stuff, duplicates may exist. Where
                            // duplicates exist, the functions will be identical.
                            if let Some(old) = bodies.get(&body.symbol_name) {
                                debug_assert!(old == &body);
                            } else {
                                bodies.insert(body.symbol_name.clone(), body);
                            }
                        },
                    }
                }
            }
        }

        assert!(!trace_heads.is_empty(), "no trace heads found!");
        assert!(!trace_tails.is_empty(), "no trace tails found!");
        let markers = SirMarkers { trace_heads, trace_tails };

        Sir {bodies, markers}
    };
}

/// The serialised IR loaded in from disk. One of these structures is generated in the above
/// `lazy_static` and is shared immutably for all threads.
pub struct Sir {
    /// Lets us map a symbol name to a SIR body.
    pub bodies: HashMap<String, Body>,
    // Interesting locations that we need quick access to.
    pub markers: SirMarkers
}

/// Records interesting locations required for trace manipulation.
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
    pub trace_heads: Vec<String>,
    /// Similar to `trace_heads`, functions which stop tracing and whose prefix gets trimmed off
    /// the bottom of traces.
    ///
    /// The compiler is made aware of these locations by the `#[trace_tail]` annotation.
    pub trace_tails: Vec<String>
}

/// A TIR trace is conceptually a straight-line path through the SIR with guarded speculation.
#[derive(Debug)]
pub struct TirTrace {
    ops: Vec<TirOp>
}

impl TirTrace {
    /// Create a TirTrace from a SirTrace, trimming remnants of the code which starts/stops the
    /// tracer. Returns a TIR trace and the bounds the SIR trace was trimmed to, or Err if a symbol
    /// is encountered for which no SIR is available.
    pub fn new<'s>(trace: &'s dyn SirTrace) -> Result<Self, InvalidTraceError> {
        let mut ops = Vec::new();
        let mut itr = trace.into_iter().peekable();
        let mut rename_map: HashMap<Local, Vec<Local>> = HashMap::new();
        let mut var_len = 1;
        let mut cur_call_args: Vec<u32> = Vec::new();
        while let Some(loc) = itr.next() {
            let body = match SIR.bodies.get(&loc.symbol_name) {
                Some(b) => b,
                None => {
                    return Err(InvalidTraceError::no_sir(&loc.symbol_name));
                }
            };

            // When adding statements to the trace, we clone them (rather than referencing the
            // statements in the SIR) so that we have the freedom to mutate them later.
            let user_bb_idx_usize = usize::try_from(loc.bb_idx).unwrap();
            // When converting the SIR trace into a TIR trace we turn it into SSA form by alpha
            // renaming all `Local`s and storing a mapping from old to new name in `rename_map`.
            // This is a bit tricky when it comes to inlined function calls, which assume arguments
            // start at `Local(1)`, which may have already been used. We thus need to store
            // multiple levels in `rename_map` and have an additional variable `cur_call_args`,
            // which provides information about which `Local`s we need to remove again from
            // `rename_map` once we leave the inlined function call again (since we won't get
            // `StorageDead` statements for arguments).
            for stmt in body.blocks[user_bb_idx_usize].stmts.iter() {
                let op = match stmt {
                    Statement::StorageLive(local) => {
                        let newlocal = Local(var_len);
                        match rename_map.get_mut(local) {
                            Some(v) => v.push(newlocal),
                            None => {
                                rename_map.insert(local.clone(), vec![newlocal]);
                            }
                        };
                        var_len += 1;
                        Statement::StorageLive(newlocal)
                    }
                    Statement::StorageDead(local) => {
                        if let Some(v) = rename_map.get_mut(local) {
                            let l = v.pop();
                            Statement::StorageDead(l.unwrap())
                        } else {
                            stmt.clone()
                        }
                    }
                    Statement::Assign(place, rvalue) => {
                        let newplace = TirTrace::rename_place(&rename_map, &place);
                        let newrvalue = TirTrace::rename_rvalue(&rename_map, &rvalue);
                        Statement::Assign(newplace, newrvalue)
                    }
                    Statement::Call(_, _, _) => {
                        // Statement::Call is a terminator and thus should never appear here.
                        unreachable!();
                    }
                    Statement::Nop => stmt.clone(),
                    Statement::Enter(_, _, _) => unreachable!(),
                    Statement::Leave => stmt.clone(),
                    Statement::Unimplemented(_) => stmt.clone()
                };
                ops.push(TirOp::Statement(op));
            }

            match &body.blocks[user_bb_idx_usize].term {
                Terminator::Call {
                    operand: op,
                    args,
                    destination: dest
                } => {
                    if let Some(callee_sym) = op.symbol() {
                        // We know the symbol name of the callee at least.
                        let op = if SIR.bodies.contains_key(callee_sym) {
                            // We have SIR for the callee, so it will appear inlined in the trace
                            // and we only need to emit Enter/Leave statements.

                            // Rename the destination if there is one.
                            let newdest = dest.as_ref().map(|(ret_val, _ret_bb)| {
                                TirTrace::rename_place(&rename_map, &ret_val)
                            });
                            // Rename all `Local`s within the arguments.
                            let mut newargs = Vec::new();
                            for (i, op) in args.iter().enumerate() {
                                newargs.push(TirTrace::rename_operand(&rename_map, &op));
                                let oldlocal = Local(i as u32 + 1);
                                let newlocal = Local(var_len);
                                if let Some(v) = rename_map.get_mut(&oldlocal) {
                                    v.push(newlocal);
                                } else {
                                    rename_map.insert(oldlocal, vec![newlocal]);
                                }
                                var_len += 1;
                            }
                            cur_call_args.push(newargs.len() as u32);
                            // FIXME It seems that calls always have a destination despite it being
                            // an `Option`. If this is not always the case, we may want add the
                            // `Local` offset (`var_len`) to this statement so we can assign the
                            // arguments to the correct `Local`s during trace compilation.
                            assert!(newdest.is_some());
                            TirOp::Statement(Statement::Enter(op.clone(), newargs, newdest))
                        } else {
                            // We have a symbol name but no SIR. Without SIR the callee can't
                            // appear inlined in the trace, so we should emit a native call to the
                            // symbol instead.
                            TirOp::Statement(Statement::Call(
                                op.clone(),
                                args.to_vec(),
                                dest.as_ref().map(|(ret_val, _ret_bb)| ret_val.clone())
                            ))
                        };
                        ops.push(op);
                    } else {
                        todo!("Unknown callee encountered");
                    }
                }
                Terminator::Return => {
                    // After leaving an inlined function call we need to clean up any renaming
                    // mappings we have added manually, because we don't get `StorageDead`
                    // statements for call arguments. Which mappings we need to remove depends on
                    // the number of arguments the function call had, which we keep track of in
                    // `cur_call_args`.
                    let call_args = cur_call_args.pop().unwrap();
                    for i in 1..(call_args + 1u32) {
                        rename_map.get_mut(&Local(i)).unwrap().pop();
                    }
                    ops.push(TirOp::Statement(Statement::Leave))
                }
                _ => {}
            }

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
                    let next_blk = itr.peek().expect("no block to peek at").bb_idx;
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

    fn rename_rvalue(rename_map: &HashMap<Local, Vec<Local>>, rvalue: &Rvalue) -> Rvalue {
        match rvalue {
            Rvalue::Use(op) => {
                let newop = TirTrace::rename_operand(rename_map, op);
                Rvalue::Use(newop)
            }
            Rvalue::BinaryOp(binop, op1, op2) => {
                let newop1 = TirTrace::rename_operand(rename_map, op1);
                let newop2 = TirTrace::rename_operand(rename_map, op2);
                Rvalue::BinaryOp(binop.clone(), newop1, newop2)
            }
            Rvalue::CheckedBinaryOp(binop, op1, op2) => {
                let newop1 = TirTrace::rename_operand(rename_map, op1);
                let newop2 = TirTrace::rename_operand(rename_map, op2);
                Rvalue::CheckedBinaryOp(binop.clone(), newop1, newop2)
            }
            Rvalue::Unimplemented(_) => rvalue.clone()
        }
    }
    fn rename_operand(rename_map: &HashMap<Local, Vec<Local>>, operand: &Operand) -> Operand {
        match operand {
            Operand::Place(p) => Operand::Place(TirTrace::rename_place(rename_map, p)),
            Operand::Constant(_) => operand.clone()
        }
    }

    fn rename_place(rename_map: &HashMap<Local, Vec<Local>>, place: &Place) -> Place {
        // In the future there should always be a mapping for any local in the trace.
        // Unfortunately, since we are still getting remnants of the trace header and tail in our
        // trace, this is not always the case. So for now print warnings and don't attempt to
        // rename those locals.
        match rename_map.get(&place.local) {
            Some(v) => match v.last() {
                Some(l) => {
                    let mut p = place.clone();
                    p.local = l.clone();
                    p
                }
                None => {
                    eprintln!("warning: mapping for {} already empty", &place.local);
                    place.clone()
                }
            },
            None => {
                if &place.local != &Local(0) {
                    // Local(0) is used for function returns and is thus never
                    // renamed, so we don't need to print a warning for it.
                    eprintln!("warning: could not find mapping for {}", &place.local);
                }
                place.clone()
            }
        }
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

impl Display for TirTrace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[Start TIR Trace]")?;
        for op in &self.ops {
            writeln!(f, "  {}", op)?;
        }
        writeln!(f, "[End TIR Trace]")
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

    #[ignore] // FIXME Fails becuase we have not properly populated terminators yet.
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
