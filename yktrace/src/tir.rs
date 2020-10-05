//! Conceptually this module takes an ordered collection of SIR block locations and converts it
//! into a Tracing IR (TIR) Trace using the SIR found in the `.yk_sir` section of the currently
//! running executable.

use super::SirTrace;
use crate::{
    errors::InvalidTraceError,
    sir::{Sir, SirTraceIterator}
};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt::{self, Display}
};
pub use ykpack::{
    BinOp, CallOperand, Constant, ConstantInt, Local, LocalDecl, LocalIndex, Operand, Place,
    PlaceBase, Projection, Rvalue, SignedInt, Statement, Terminator, UnsignedInt
};

/// A TIR trace is conceptually a straight-line path through the SIR with guarded speculation.
#[derive(Debug)]
pub struct TirTrace<'a> {
    ops: Vec<TirOp>,
    trace_inputs_local: Option<Local>,
    /// Maps each local variable to its declaration, including type.
    pub local_decls: HashMap<Local, LocalDecl>,
    pub addr_map: HashMap<String, u64>,
    sir: &'a Sir
}

impl<'a> TirTrace<'a> {
    /// Create a TirTrace from a SirTrace, trimming remnants of the code which starts/stops the
    /// tracer. Returns a TIR trace and the bounds the SIR trace was trimmed to, or Err if a symbol
    /// is encountered for which no SIR is available.
    pub fn new<'s>(sir: &'a Sir, trace: &'s dyn SirTrace) -> Result<Self, InvalidTraceError> {
        let mut ops = Vec::new();
        let mut itr = SirTraceIterator::new(sir, trace).peekable();
        let mut rnm = VarRenamer::new();
        let mut trace_inputs_local: Option<Local> = None;
        // Symbol name of the function currently being ignored during tracing.
        let mut ignore: Option<String> = None;
        // Maps symbol names to their virtual addresses.
        let mut addr_map: HashMap<String, u64> = HashMap::new();

        // As we compile, we are going to check the define-use (DU) chain of our local
        // variables. No local should be used without first being defined. If that happens it's
        // likely that the user used a variable from outside the scope of the trace without
        // introducing it via `trace_locals()`.
        let mut defined_locals: HashSet<Local> = HashSet::new();
        let mut def_sites: HashMap<Local, usize> = HashMap::new();
        let mut last_use_sites = HashMap::new();

        let mut update_defined_locals =
            |renamer: &mut VarRenamer, op: &Statement, op_idx: usize| {
                // Locals reported by `maybe_defined_locals()` are only defined if they are not already
                // defined.
                //
                // FIXME: Note that we are unable to detect variables which are defined outside of the
                // traced code and which are not introduced as trace inputs. The user should not do
                // this, but it would be nice to detect that somehow and panic.
                let newly_defined = op
                    .maybe_defined_locals()
                    .iter()
                    .filter_map(|l| {
                        if !defined_locals.contains(l) {
                            Some(*l)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Local>>();
                defined_locals.extend(&newly_defined);
                for d in newly_defined {
                    def_sites.insert(d, op_idx);
                }

                for lcl in op.used_locals() {
                    // The trace inputs local is regarded as being live for the whole trace.
                    if let Some(til) = renamer.trace_inputs_local {
                        if lcl == til {
                            continue;
                        }
                    }
                    if !defined_locals.contains(&lcl) {
                        panic!("undefined local: {}", lcl);
                    }
                    last_use_sites.insert(lcl, op_idx);
                }
            };

        while let Some(loc) = itr.next() {
            let body = match sir.bodies.get(&loc.symbol_name) {
                Some(b) => b,
                None => {
                    return Err(InvalidTraceError::no_sir(&loc.symbol_name));
                }
            };

            // Store trace inputs local and forward it to the TIR trace.
            trace_inputs_local = body.trace_inputs_local;

            // Initialise VarRenamer's accumulator (and thus also set the first offset) to the
            // traces most outer number of locals.
            rnm.init_acc(body.local_decls.len());

            // When adding statements to the trace, we clone them (rather than referencing the
            // statements in the SIR) so that we have the freedom to mutate them later.
            let user_bb_idx_usize = usize::try_from(loc.bb_idx).unwrap();

            // When we see the first block of a SirFunc, store its virtual address so we can turn
            // this function into a `Call` if the user decides not to trace it.
            let addr = &loc.addr;
            if user_bb_idx_usize == 0 {
                addr_map.insert(loc.symbol_name.to_string(), addr.unwrap());
            }

            // If a function was annotated with `do_not_trace`, skip all instructions within it as
            // well. FIXME: recursion.
            if let Some(sym) = &ignore {
                if sym == &loc.symbol_name {
                    match &body.blocks[user_bb_idx_usize].term {
                        Terminator::Return => {
                            ignore = None;
                        }
                        _ => {}
                    }
                }
                continue;
            }

            // When converting the SIR trace into a TIR trace we alpha-rename the `Local`s from
            // inlined functions by adding an offset to each. This offset is derived from the
            // number of assigned variables in the functions outer context. For example, if a
            // function `bar` is inlined into a function `foo`, and `foo` used 5 variables, then
            // all variables in `bar` are offset by 5.
            for stmt in body.blocks[user_bb_idx_usize].stmts.iter() {
                // If the statement references a thread tracer local then discard the statement.
                let mut skip = false;
                for lcl in stmt.used_locals() {
                    if sir
                        .is_thread_tracer_ty(&body.local_decls[usize::try_from(lcl.0).unwrap()].ty)
                    {
                        skip = true;
                        break;
                    }
                }
                if skip {
                    continue;
                }

                let op = match stmt {
                    // StorageDead can't appear in SIR, only TIR.
                    Statement::StorageDead(_) => unreachable!(),
                    Statement::Assign(place, rvalue) => {
                        let newplace = rnm.rename_place(&place, body);
                        let newrvalue = rnm.rename_rvalue(&rvalue, body);
                        Statement::Assign(newplace, newrvalue)
                    }
                    Statement::Nop => stmt.clone(),
                    Statement::Unimplemented(_) => stmt.clone(),
                    // The following statements kinds are specific to TIR and cannot appear in SIR.
                    Statement::Call(..) | Statement::Enter(..) | Statement::Leave => unreachable!()
                };

                update_defined_locals(&mut rnm, &op, ops.len());
                ops.push(TirOp::Statement(op));
            }

            let stmt = match &body.blocks[user_bb_idx_usize].term {
                Terminator::Call {
                    operand: op,
                    args,
                    destination: dest
                } => {
                    if let Some(callee_sym) = op.symbol() {
                        if let Some(callee_body) = sir.bodies.get(callee_sym) {
                            if callee_body.flags & ykpack::bodyflags::TRACE_TAIL != 0 {
                                break;
                            }
                        }
                    }

                    // Rename the return value.
                    //
                    // FIXME It seems that calls always have a destination despite the field being
                    // `Option`. If this is not always the case, we may want add the `Local` offset
                    // (`var_len`) to this statement so we can assign the arguments to the correct
                    // `Local`s during trace compilation.
                    let ret_val = dest
                        .as_ref()
                        .map(|(ret_val, _)| rnm.rename_place(&ret_val, body))
                        .unwrap();

                    if let Some(callee_sym) = op.symbol() {
                        // We know the symbol name of the callee at least.
                        // Rename all `Local`s within the arguments.
                        let newargs = rnm.rename_args(&args, body);
                        let op = if let Some(callbody) = sir.bodies.get(callee_sym) {
                            // We have SIR for the callee, so it will appear inlined in the trace
                            // and we only need to emit Enter/Leave statements.

                            // If the function has been annotated with do_not_trace, turn it into a
                            // call.
                            if callbody.flags & ykpack::bodyflags::DO_NOT_TRACE != 0 {
                                ignore = Some(callee_sym.to_string());
                                Statement::Call(op.clone(), newargs, Some(ret_val))
                            } else {
                                // Inform VarRenamer about this function's offset, which is equal to the
                                // number of variables assigned in the outer body.
                                rnm.enter(callbody.local_decls.len(), ret_val.clone());

                                // Ensure the callee's arguments get TIR local decls. This is required
                                // because arguments are implicitly live at the start of each function,
                                // and we usually instantiate local decls when we see a StorageLive.
                                //
                                // This must happen after rnm.enter() so that self.offset is up-to-date.
                                for lidx in 0..newargs.len() {
                                    let lidx = lidx + 1; // Skipping the return local.
                                    let decl =
                                        &callbody.local_decls[usize::try_from(lidx).unwrap()];
                                    rnm.local_decls.insert(
                                        Local(rnm.offset + u32::try_from(lidx).unwrap()),
                                        decl.clone()
                                    );
                                }

                                Statement::Enter(op.clone(), newargs, Some(ret_val), rnm.offset())
                            }
                        } else {
                            // We have a symbol name but no SIR. Without SIR the callee can't
                            // appear inlined in the trace, so we should emit a native call to the
                            // symbol instead.
                            Statement::Call(op.clone(), newargs, Some(ret_val))
                        };
                        Some(op)
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
                    rnm.leave();
                    Some(Statement::Leave)
                }
                _ => None
            };
            if let Some(stmt) = stmt {
                update_defined_locals(&mut rnm, &stmt, ops.len());
                ops.push(TirOp::Statement(stmt));
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

        let mut local_decls = rnm.done();

        // Insert `StorageDead` statements after the last use of each local variable. We process
        // the locals in reverse order of death site, so that inserting a statement cannot not skew
        // the indices for subsequent insertions.
        let mut deads = last_use_sites.iter().collect::<Vec<(&Local, &usize)>>();
        deads.sort_by(|a, b| b.1.cmp(a.1));
        for (local, idx) in deads {
            // The trace inputs local is always live.
            if trace_inputs_local.is_none() || *local != trace_inputs_local.unwrap() {
                if def_sites[local] == *idx && !ops[*idx].may_have_side_effects() {
                    // If a defined local is never used, and the statement that defines it isn't
                    // side-effecting, then we can remove the statement and local's decl entirely.
                    //
                    // FIXME This is not perfect. Consider `x.0 = 0; x.1 = 1` and then x is not
                    // used after. The first operation will be seen to define `x`, the second will
                    // be seen as a use of `x`, and thus neither of these statements will be
                    // removed.
                    ops.remove(*idx);
                    let prev = local_decls.remove(&local);
                    debug_assert!(prev.is_some());
                } else {
                    ops.insert(
                        *idx + 1,
                        TirOp::Statement(ykpack::Statement::StorageDead(local.clone()))
                    );
                }
            }
        }

        Ok(Self {
            ops,
            trace_inputs_local,
            local_decls,
            addr_map,
            sir
        })
    }

    /// Return the TIR operation at index `idx` in the trace.
    /// The index must not be out of bounds.
    pub unsafe fn op(&self, idx: usize) -> &TirOp {
        debug_assert!(idx <= self.ops.len() - 1, "bogus trace index");
        &self.ops.get_unchecked(idx)
    }

    pub fn inputs(&self) -> &Option<Local> {
        &self.trace_inputs_local
    }

    /// Return the length of the trace measure in operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }
}

struct VarRenamer {
    /// Stores the offset before entering an inlined call, so that the correct offset can be
    /// restored again after leaving that call.
    stack: Vec<u32>,
    /// Current offset used to rename variables.
    offset: u32,
    /// Accumulator keeping track of total number of variables used. Needed to use different
    /// offsets for consecutive inlined function calls.
    acc: Option<u32>,
    /// Stores the return variables of inlined function calls. Used to replace `$0` during
    /// renaming.
    returns: Vec<Place>,
    /// Maps a renamed local to its local declaration.
    local_decls: HashMap<Local, LocalDecl>,
    /// The renamed trace input local, if it is known yet.
    trace_inputs_local: Option<Local>
}

impl VarRenamer {
    fn new() -> Self {
        VarRenamer {
            stack: vec![0],
            offset: 0,
            acc: None,
            returns: Vec::new(),
            local_decls: HashMap::new(),
            trace_inputs_local: None
        }
    }

    /// Finalises the renamer, returning the local decls.
    fn done(self) -> HashMap<Local, LocalDecl> {
        self.local_decls
    }

    fn offset(&self) -> u32 {
        self.offset
    }

    fn init_acc(&mut self, num_locals: usize) {
        if self.acc.is_none() {
            self.acc.replace(num_locals as u32);
        }
    }

    fn enter(&mut self, num_locals: usize, dest: Place) {
        // When entering an inlined function call set the offset to the current accumulator. Then
        // increment the accumulator by the number of locals in the current function. Also add the
        // offset to the stack, so we can restore it once we leave the inlined function call again.
        self.offset = self.acc.unwrap();
        self.stack.push(self.offset);
        match self.acc.as_mut() {
            Some(v) => *v += num_locals as u32,
            None => {}
        }
        self.returns.push(dest);
    }

    fn leave(&mut self) {
        // When we leave an inlined function call, we pop the previous offset from the stack,
        // reverting the offset to what it was before the function was entered.
        self.stack.pop();
        self.returns.pop();
        if let Some(v) = self.stack.last() {
            self.offset = *v;
        } else {
            panic!("Unbalanced enter/leave statements!")
        }
    }

    fn rename_args(&mut self, args: &Vec<Operand>, body: &ykpack::Body) -> Vec<Operand> {
        args.iter()
            .map(|op| self.rename_operand(&op, body))
            .collect()
    }

    fn rename_rvalue(&mut self, rvalue: &Rvalue, body: &ykpack::Body) -> Rvalue {
        match rvalue {
            Rvalue::Use(op) => {
                let newop = self.rename_operand(op, body);
                Rvalue::Use(newop)
            }
            Rvalue::BinaryOp(binop, op1, op2) => {
                let newop1 = self.rename_operand(op1, body);
                let newop2 = self.rename_operand(op2, body);
                Rvalue::BinaryOp(binop.clone(), newop1, newop2)
            }
            Rvalue::CheckedBinaryOp(binop, op1, op2) => {
                let newop1 = self.rename_operand(op1, body);
                let newop2 = self.rename_operand(op2, body);
                Rvalue::CheckedBinaryOp(binop.clone(), newop1, newop2)
            }
            Rvalue::Ref(place) => {
                let newplace = self.rename_place(place, body);
                Rvalue::Ref(newplace)
            }
            Rvalue::Len(place) => {
                let newplace = self.rename_place(place, body);
                Rvalue::Len(newplace)
            }
            Rvalue::Unimplemented(_) => rvalue.clone()
        }
    }

    fn rename_operand(&mut self, operand: &Operand, body: &ykpack::Body) -> Operand {
        match operand {
            Operand::Place(p) => Operand::Place(self.rename_place(p, body)),
            Operand::Constant(_) => operand.clone()
        }
    }

    fn rename_place(&mut self, place: &Place, body: &ykpack::Body) -> Place {
        let newproj = self.rename_projection(&place.projection, body);

        if &place.local == &Local(0) {
            // Replace the default return variable $0 with the variable in the outer context where
            // the return value will end up after leaving the function. This saves us an
            // instruction when we compile the trace.
            let mut ret = if let Some(v) = self.returns.last() {
                v.clone()
            } else {
                panic!("Expected return value!")
            };

            self.local_decls.insert(
                ret.local,
                body.local_decls[usize::try_from(place.local.0).unwrap()].clone()
            );
            ret.projection = newproj;
            ret
        } else {
            let mut p = place.clone();
            p.local = self.rename_local(&p.local, body);
            p.projection = newproj;
            p
        }
    }

    fn rename_projection(
        &mut self,
        projection: &Vec<Projection>,
        body: &ykpack::Body
    ) -> Vec<Projection> {
        let mut v = Vec::new();
        for p in projection {
            match p {
                Projection::Index(local) => {
                    v.push(Projection::Index(self.rename_local(&local, body)))
                }
                _ => v.push(p.clone())
            }
        }
        v
    }

    fn rename_local(&mut self, local: &Local, body: &ykpack::Body) -> Local {
        let renamed = Local(local.0 + self.offset);
        self.local_decls.insert(
            renamed.clone(),
            body.local_decls[usize::try_from(local.0).unwrap()].clone()
        );

        if let Some(til) = body.trace_inputs_local {
            if local == &til {
                self.trace_inputs_local = Some(renamed);
            }
        }

        renamed
    }
}

impl Display for TirTrace<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "local_decls:")?;
        let mut sort_decls = self
            .local_decls
            .iter()
            .collect::<Vec<(&Local, &LocalDecl)>>();
        sort_decls.sort_by(|l, r| l.0.partial_cmp(r.0).unwrap());
        for (l, dcl) in sort_decls {
            let thread_tracer = if self.sir.is_thread_tracer_ty(&dcl.ty) {
                "[THREAD TRACER] "
            } else {
                ""
            };

            writeln!(
                f,
                "  {}: ({}, {}) => {}{}",
                l,
                dcl.ty.0,
                dcl.ty.1,
                thread_tracer,
                self.sir.ty(&dcl.ty)
            )?;
        }

        writeln!(f, "ops:")?;
        for op in &self.ops {
            writeln!(f, "  {}", op)?;
        }
        Ok(())
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

impl TirOp {
    /// Returns true if the operation may affect locals besides those appearing in the operation.
    fn may_have_side_effects(&self) -> bool {
        if let TirOp::Statement(s) = self {
            s.may_have_side_effects()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TirTrace;
    use crate::{sir::SIR, start_tracing, TracingKind};
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
        let tracer = start_tracing(TracingKind::default());
        let res = black_box(work(black_box(3), black_box(13)));
        let sir_trace = tracer.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        assert_eq!(res, 15);
        assert!(tir_trace.len() > 0);
    }

    #[test]
    #[should_panic]
    fn use_undefined_var() {
        let outside_var = 100;
        let tracer = start_tracing(TracingKind::default());
        let _x = outside_var + 1; // Use of an undefined variable in trace.
        let sir_trace = tracer.stop_tracing().unwrap();
        let _tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
    }
}
