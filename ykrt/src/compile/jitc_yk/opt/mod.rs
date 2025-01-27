// A trace IR optimiser.
//
// The optimiser works in a single forward pass (well, it also does a single backwards pass at the
// end too, but that's only because we can't yet do backwards code generation). As it progresses
// through a trace, it both mutates the trace IR directly and also refines its idea about what
// value an instruction might produce. These two actions are subtly different: mutation is done
// in this module; the refinement of values in the [Analyse] module.

use super::{
    arbbitint::ArbBitInt,
    jit_ir::{
        BinOp, BinOpInst, Const, ConstIdx, DynPtrAddInst, GuardInst, ICmpInst, Inst, InstIdx,
        LoadInst, Module, Operand, Predicate, PtrAddInst, SExtInst, SelectInst, StoreInst,
        TraceKind, Ty, ZExtInst,
    },
};
use crate::compile::CompilationError;
use std::assert_matches::debug_assert_matches;

mod analyse;
mod heapvalues;
mod instll;

use analyse::{Analyse, Value};
use heapvalues::Address;
use instll::InstLinkedList;

struct Opt {
    m: Module,
    an: Analyse,
}

impl Opt {
    fn new(m: Module) -> Self {
        let an = Analyse::new(&m);
        Self { m, an }
    }

    fn opt(mut self) -> Result<Module, CompilationError> {
        let base = self.m.insts_len();
        let peel = match self.m.tracekind() {
            TraceKind::HeaderOnly => {
                #[cfg(not(test))]
                {
                    true
                }

                #[cfg(test)]
                {
                    // Not all tests create "fully valid" traces, in the sense that -- to keep
                    // things simple -- they don't end with `TraceHeaderEnd`. We don't want to peel
                    // such traces, but nor, in testing mode, do we consider them ill-formed.
                    matches!(self.m.inst(self.m.last_inst_idx()), Inst::TraceHeaderEnd)
                }
            }
            // If we hit this case, someone's tried to run the optimiser twice.
            TraceKind::HeaderAndBody => unreachable!(),
            // If this is a sidetrace, we perform optimisations up to, but not including, loop
            // peeling.
            TraceKind::Sidetrace(_) => false,
        };

        // Note that since we will apply loop peeling here, the list of instructions grows as this
        // loop runs. Each instruction we process is (after optimisations were applied), duplicated
        // and copied to the end of the module.
        let mut instll = InstLinkedList::new(&self.m);
        let skipping = self.m.iter_skipping_insts().collect::<Vec<_>>();
        for (iidx, inst) in skipping.into_iter() {
            match inst {
                Inst::TraceHeaderStart => (),
                _ => {
                    self.opt_inst(iidx)?;
                    self.cse(&mut instll, iidx);
                }
            }
        }

        if !peel {
            return Ok(self.m);
        }

        debug_assert_matches!(self.m.tracekind(), TraceKind::HeaderOnly);
        self.m.set_tracekind(TraceKind::HeaderAndBody);

        // Now that we've processed the trace header, duplicate it to create the loop body.
        let mut iidx_map = vec![InstIdx::max(); base];
        let skipping = self.m.iter_skipping_insts().collect::<Vec<_>>();
        for (iidx, inst) in skipping.into_iter() {
            match inst {
                Inst::TraceHeaderStart => {
                    self.m.trace_body_start = self.m.trace_header_start().to_vec();
                    self.m.push(Inst::TraceBodyStart)?;
                    // FIXME: We rely on `dup_and_remap_vars` not being idempotent here.
                    let _ = Inst::TraceBodyStart
                        .dup_and_remap_vars(&mut self.m, |_, op_iidx: InstIdx| {
                            Operand::Var(iidx_map[usize::from(op_iidx)])
                        })?;
                    for (headop, bodyop) in self
                        .m
                        .trace_header_end()
                        .iter()
                        .zip(self.m.trace_body_start())
                    {
                        // Inform the analyser about any constants being passed from the header into
                        // the body.
                        if let Operand::Const(cidx) = headop.unpack(&self.m) {
                            let Operand::Var(op_iidx) = bodyop.unpack(&self.m) else {
                                panic!()
                            };
                            self.an.set_value(&self.m, op_iidx, Value::Const(cidx));
                        }
                    }
                }
                Inst::TraceHeaderEnd => {
                    self.m.trace_body_end = self.m.trace_header_end().to_vec();
                    self.m.push(Inst::TraceBodyEnd)?;
                    // FIXME: We rely on `dup_and_remap_vars` not being idempotent here.
                    let _ = Inst::TraceBodyEnd
                        .dup_and_remap_vars(&mut self.m, |_, op_iidx: InstIdx| {
                            Operand::Var(iidx_map[usize::from(op_iidx)])
                        })?;
                }
                _ => {
                    let c = inst.dup_and_remap_vars(&mut self.m, |_, op_iidx: InstIdx| {
                        Operand::Var(iidx_map[usize::from(op_iidx)])
                    })?;
                    let copy_iidx = self.m.push(c)?;
                    iidx_map[usize::from(iidx)] = copy_iidx;
                }
            }
        }

        // Create a fresh `instll`. Normal CSE in the body (a) can't possibly reference the header
        // (b) the number of instructions in the `instll`-for-the-header is wrong as a result of
        // peeling. So create a fresh `instll`.
        let mut instll = InstLinkedList::new(&self.m);
        let skipping = self
            .m
            .iter_skipping_insts()
            .skip_while(|(_, inst)| !matches!(inst, Inst::TraceBodyStart))
            .collect::<Vec<_>>();
        for (iidx, inst) in skipping.into_iter() {
            match inst {
                Inst::TraceHeaderStart | Inst::TraceHeaderEnd => panic!(),
                Inst::TraceBodyStart => (),
                _ => {
                    self.opt_inst(iidx)?;
                    self.cse(&mut instll, iidx);
                }
            }
        }

        Ok(self.m)
    }

    /// Optimise instruction `iidx`.
    fn opt_inst(&mut self, iidx: InstIdx) -> Result<(), CompilationError> {
        // First rewrite the instruction so that all changes from the analyser are reflected
        // straight away. Note: we deliberately do this before some of the changes below. Most
        // notably we need to call `rewrite` before telling the analyser about a `Guard`: if we
        // swap that order, the guard will pick up the wrong value for operand(s) related to
        // whether the guard succeeds!
        self.rewrite(iidx)?;

        match self.m.inst(iidx) {
            #[cfg(test)]
            Inst::BlackBox(_) => (),
            Inst::Const(_) | Inst::Copy(_) | Inst::Tombstone | Inst::TraceHeaderStart => {
                unreachable!()
            }
            Inst::BinOp(x) => self.opt_binop(iidx, x)?,
            Inst::Call(_) | Inst::IndirectCall(_) => {
                self.an.heap_barrier();
            }
            Inst::DynPtrAdd(x) => self.opt_dynptradd(iidx, x)?,
            Inst::Guard(x) => self.opt_guard(iidx, x)?,
            Inst::ICmp(x) => self.opt_icmp(iidx, x)?,
            Inst::Load(x) => self.opt_load(iidx, x)?,
            Inst::Param(x) => {
                // FIXME: This feels like it should be handled by trace_builder, but we can't
                // do so yet because of https://github.com/ykjit/yk/issues/1435.
                if let yksmp::Location::Constant(v) = self.m.param(x.paramidx()) {
                    let Ty::Integer(bitw) = self.m.type_(x.tyidx()) else {
                        unreachable!()
                    };
                    // `Location::Constant` is a u32
                    assert!(*bitw <= 32);
                    let cidx = self.m.insert_const(Const::Int(
                        x.tyidx(),
                        ArbBitInt::from_u64(*bitw, u64::from(*v)),
                    ))?;
                    self.an.set_value(&self.m, iidx, Value::Const(cidx));
                }
            }
            Inst::PtrAdd(x) => self.opt_ptradd(iidx, x)?,
            Inst::Select(x) => self.opt_select(iidx, x)?,
            Inst::SExt(x) => self.opt_sext(iidx, x)?,
            Inst::Store(x) => self.opt_store(iidx, x)?,
            Inst::ZExt(x) => self.opt_zext(iidx, x)?,
            _ => (),
        };

        Ok(())
    }

    fn opt_binop(&mut self, iidx: InstIdx, inst: BinOpInst) -> Result<(), CompilationError> {
        // Don't forget to add canonicalisations to the `canonicalisation` test!
        match inst.binop() {
            BinOp::Add => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Const(op_cidx), Operand::Var(op_iidx))
                | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    match self.m.const_(op_cidx) {
                        Const::Int(_, x) if x.to_zero_ext_u64().unwrap() == 0 => {
                            // Replace `x + 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                        _ => {
                            // Canonicalise to (Var, Const).
                            self.m.replace(
                                iidx,
                                BinOpInst::new(
                                    Operand::Var(op_iidx),
                                    BinOp::Add,
                                    Operand::Const(op_cidx),
                                )
                                .into(),
                            );
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(_rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            debug_assert_eq!(lhs_tyidx, _rhs_tyidx);
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.wrapping_add(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => todo!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::And => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Const(op_cidx), Operand::Var(op_iidx))
                | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    match self.m.const_(op_cidx) {
                        Const::Int(_, x) if x.to_zero_ext_u64().unwrap() == 0 => {
                            // Replace `x & 0` with `0`.
                            self.m.replace(iidx, Inst::Const(op_cidx));
                        }
                        _ => {
                            // Canonicalise to (Var, Const).
                            self.m.replace(
                                iidx,
                                BinOpInst::new(
                                    Operand::Var(op_iidx),
                                    BinOp::And,
                                    Operand::Const(op_cidx),
                                )
                                .into(),
                            );
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.bitand(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => panic!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::LShr => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    if let Const::Int(_, y) = self.m.const_(op_cidx) {
                        if y.to_zero_ext_u64().unwrap() == 0 {
                            // Replace `x >> 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                    }
                }
                (Operand::Const(op_cidx), Operand::Var(_)) => {
                    if let Const::Int(tyidx, y) = self.m.const_(op_cidx) {
                        if y.to_zero_ext_u64().unwrap() == 0 {
                            // Replace `0 >> x` with `0`.
                            let new_cidx = self.m.insert_const(Const::Int(
                                *tyidx,
                                ArbBitInt::from_u64(y.bitw(), 0),
                            ))?;
                            self.m.replace(iidx, Inst::Const(new_cidx));
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(_rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            debug_assert_eq!(lhs_tyidx, _rhs_tyidx);
                            // If checked_shr fails, we've encountered LLVM poison and can
                            // choose any value.
                            let shr = lhs
                                .checked_shr(rhs.to_zero_ext_u32().unwrap())
                                .unwrap_or_else(|| ArbBitInt::all_bits_set(lhs.bitw()));
                            let cidx = self.m.insert_const(Const::Int(*lhs_tyidx, shr))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => panic!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::Mul => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Const(op_cidx), Operand::Var(op_iidx))
                | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    match self.m.const_(op_cidx) {
                        Const::Int(_, y) if y.to_zero_ext_u64().unwrap() == 0 => {
                            // Replace `x * 0` with `0`.
                            self.m.replace(iidx, Inst::Const(op_cidx));
                        }
                        Const::Int(_, y) if y.to_zero_ext_u64().unwrap() == 1 => {
                            // Replace `x * 1` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                        Const::Int(tyidx, y) if y.to_zero_ext_u64().unwrap().is_power_of_two() => {
                            // Replace `x * y` with `x << ...`.
                            let shl = u64::from(y.to_zero_ext_u64().unwrap().ilog2());
                            let shl_op = Operand::Const(self.m.insert_const(Const::Int(
                                *tyidx,
                                ArbBitInt::from_u64(y.bitw(), shl),
                            ))?);
                            let new_inst =
                                BinOpInst::new(Operand::Var(op_iidx), BinOp::Shl, shl_op).into();
                            self.m.replace(iidx, new_inst);
                        }
                        _ => {
                            // Canonicalise to (Var, Const).
                            self.m.replace(
                                iidx,
                                BinOpInst::new(
                                    Operand::Var(op_iidx),
                                    BinOp::Mul,
                                    Operand::Const(op_cidx),
                                )
                                .into(),
                            );
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(_rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, _rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.wrapping_mul(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => todo!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::Or => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Const(op_cidx), Operand::Var(op_iidx))
                | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    match self.m.const_(op_cidx) {
                        Const::Int(_, y) if y.to_zero_ext_u64().unwrap() == 0 => {
                            // Replace `x | 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                        _ => {
                            // Canonicalise to (Var, Const).
                            self.m.replace(
                                iidx,
                                BinOpInst::new(
                                    Operand::Var(op_iidx),
                                    BinOp::Or,
                                    Operand::Const(op_cidx),
                                )
                                .into(),
                            );
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.bitor(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => panic!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::Shl => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    if let Const::Int(_, y) = self.m.const_(op_cidx) {
                        if y.to_zero_ext_u64().unwrap() == 0 {
                            // Replace `x << 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                    }
                }
                (Operand::Const(op_cidx), Operand::Var(_)) => {
                    if let Const::Int(tyidx, y) = self.m.const_(op_cidx) {
                        if y.to_zero_ext_u64().unwrap() == 0 {
                            // Replace `0 << x` with `0`.
                            let new_cidx = self.m.insert_const(Const::Int(
                                *tyidx,
                                ArbBitInt::from_u64(y.bitw(), 0),
                            ))?;
                            self.m.replace(iidx, Inst::Const(new_cidx));
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            // If checked_shl fails, we've encountered LLVM poison and can
                            // choose any value.
                            let shl = lhs
                                .checked_shl(rhs.to_zero_ext_u32().unwrap())
                                .unwrap_or_else(|| ArbBitInt::all_bits_set(lhs.bitw()));
                            let cidx = self.m.insert_const(Const::Int(*lhs_tyidx, shl))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => panic!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::Sub => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    if let Const::Int(_, y) = self.m.const_(op_cidx) {
                        if y.to_zero_ext_u64().unwrap() == 0 {
                            // Replace `x - 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.wrapping_sub(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => todo!(),
                    }
                }
                (Operand::Const(_), Operand::Var(_)) | (Operand::Var(_), Operand::Var(_)) => (),
            },
            BinOp::Xor => match (
                self.an.op_map(&self.m, inst.lhs(&self.m)),
                self.an.op_map(&self.m, inst.rhs(&self.m)),
            ) {
                (Operand::Const(op_cidx), Operand::Var(op_iidx))
                | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                    match self.m.const_(op_cidx) {
                        Const::Int(_, y) if y.to_zero_ext_u64().unwrap() == 0 => {
                            // Replace `x ^ 0` with `x`.
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                        _ => {
                            // Canonicalise to (Var, Const).
                            self.m.replace(
                                iidx,
                                BinOpInst::new(
                                    Operand::Var(op_iidx),
                                    BinOp::Xor,
                                    Operand::Const(op_cidx),
                                )
                                .into(),
                            );
                        }
                    }
                }
                (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                    match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            let cidx = self
                                .m
                                .insert_const(Const::Int(*lhs_tyidx, lhs.bitxor(rhs)))?;
                            self.m.replace(iidx, Inst::Const(cidx));
                        }
                        _ => panic!(),
                    }
                }
                (Operand::Var(_), Operand::Var(_)) => (),
            },
            _ => {
                if let (Operand::Const(_), Operand::Const(_)) = (
                    self.an.op_map(&self.m, inst.lhs(&self.m)),
                    self.an.op_map(&self.m, inst.rhs(&self.m)),
                ) {
                    todo!("{:?}", inst.binop());
                }
            }
        }

        Ok(())
    }

    fn opt_dynptradd(
        &mut self,
        iidx: InstIdx,
        inst: DynPtrAddInst,
    ) -> Result<(), CompilationError> {
        if let Operand::Const(cidx) = self.an.op_map(&self.m, inst.num_elems(&self.m)) {
            let Const::Int(_, v) = self.m.const_(cidx) else {
                panic!()
            };
            // DynPtrAdd indices are signed, so we have to be careful to interpret the
            // constant as such.
            let v = v.to_sign_ext_i64().unwrap();
            // LLVM IR allows `off` to be an `i64` but our IR currently allows only an
            // `i32`. On that basis, we can hit our limits before the program has
            // itself hit UB, at which point we can't go any further.
            let off = i32::try_from(v)
                .map_err(|_| ())
                .and_then(|v| v.checked_mul(i32::from(inst.elem_size())).ok_or(()))
                .map_err(|_| {
                    CompilationError::LimitExceeded(
                        "`DynPtrAdd` offset exceeded `i32` bounds".into(),
                    )
                })?;
            if off == 0 {
                match self.an.op_map(&self.m, inst.ptr(&self.m)) {
                    Operand::Var(op_iidx) => self.m.replace(iidx, Inst::Copy(op_iidx)),
                    Operand::Const(cidx) => self.m.replace(iidx, Inst::Const(cidx)),
                }
            } else {
                self.m
                    .replace(iidx, Inst::PtrAdd(PtrAddInst::new(inst.ptr(&self.m), off)));
            }
        }

        Ok(())
    }

    fn opt_guard(&mut self, iidx: InstIdx, inst: GuardInst) -> Result<(), CompilationError> {
        if let Operand::Const(cidx) = self.an.op_map(&self.m, inst.cond(&self.m)) {
            // A guard that references a constant is, by definition, not needed and
            // doesn't affect future analyses.
            let Const::Int(_, v) = self.m.const_(cidx) else {
                panic!()
            };
            assert_eq!(v.bitw(), 1);
            assert!(
                (inst.expect() && v.to_zero_ext_u8().unwrap() == 1)
                    || (!inst.expect() && v.to_zero_ext_u8().unwrap() == 0)
            );
            self.m.replace(iidx, Inst::Tombstone);
        } else {
            self.an.guard(&self.m, inst);
        }

        Ok(())
    }

    fn opt_icmp(&mut self, iidx: InstIdx, inst: ICmpInst) -> Result<(), CompilationError> {
        let lhs = self.an.op_map(&self.m, inst.lhs(&self.m));
        let pred = inst.predicate();
        let rhs = self.an.op_map(&self.m, inst.rhs(&self.m));
        match (&lhs, &rhs) {
            (&Operand::Const(lhs_cidx), &Operand::Const(rhs_cidx)) => {
                self.opt_icmp_both_const(iidx, lhs_cidx, pred, rhs_cidx)
            }
            (&Operand::Var(_), &Operand::Const(_)) | (&Operand::Const(_), &Operand::Var(_)) => (),
            (&Operand::Var(_), &Operand::Var(_)) => (),
        }

        Ok(())
    }

    /// Optimise an `ICmp` if both sides are constants. It is required that [Opt::op_map] has been
    /// called on both `lhs` and `rhs` to obtain the `ConstIdx`s.
    fn opt_icmp_both_const(
        &mut self,
        iidx: InstIdx,
        lhs: ConstIdx,
        pred: Predicate,
        rhs: ConstIdx,
    ) {
        let lhs_c = self.m.const_(lhs);
        let rhs_c = self.m.const_(rhs);
        match (lhs_c, rhs_c) {
            (Const::Float(..), Const::Float(..)) => (),
            (Const::Int(_, x), Const::Int(_, y)) => {
                debug_assert_eq!(x.bitw(), y.bitw());
                // Constant fold comparisons of simple integers.
                let r = match pred {
                    Predicate::Equal => {
                        x.to_zero_ext_u64().unwrap() == y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::NotEqual => {
                        x.to_zero_ext_u64().unwrap() != y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::UnsignedGreater => {
                        x.to_zero_ext_u64().unwrap() > y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::UnsignedGreaterEqual => {
                        x.to_zero_ext_u64().unwrap() >= y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::UnsignedLess => {
                        x.to_zero_ext_u64().unwrap() < y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::UnsignedLessEqual => {
                        x.to_zero_ext_u64().unwrap() <= y.to_zero_ext_u64().unwrap()
                    }
                    Predicate::SignedGreater => {
                        (x.to_sign_ext_i64().unwrap()) > (y.to_sign_ext_i64().unwrap())
                    }
                    Predicate::SignedGreaterEqual => {
                        (x.to_sign_ext_i64().unwrap()) >= (y.to_sign_ext_i64().unwrap())
                    }
                    Predicate::SignedLess => {
                        (x.to_sign_ext_i64().unwrap()) < (y.to_sign_ext_i64().unwrap())
                    }
                    Predicate::SignedLessEqual => {
                        (x.to_sign_ext_i64().unwrap()) <= (y.to_sign_ext_i64().unwrap())
                    }
                };

                self.m.replace(
                    iidx,
                    Inst::Const(if r {
                        self.m.true_constidx()
                    } else {
                        self.m.false_constidx()
                    }),
                );
            }
            (Const::Ptr(x), Const::Ptr(y)) => {
                // Constant fold comparisons of pointers.
                let x = *x;
                let y = *y;
                let r = match pred {
                    Predicate::Equal => x == y,
                    Predicate::NotEqual => x != y,
                    Predicate::UnsignedGreater => x > y,
                    Predicate::UnsignedGreaterEqual => x >= y,
                    Predicate::UnsignedLess => x < y,
                    Predicate::UnsignedLessEqual => x <= y,
                    Predicate::SignedGreater
                    | Predicate::SignedGreaterEqual
                    | Predicate::SignedLess
                    | Predicate::SignedLessEqual => unreachable!(),
                };

                self.m.replace(
                    iidx,
                    Inst::Const(if r {
                        self.m.true_constidx()
                    } else {
                        self.m.false_constidx()
                    }),
                );
            }
            _ => unreachable!(),
        }
    }

    fn opt_load(&mut self, iidx: InstIdx, inst: LoadInst) -> Result<(), CompilationError> {
        if !inst.is_volatile() {
            let tgt = self.an.op_map(&self.m, inst.ptr(&self.m));
            let bytesize = Inst::Load(inst).def_byte_size(&self.m);
            match self.an.heapvalue(
                &self.m,
                Address::from_operand(&self.m, tgt.clone()),
                bytesize,
            ) {
                None => {
                    self.an.push_heap_load(
                        &self.m,
                        Address::from_operand(&self.m, tgt),
                        Operand::Var(iidx),
                    );
                }
                Some(op) => {
                    let (r_tyidx, r) = match op {
                        Operand::Var(op_iidx) => (
                            self.m.inst_nocopy(op_iidx).unwrap().tyidx(&self.m),
                            Inst::Copy(op_iidx),
                        ),
                        Operand::Const(op_cidx) => {
                            (self.m.const_(op_cidx).tyidx(&self.m), Inst::Const(op_cidx))
                        }
                    };
                    // OPT: If the next check fails, it means that type punning has
                    // occurred. For example, on x64, we may have last seen a load of an
                    // `i64`, but now that pointer is being used for a `ptr`. If the
                    // program is well-formed, we can always deal with such punning. For
                    // now, we simply don't optimise away the load.
                    if self.m.inst_nocopy(iidx).unwrap().tyidx(&self.m) == r_tyidx {
                        self.m.replace(iidx, r);
                    }
                }
            };
        }
        Ok(())
    }

    fn opt_ptradd(&mut self, iidx: InstIdx, inst: PtrAddInst) -> Result<(), CompilationError> {
        match self.an.op_map(&self.m, inst.ptr(&self.m)) {
            Operand::Const(_) => todo!(),
            Operand::Var(op_iidx) => {
                if inst.off() == 0 {
                    self.m.replace(iidx, Inst::Copy(op_iidx));
                }
            }
        }

        Ok(())
    }

    fn opt_select(&mut self, iidx: InstIdx, inst: SelectInst) -> Result<(), CompilationError> {
        if let Operand::Const(cidx) = self.an.op_map(&self.m, inst.cond(&self.m)) {
            let Const::Int(_, v) = self.m.const_(cidx) else {
                panic!()
            };
            debug_assert_eq!(v.bitw(), 1);
            let op = match v.to_zero_ext_u8().unwrap() {
                0 => inst.falseval(&self.m),
                1 => inst.trueval(&self.m),
                _ => panic!(),
            };
            self.m.replace_with_op(iidx, op);
        } else if self.an.op_map(&self.m, inst.trueval(&self.m))
            == self.an.op_map(&self.m, inst.falseval(&self.m))
        {
            // Both true and false operands are equal, so it doesn't matter which we use.
            self.m.replace_with_op(iidx, inst.trueval(&self.m));
        }
        Ok(())
    }

    fn opt_sext(&mut self, iidx: InstIdx, inst: SExtInst) -> Result<(), CompilationError> {
        if let Operand::Const(cidx) = self.an.op_map(&self.m, inst.val(&self.m)) {
            let Const::Int(_, src_val) = self.m.const_(cidx) else {
                unreachable!()
            };
            let dst_ty = self.m.type_(inst.dest_tyidx());
            let Ty::Integer(dst_bits) = dst_ty else {
                unreachable!()
            };
            let dst_val = Const::Int(inst.dest_tyidx(), src_val.sign_extend(*dst_bits));
            let dst_cidx = self.m.insert_const(dst_val)?;
            self.m.replace(iidx, Inst::Const(dst_cidx));
        }

        Ok(())
    }

    fn opt_store(&mut self, iidx: InstIdx, inst: StoreInst) -> Result<(), CompilationError> {
        if !inst.is_volatile() {
            let tgt = self.an.op_map(&self.m, inst.ptr(&self.m));
            let val = self.an.op_map(&self.m, inst.val(&self.m));
            let is_dead = match self.an.heapvalue(
                &self.m,
                Address::from_operand(&self.m, tgt.clone()),
                val.byte_size(&self.m),
            ) {
                None => false,
                Some(Operand::Var(hv_iidx)) => Operand::Var(hv_iidx) == inst.val(&self.m),
                Some(Operand::Const(hv_cidx)) => match val {
                    Operand::Var(_) => false,
                    Operand::Const(cidx) => match (self.m.const_(cidx), self.m.const_(hv_cidx)) {
                        (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                            debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                            debug_assert_eq!(lhs.bitw(), rhs.bitw());
                            lhs == rhs
                        }
                        x => todo!("{x:?}"),
                    },
                },
            };
            if is_dead {
                self.m.replace(iidx, Inst::Tombstone);
            } else {
                self.an
                    .push_heap_store(&self.m, Address::from_operand(&self.m, tgt), val);
            }
        }

        Ok(())
    }

    fn opt_zext(&mut self, iidx: InstIdx, inst: ZExtInst) -> Result<(), CompilationError> {
        if let Operand::Const(cidx) = self.an.op_map(&self.m, inst.val(&self.m)) {
            let Const::Int(_src_ty, src_val) = self.m.const_(cidx) else {
                unreachable!()
            };
            let dst_ty = self.m.type_(inst.dest_tyidx());
            let Ty::Integer(dst_bits) = dst_ty else {
                unreachable!()
            };
            debug_assert!(*dst_bits <= 64);
            let dst_cidx = self.m.insert_const(Const::Int(
                inst.dest_tyidx(),
                src_val.zero_extend(*dst_bits),
            ))?;
            self.m.replace(iidx, Inst::Const(dst_cidx));
        }

        Ok(())
    }

    /// Rewrite the instruction at `iidx`: duplicate it and remap its operands so that it reflects
    /// everything learnt by the analyser.
    fn rewrite(&mut self, iidx: InstIdx) -> Result<(), CompilationError> {
        match self.m.inst_nocopy(iidx) {
            None => Ok(()),
            Some(Inst::Guard(_)) => {
                // We can't safely rewrite guard operands as we pick up the result of the analysis
                // on the guard itself!
                Ok(())
            }
            Some(inst) => {
                let r = inst.dup_and_remap_vars(&mut self.m, |m, op_iidx| {
                    self.an.op_map(m, Operand::Var(op_iidx))
                })?;
                self.m.replace(iidx, r);
                Ok(())
            }
        }
    }

    /// Attempt common subexpression elimination on `iidx`, replacing it with a `Copy` or
    /// `Tombstone` if possible.
    fn cse(&mut self, instll: &mut InstLinkedList, iidx: InstIdx) {
        let inst = match self.m.inst_nocopy(iidx) {
            // If this instruction is already a `Copy`, then there is nothing for CSE to do.
            None => return,
            // There's no point in trying CSE on a `Const` or `Tombstone`.
            Some(Inst::Const(_)) | Some(Inst::Tombstone) => return,
            Some(inst @ Inst::Guard(ginst)) => {
                for (_, back_inst) in instll.rev_iter(&self.m, inst) {
                    if let Inst::Guard(back_ginst) = back_inst {
                        if ginst.cond(&self.m) == back_ginst.cond(&self.m)
                            && ginst.expect() == back_ginst.expect()
                        {
                            self.m.replace(iidx, Inst::Tombstone);
                            return;
                        }
                    }
                }
                inst
            }
            Some(inst) => {
                // We don't perform CSE on instructions that have / enforce effects.
                debug_assert!(!inst.is_guard());
                if inst.has_store_effect(&self.m)
                    || inst.is_internal_inst()
                    || inst.has_load_effect(&self.m)
                    || inst.has_store_effect(&self.m)
                {
                    return;
                }

                // Can we CSE the instruction at `iidx`?
                for (back_iidx, back_inst) in instll.rev_iter(&self.m, inst) {
                    if inst.decopy_eq(&self.m, back_inst) {
                        self.m.replace(iidx, Inst::Copy(back_iidx));
                        return;
                    }
                }
                inst
            }
        };

        // We only need to `push` instructions that:
        //
        //   1. Can possibly be CSE candidates. Earlier in the function we've ruled out a number of
        //      things that aren't CSE candidates, which saves us some pointless work.
        //   2. Haven't been turned into `Copy`s. So only if we've failed to CSE a given
        //      instruction is it worth pushing to the `instll`.
        instll.push(iidx, inst);
    }
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn opt(m: Module) -> Result<Module, CompilationError> {
    Opt::new(m).opt()
}

#[cfg(test)]
mod test {
    use super::*;

    fn opt(m: Module) -> Result<Module, CompilationError> {
        Opt::new(m).opt().map(|mut m| {
            // Testing is much easier if we explicitly run DCE.
            m.dead_code_elimination();
            m
        })
    }

    #[test]
    fn opt_const_guard() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = param 0
            guard false, 0i1, [%0]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i1 = param ...
        ",
        );
    }

    #[test]
    fn opt_const_guard_indirect() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = eq 0i8, 0i8
            guard true, %0, []
            %2: i1 = eq 0i8, 1i8
            guard false, %2, [%0]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
        ",
        );
    }

    #[test]
    fn opt_const_guard_chain() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = mul %0, 0i8
            %2: i1 = eq %1, 0i8
            guard true, %2, [%0, %1]
            black_box %0
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_add_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = add %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_add_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 0i8
            %1: i8 = add %0, 1i8
            %2: i64 = 18446744073709551614i64
            %3: i64 = add %2, 4i64
            black_box %1
            black_box %3
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i8
            black_box 2i64
        ",
        );
    }

    #[test]
    fn opt_sub_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = 0i8
            %2: i8 = sub %1, 1i8
            %3: i64 = 18446744073709551614i64
            %4: i64 = sub %3, 4i64
            %5: i8 = sub %0, 0i8
            %6: i8 = sub 0i8, %0
            black_box %2
            black_box %4
            black_box %5
            black_box %6
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %6: i8 = sub 0i8, %0
            black_box 255i8
            black_box 18446744073709551610i64
            black_box %0
            black_box %6
        ",
        );
    }

    #[test]
    fn opt_and_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = and %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box 0i8
        ",
        );
    }

    #[test]
    fn opt_and_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 3i8
            %2: i8 = and %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 2i8
        ",
        );
    }

    #[test]
    fn opt_dyn_ptr_add_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = dyn_ptr_add %0, 2i8, 3
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 6
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_lshr_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = lshr %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = lshr 0i8, %0
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box 0i8
        ",
        );
    }

    #[test]
    fn opt_lshr_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 1i8
            %2: i8 = lshr %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i8
        ",
        );
    }

    #[test]
    fn opt_shl_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = shl %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = shl 0i8, %0
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box 0i8
        ",
        );
    }

    #[test]
    fn opt_shl_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 1i8
            %2: i8 = shl %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 4i8
        ",
        );
    }

    #[test]
    fn opt_or_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = or %0, 0i8
            %2: i8 = or 0i8, %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_or_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 3i8
            %2: i8 = or %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 3i8
        ",
        );
    }

    #[test]
    fn opt_mul_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            %2: i8 = mul %0, 0i8
            %3: i8 = add %1, %2
            %4: i8 = mul 0i8, %0
            %5: i8 = add %1, %2
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            black_box %1
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_mul_one() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            %2: i8 = mul %0, 1i8
            %3: i8 = add %1, %2
            %4: i8 = mul 1i8, %0
            %5: i8 = add %1, %4
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            %3: i8 = add %1, %0
            black_box %3
            black_box %3
        ",
        );
    }

    #[test]
    fn opt_mul_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 0i8
            %1: i8 = mul %0, 1i8
            %2: i8 = 1i8
            %3: i8 = mul %2, 1i8
            %4: i64 = 9223372036854775809i64
            %5: i64 = mul %4, 2i64
            black_box %1
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 0i8
            black_box 1i8
            black_box 2i64
        ",
        );
    }

    #[test]
    fn opt_mul_shl() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i64 = param 0
            %1: i64 = mul %0, 2i64
            %2: i64 = mul %0, 4i64
            %3: i64 = mul %0, 4611686018427387904i64
            %4: i64 = mul %0, 9223372036854775807i64
            %5: i64 = mul %0, 12i64
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i64 = param ...
            %1: i64 = shl %0, 1i64
            %2: i64 = shl %0, 2i64
            %3: i64 = shl %0, 62i64
            %4: i64 = mul %0, 9223372036854775807i64
            %5: i64 = mul %0, 12i64
            black_box ...
            ...
        ",
        );
    }

    #[test]
    fn opt_xor_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = xor %0, 0i8
            %2: i8 = xor 0i8, %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_xor_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 3i8
            %2: i8 = xor %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i8
        ",
        );
    }

    #[test]
    fn opt_icmp_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = eq 0i8, 0i8
            %1: i1 = eq 0i8, 1i8
            %2: i1 = ne 0i8, 0i8
            %3: i1 = ne 0i8, 1i8
            %4: i1 = ugt 0i8, 0i8
            %5: i1 = ugt 0i8, 1i8
            %6: i1 = ugt 1i8, 0i8
            %7: i1 = uge 0i8, 0i8
            %8: i1 = uge 0i8, 1i8
            %9: i1 = uge 1i8, 0i8
            %10: i1 = ult 0i8, 0i8
            %11: i1 = ult 0i8, 1i8
            %12: i1 = ult 1i8, 0i8
            %13: i1 = ule 0i8, 0i8
            %14: i1 = ule 0i8, 1i8
            %15: i1 = ule 1i8, 0i8
            %16: i1 = sgt 0i8, 0i8
            %17: i1 = sgt 0i8, -1i8
            %18: i1 = sgt -1i8, 0i8
            %19: i1 = sge 0i8, 0i8
            %20: i1 = sge 0i8, -1i8
            %21: i1 = sge -1i8, 0i8
            %22: i1 = slt 0i8, 0i8
            %23: i1 = slt 0i8, -1i8
            %24: i1 = slt -1i8, 0i8
            %25: i1 = sle 0i8, 0i8
            %26: i1 = sle 0i8, -1i8
            %27: i1 = sle -1i8, 0i8
            black_box %0
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
            black_box %6
            black_box %7
            black_box %8
            black_box %9
            black_box %10
            black_box %11
            black_box %12
            black_box %13
            black_box %14
            black_box %15
            black_box %16
            black_box %17
            black_box %18
            black_box %19
            black_box %20
            black_box %21
            black_box %22
            black_box %23
            black_box %24
            black_box %25
            black_box %26
            black_box %27
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
        ",
        );
    }

    #[test]
    fn opt_select() {
        // Test constant condition.
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            %2: i8 = 1i1 ? %0 : %1
            %3: i8 = 0i1 ? %0 : %1
            black_box %2
            black_box %3
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            black_box %0
            black_box %1
        ",
        );

        // Test equivalent true/false values.
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = param 0
            %1: i8 = param 1
            %2: i8 = param 2
            %3: i8 = %0 ? 0i8 : 0i8
            %4: i8 = %0 ? %1 : %1
            %5: i8 = %0 ? %1 : %2
            black_box %3
            black_box %4
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i1 = param ...
            %1: i8 = param ...
            %2: i8 = param ...
            %5: i8 = %0 ? %1 : %2
            black_box 0i8
            black_box %1
            black_box %5
        ",
        );
    }

    #[test]
    fn opt_zext_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i16 = zext 1i8
            %1: i32 = zext 4294967295i32
            %2: i64 = zext 4294967295i32
            black_box %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i16
            black_box 4294967295i32
            black_box 4294967295i64
        ",
        );
    }

    #[test]
    fn opt_ptradd_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = ptr_add %0, 0
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_dynptradd_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = dyn_ptr_add %0, 2i64, 8
            black_box %1
            %3: ptr = dyn_ptr_add %0, -1i64, 8
            black_box %3
            %5: ptr = dyn_ptr_add %0, -8i64, 16
            black_box %5
            %7: ptr = dyn_ptr_add %0, 0i64, 16
            black_box %7
            %9: ptr = dyn_ptr_add 0x1234, 0i64, 16
            black_box %9
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 16
            black_box %1
            %3: ptr = ptr_add %0, -8
            black_box %3
            %5: ptr = ptr_add %0, -128
            black_box %5
            black_box %0
            black_box 0x1234
        ",
        );
    }

    #[test]
    fn opt_guard_dup() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            %2: i1 = eq %0, %1
            guard true, %2, [%0, %1]
            guard true, %2, [%0, %1]
            guard false, %2, [%0, %1]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            %2: i1 = eq %0, %1
            guard true, %2, ...
            guard false, %2, ...
        ",
        );
    }

    #[test]
    fn opt_peeling_simple() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            header_start [%0]
            %2: i8 = add %0, 1i8
            header_end [%2]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            header_start [%0]
            %2: i8 = add %0, 1i8
            header_end [%2]
            %4: i8 = param ...
            body_start [%4]
            %6: i8 = add %4, 1i8
            body_end [%6]
        ",
        );
    }

    #[ignore]
    #[test]
    fn opt_peeling_hoist() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            body_start [%0, %1]
            %2: i8 = add %0, 1i8
            %3: i8 = add %1, %2
            body_end [%0, %3]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            head_start [%0, %1]
            %3: i8 = add %0, 1i8
            %4: i8 = add %1, %3
            head_end [%0, %4, %3]
            body_start [%0, %4, %3]
            %10: i8 = add %4, %3
            body_end [%0, %10, %3]
        ",
        );
    }

    #[test]
    fn opt_dead_load() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            %2: i8 = load %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            black_box %1
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            *%0 = %1
            %3: i8 = load %0
            black_box %1
            black_box %3
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            black_box %1
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_dead_load_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, []
            %4: i8 = load %0
            black_box %1
            black_box %4
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, ...
            black_box 3i8
            black_box 3i8
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            *%0 = %1
            %3: i8 = load %0
            black_box %1
            black_box %3
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            black_box %1
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_dead_store() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            *%0 = %1
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            *%0 = %1
            *%0 = %1
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            *%0 = %1
            *%0 = 3i8
            *%0 = %1
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            *%0 = 3i8
            *%0 = %1
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_dead_store_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, []
            *%0 = 3i8
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, ...
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, []
            *%0 = 2i8
            *%0 = 3i8
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = load %0
            %2: i1 = eq %1, 3i8
            guard true, %2, ...
            *%0 = 2i8
            *%0 = 3i8
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = ptr_add %0, 8
            %2: ptr = ptr_add %0, 9
            %3: i8 = load %2
            %4: i1 = eq %3, 3i8
            guard true, %4, []
            *%1 = 2i8
            *%2 = 3i8
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 8
            %2: ptr = ptr_add %0, 9
            %3: i8 = load %2
            %4: i1 = eq %3, 3i8
            guard true, %4, ...
            *%1 = 2i8
        ",
        );
    }

    #[test]
    fn opt_dead_store_overlap() {
        // Check the inclusive/exclusive ranges
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = ptr_add %0, 8
            %2: ptr = ptr_add %0, 16
            *%1 = 1i64
            *%2 = 1i64
            *%1 = 1i64
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 8
            %2: ptr = ptr_add %0, 16
            *%1 = 1i64
            *%2 = 1i64
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = ptr_add %0, 1
            %2: ptr = ptr_add %0, 8
            *%1 = 1i8
            *%2 = 1i64
            *%1 = 1i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 1
            %2: ptr = ptr_add %0, 8
            *%1 = 1i8
            *%2 = 1i64
            *%1 = 1i8
            black_box %1
        ",
        );
    }

    #[test]
    fn canonicalisation() {
        // Those that can be canonicalised
        Module::assert_ir_transform_eq(
            "
        entry:
          %0: i8 = param 0
          %1: i8 = add 3i8, %0
          %2: i8 = and 3i8, %0
          %3: i8 = mul 3i8, %0
          %4: i8 = or 3i8, %0
          %5: i8 = xor 3i8, %0
          black_box %1
          black_box %2
          black_box %3
          black_box %4
          black_box %5
",
            |m| opt(m).unwrap(),
            "
        ...
        entry:
          %0: i8 = param ...
          %1: i8 = add %0, 3i8
          %2: i8 = and %0, 3i8
          %3: i8 = mul %0, 3i8
          %4: i8 = or %0, 3i8
          %5: i8 = xor %0, 3i8
          black_box %1
          black_box %2
          black_box %3
          black_box %4
          black_box %5
",
        );

        // Those that cannot be canonicalised
        Module::assert_ir_transform_eq(
            "
        entry:
          %0: i8 = param 0
          %1: i8 = lshr 3i8, %0
          %2: i8 = shl 3i8, %0
          %3: i8 = sub 3i8, %0
          black_box %1
          black_box %2
          black_box %3
",
            |m| opt(m).unwrap(),
            "
        ...
        entry:
          %0: i8 = param ...
          %1: i8 = lshr 3i8, %0
          %2: i8 = shl 3i8, %0
          %3: i8 = sub 3i8, %0
          black_box %1
          black_box %2
          black_box %3
",
        );
    }
}
