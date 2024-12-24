use super::reg_alloc::{Register, VarLocation};
use crate::compile::{
    jitc_yk::jit_ir::{
        BinOp, BinOpInst, DynPtrAddInst, ICmpInst, Inst, InstIdx, LoadInst, Module, Operand,
        PtrAddInst, SExtInst, SelectInst, StoreInst, TruncInst, ZExtInst,
    },
    CompilationError,
};
use dynasmrt::x64::Rq;
use vob::Vob;

struct RevAnalyse<'a> {
    m: &'a Module,
    inst_vals_alive_until: Vec<InstIdx>,
    ptradds: Vec<Option<PtrAddInst>>,
    used_insts: Vob,
    vloc_hints: Vec<Option<VarLocation>>,
}

impl<'a> RevAnalyse<'a> {
    fn new(m: &'a Module) -> RevAnalyse<'a> {
        Self {
            m,
            inst_vals_alive_until: vec![InstIdx::try_from(0).unwrap(); m.insts_len()],
            ptradds: vec![None; m.insts_len()],
            used_insts: Vob::from_elem(false, usize::from(m.last_inst_idx()) + 1),
            vloc_hints: vec![None; m.insts_len()],
        }
    }

    fn analyse(&mut self) {
        for (iidx, inst) in self.m.iter_skipping_insts().rev() {
            if self.used_insts.get(usize::from(iidx)).unwrap()
                || inst.has_store_effect(self.m)
                || inst.is_barrier(self.m)
            {
                self.used_insts.set(usize::from(iidx), true);

                match inst {
                    Inst::BinOp(x) => self.an_binop(iidx, x),
                    Inst::ICmp(x) => self.an_icmp(iidx, x),
                    Inst::PtrAdd(x) => self.an_ptradd(iidx, x),
                    Inst::DynPtrAdd(x) => self.an_dynptradd(iidx, x),
                    // "Inline" `PtrAdd`s into loads/stores, and don't mark the `PtrAdd` as used. This
                    // means that some (though not all) `PtrAdd`s will not lead to actual code being
                    // generated.
                    Inst::Load(x) => {
                        if self.an_load(iidx, x) {
                            continue;
                        }
                    }
                    Inst::Store(x) => {
                        if self.an_store(iidx, x) {
                            continue;
                        }
                    }
                    Inst::TraceHeaderEnd => {
                        self.an_header_end();
                    }
                    Inst::TraceBodyEnd => {
                        self.an_body_end();
                    }
                    Inst::SidetraceEnd => {
                        self.an_sidetrace_end();
                    }
                    Inst::SExt(x) => self.an_sext(iidx, x),
                    Inst::ZExt(x) => self.an_zext(iidx, x),
                    Inst::Select(x) => self.an_select(iidx, x),
                    Inst::Trunc(x) => self.an_trunc(iidx, x),
                    _ => (),
                }

                // Calculate inst_vals_alive_until
                inst.map_operand_vars(self.m, &mut |x| {
                    self.used_insts.set(usize::from(x), true);
                    if self.inst_vals_alive_until[usize::from(x)] < iidx {
                        self.inst_vals_alive_until[usize::from(x)] = iidx;
                    }
                });
            }
        }
    }

    fn an_binop(&mut self, iidx: InstIdx, binst: BinOpInst) {
        match binst.binop() {
            BinOp::Add | BinOp::And | BinOp::Or | BinOp::Xor => {
                if let Operand::Var(op_iidx) = binst.lhs(self.m) {
                    self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
                }
            }
            BinOp::AShr | BinOp::LShr | BinOp::Shl => {
                if let Operand::Var(op_iidx) = binst.lhs(self.m) {
                    self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
                }
                if let Operand::Var(op_iidx) = binst.rhs(self.m) {
                    self.vloc_hints[usize::from(op_iidx)] =
                        Some(VarLocation::Register(Register::GP(Rq::RCX)));
                }
            }
            BinOp::Mul | BinOp::SDiv | BinOp::UDiv => {
                if let Operand::Var(op_iidx) = binst.lhs(self.m) {
                    self.vloc_hints[usize::from(op_iidx)] =
                        Some(VarLocation::Register(Register::GP(Rq::RAX)));
                }
            }
            BinOp::Sub => match (binst.lhs(self.m), binst.rhs(self.m)) {
                (_, Operand::Const(_)) => {
                    if let Operand::Var(op_iidx) = binst.rhs(self.m) {
                        assert!(self.vloc_hints[usize::from(iidx)].is_none());
                        self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
                    }
                }
                (Operand::Var(_), _) => {
                    if let Operand::Var(op_iidx) = binst.lhs(self.m) {
                        self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
                    }
                }
                _ => (),
            },
            _ => (),
        }
    }

    fn an_icmp(&mut self, iidx: InstIdx, icinst: ICmpInst) {
        if let Operand::Var(op_iidx) = icinst.lhs(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    fn an_ptradd(&mut self, iidx: InstIdx, painst: PtrAddInst) {
        if let Operand::Var(op_iidx) = painst.ptr(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    fn an_dynptradd(&mut self, iidx: InstIdx, painst: DynPtrAddInst) {
        if let Operand::Var(op_iidx) = painst.num_elems(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    /// Analyse a [LoadInst]. Returns `true` if it has been inlined and should not go through the
    /// normal "calculate `inst_vals_alive_until`" phase.
    fn an_load(&mut self, iidx: InstIdx, inst: LoadInst) -> bool {
        if let Operand::Var(op_iidx) = inst.operand(self.m) {
            if let Inst::PtrAdd(pa_inst) = self.m.inst(op_iidx) {
                self.ptradds[usize::from(iidx)] = Some(pa_inst);
                if let Operand::Var(y) = pa_inst.ptr(self.m) {
                    if self.inst_vals_alive_until[usize::from(y)] < iidx {
                        self.inst_vals_alive_until[usize::from(y)] = iidx;
                        self.vloc_hints[usize::from(y)] = self.vloc_hints[usize::from(iidx)];
                    }
                    self.used_insts.set(usize::from(y), true);
                }
                return true;
            }
        }
        false
    }

    /// Analyse a [StoreInst]. Returns `true` if it has been inlined and should not go through the
    /// normal "calculate `inst_vals_alive_until`" phase.
    fn an_store(&mut self, iidx: InstIdx, inst: StoreInst) -> bool {
        if let Operand::Var(op_iidx) = inst.tgt(self.m) {
            if let Inst::PtrAdd(pa_inst) = self.m.inst(op_iidx) {
                self.ptradds[usize::from(iidx)] = Some(pa_inst);
                if let Operand::Var(y) = pa_inst.ptr(self.m) {
                    if self.inst_vals_alive_until[usize::from(y)] < iidx {
                        self.inst_vals_alive_until[usize::from(y)] = iidx;
                    }
                    self.used_insts.set(usize::from(y), true);
                }
                if let Operand::Var(y) = inst.val(self.m) {
                    if self.inst_vals_alive_until[usize::from(y)] < iidx {
                        self.inst_vals_alive_until[usize::from(y)] = iidx;
                    }
                    self.used_insts.set(usize::from(y), true);
                }
                return true;
            }
        }
        false
    }

    fn an_header_end(&mut self) {
        let mut param_vlocs = Vec::new();
        for (iidx, inst) in self.m.iter_skipping_insts() {
            match inst {
                Inst::Param(pinst) => {
                    param_vlocs.push(VarLocation::from_yksmp_location(
                        self.m,
                        iidx,
                        self.m.param(pinst.paramidx()),
                    ));
                }
                _ => break,
            }
        }

        debug_assert_eq!(param_vlocs.len(), self.m.trace_header_end().len());

        for (param_vloc, jump_op) in param_vlocs.into_iter().zip(self.m.trace_header_end()) {
            if let Operand::Var(op_iidx) = jump_op.unpack(self.m) {
                self.vloc_hints[usize::from(op_iidx)] = Some(param_vloc);
            }
        }
    }

    fn an_body_end(&mut self) {
        let mut param_vlocs = Vec::new();
        for (iidx, inst) in self.m.iter_skipping_insts() {
            match inst {
                Inst::Param(pinst) => {
                    param_vlocs.push(VarLocation::from_yksmp_location(
                        self.m,
                        iidx,
                        self.m.param(pinst.paramidx()),
                    ));
                }
                _ => break,
            }
        }

        debug_assert_eq!(param_vlocs.len(), self.m.trace_body_end().len());

        for (param_vloc, jump_op) in param_vlocs.into_iter().zip(self.m.trace_body_end()) {
            if let Operand::Var(op_iidx) = jump_op.unpack(self.m) {
                self.vloc_hints[usize::from(op_iidx)] = Some(param_vloc);
            }
        }
    }

    fn an_sidetrace_end(&mut self) {
        let vlocs = self.m.root_entry_vars();
        // Side-traces don't have a trace body since we don't apply loop peeling and thus use
        // `trace_header_end` to store the jump variables.
        debug_assert_eq!(vlocs.len(), self.m.trace_header_end().len());

        for (vloc, jump_op) in vlocs.iter().zip(self.m.trace_header_end()) {
            if let Operand::Var(op_iidx) = jump_op.unpack(self.m) {
                self.vloc_hints[usize::from(op_iidx)] = Some(*vloc);
            }
        }
    }

    fn an_sext(&mut self, iidx: InstIdx, seinst: SExtInst) {
        if let Operand::Var(op_iidx) = seinst.val(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    fn an_zext(&mut self, iidx: InstIdx, zeinst: ZExtInst) {
        if let Operand::Var(op_iidx) = zeinst.val(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    fn an_trunc(&mut self, iidx: InstIdx, tinst: TruncInst) {
        if let Operand::Var(op_iidx) = tinst.val(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }

    fn an_select(&mut self, iidx: InstIdx, sinst: SelectInst) {
        if let Operand::Var(op_iidx) = sinst.trueval(self.m) {
            self.vloc_hints[usize::from(op_iidx)] = self.vloc_hints[usize::from(iidx)];
        }
    }
}

/// Perform a reverse analysis on a module's instructions returning, in order:
///
///   1. A `Vec<InstIdx>` with one entry per instruction. Each denotes the last instruction that
///      the value produced by an instruction is used. By definition this must either be unused (if
///      an instruction does not produce a value) or `>=` the offset in this vector.
///
///   2. A `Vob` with one entry per instruction, denoting whether the code generator use its value.
///      This is implicitly a second layer of dead-code elimination: it doesn't cause JIT IR
///      instructions to be removed, but it will stop any code being (directly) generated for some
///      of them.
///
///   2. A `Vec<Option<PtrAddInst>>` that "inlines" pointer additions into load/stores. The
///      `PtrAddInst` is not marked as used, for such instructions: note that it might be marked as
///      used by other instructions!
pub(super) fn rev_analyse(
    m: &Module,
) -> Result<
    (
        Vec<InstIdx>,
        Vob,
        Vec<Option<PtrAddInst>>,
        Vec<Option<VarLocation>>,
    ),
    CompilationError,
> {
    let mut rean = RevAnalyse::new(m);
    rean.analyse();
    Ok((
        rean.inst_vals_alive_until,
        rean.used_insts,
        rean.ptradds,
        rean.vloc_hints,
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use std::assert_matches::assert_matches;
    use vob::vob;

    #[test]
    fn alive_until() {
        let m = Module::from_str(
            "
            entry:
              %0: i8 = param 0
              body_start [%0]
              %2: i8 = %0
              body_end [%2]
            ",
        );
        let alives = rev_analyse(&m).unwrap().0;
        assert_eq!(
            alives,
            vec![3, 0, 0, 0]
                .iter()
                .map(|x: &usize| InstIdx::try_from(*x).unwrap())
                .collect::<Vec<_>>()
        );

        let m = Module::from_str(
            "
            entry:
              %0: i8 = param 0
              body_start [%0]
              %2: i8 = add %0, %0
              %3: i8 = add %0, %0
              %4: i8 = %2
              body_end [%4]
            ",
        );
        let alives = rev_analyse(&m).unwrap().0;
        assert_eq!(
            alives,
            vec![2, 0, 5, 0, 0, 0]
                .iter()
                .map(|x: &usize| InstIdx::try_from(*x).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn inline_ptradds() {
        let m = Module::from_str(
            "
            entry:
              %0: ptr = param 0
              %1: ptr = ptr_add %0, 8
              %2: i8 = load %1
              %3: ptr = ptr_add %0, 16
              *%1 = 1i8
              black_box %2
              black_box %3
            ",
        );
        let (_, used, ptradds, _) = rev_analyse(&m).unwrap();
        assert_eq!(used, vob![true, false, true, true, true, true, true]);
        assert_matches!(
            ptradds.as_slice(),
            &[None, None, Some(_), None, Some(_), None, None]
        );
        let ptradd = ptradds[2].unwrap();
        assert_eq!(ptradd.ptr(&m), Operand::Var(InstIdx::try_from(0).unwrap()));
        assert_eq!(ptradd.off(), 8);
        let ptradd = ptradds[4].unwrap();
        assert_eq!(ptradd.ptr(&m), Operand::Var(InstIdx::try_from(0).unwrap()));
        assert_eq!(ptradd.off(), 8);
    }
}
