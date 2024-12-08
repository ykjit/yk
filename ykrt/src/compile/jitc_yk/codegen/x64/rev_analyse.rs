use crate::compile::{
    jitc_yk::jit_ir::{Inst, InstIdx, Module, Operand, PtrAddInst},
    CompilationError,
};
use vob::Vob;

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
) -> Result<(Vec<InstIdx>, Vob, Vec<Option<PtrAddInst>>), CompilationError> {
    let mut inst_vals_alive_until = vec![InstIdx::try_from(0).unwrap(); m.insts_len()];
    let mut ptradds = vec![None; m.insts_len()];
    let mut used_insts = Vob::from_elem(false, usize::from(m.last_inst_idx()) + 1);
    for (iidx, inst) in m.iter_skipping_insts().rev() {
        if used_insts.get(usize::from(iidx)).unwrap()
            || inst.has_store_effect(m)
            || inst.is_barrier(m)
        {
            used_insts.set(usize::from(iidx), true);

            // "Inline" `PtrAdd`s into loads/stores, and don't mark the `PtrAdd` as used. This
            // means that some (though not all) `PtrAdd`s will not lead to actual code being
            // generated.
            match inst {
                Inst::Load(x) => {
                    if let Operand::Var(op_iidx) = x.operand(m) {
                        if let Inst::PtrAdd(pa_inst) = m.inst(op_iidx) {
                            ptradds[usize::from(iidx)] = Some(pa_inst);
                            if let Operand::Var(y) = pa_inst.ptr(m) {
                                if inst_vals_alive_until[usize::from(y)] < iidx {
                                    inst_vals_alive_until[usize::from(y)] = iidx;
                                }
                                used_insts.set(usize::from(y), true);
                            }
                            continue;
                        }
                    }
                }
                Inst::Store(x) => {
                    if let Operand::Var(op_iidx) = x.tgt(m) {
                        if let Inst::PtrAdd(pa_inst) = m.inst(op_iidx) {
                            ptradds[usize::from(iidx)] = Some(pa_inst);
                            if let Operand::Var(y) = pa_inst.ptr(m) {
                                if inst_vals_alive_until[usize::from(y)] < iidx {
                                    inst_vals_alive_until[usize::from(y)] = iidx;
                                }
                                used_insts.set(usize::from(y), true);
                            }
                            if let Operand::Var(y) = x.val(m) {
                                if inst_vals_alive_until[usize::from(y)] < iidx {
                                    inst_vals_alive_until[usize::from(y)] = iidx;
                                }
                                used_insts.set(usize::from(y), true);
                            }
                            continue;
                        }
                    }
                }
                _ => (),
            }

            // Calculate inst_vals_alive_until
            inst.map_operand_locals(m, &mut |x| {
                used_insts.set(usize::from(x), true);
                if inst_vals_alive_until[usize::from(x)] < iidx {
                    inst_vals_alive_until[usize::from(x)] = iidx;
                }
            });
        }
    }

    Ok((inst_vals_alive_until, used_insts, ptradds))
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
              tloop_start [%0]
              %2: i8 = %0
              tloop_jump [%2]
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
              tloop_start [%0]
              %2: i8 = add %0, %0
              %3: i8 = add %0, %0
              %4: i8 = %2
              tloop_jump [%4]
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
        let (_, used, ptradds) = rev_analyse(&m).unwrap();
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
