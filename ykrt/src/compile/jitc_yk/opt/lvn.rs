#[cfg(test)]
use crate::compile::jitc_yk::jit_ir::BlackBoxInst;
use crate::compile::{
    jitc_yk::jit_ir::{AddInst, Inst, InstIdx, Module, Operand},
    CompilationError,
};
use typed_index_collections::TiVec;

/// Perform (very very) simple dead-code elimination.
pub(super) fn lvn(mut m: Module) -> Result<Module, CompilationError> {
    // Which instructions have an effect on the wider world?
    let mut used_ops = vec![false; m.len()];
    for inst in m.iter_mut_insts() {
        match inst {
            Inst::LoadTraceInput(_) => (),
            Inst::Add(x) => {
                if let Operand::Local(y) = x.lhs() {
                    used_ops[usize::from(y)] = true;
                }
                if let Operand::Local(y) = x.rhs() {
                    used_ops[usize::from(y)] = true;
                }
            }
            #[cfg(test)]
            Inst::BlackBox(x) => {
                if let Operand::Local(y) = x.operand() {
                    used_ops[usize::from(y)] = true;
                }
            }
            x => todo!("{x:?}"),
        }
    }

    // Rewrite the sequence of instructions, eliminating instructions whose results are unused, and
    // performing the necessary adjustments to operands in the light of those instructions being
    // removed.
    fn operand_off(op: Operand, offs: &[InstIdx]) -> Operand {
        match op {
            Operand::Local(x) => Operand::Local(offs[usize::from(x)]),
            Operand::Const(_) => op,
        }
    }

    let mut off = 0;
    let mut insts = TiVec::<InstIdx, _>::new();
    let mut offs = Vec::new();
    for (i, inst) in m.iter_insts().enumerate() {
        // The `unwrap` can't fail because `i - off <= i` (by definition) and `i` is a
        // representable `InstIdx`.
        offs.push(InstIdx::new(i - off).unwrap());
        if !used_ops[i] {
            match inst {
                #[cfg(test)]
                Inst::BlackBox(_) => (),
                Inst::Add(_) => {
                    off += 1;
                    continue;
                }
                x => todo!("{x:?}"),
            }
        }
        let inst = match inst {
            Inst::LoadTraceInput(_) => inst.clone(),
            Inst::Add(x) => {
                AddInst::new(operand_off(x.lhs(), &offs), operand_off(x.rhs(), &offs)).into()
            }
            #[cfg(test)]
            Inst::BlackBox(x) => BlackBoxInst::new(operand_off(x.operand(), &offs)).into(),
            x => todo!("{x:?}"),
        };
        insts.push(inst);
    }
    m.replace_insts(insts);
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_add() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i16 = load_ti 0
            %1: i16 = load_ti 16
            %2: i16 = add %0, %1
            %3: i16 = add %0, %1
            %4: i16 = add %0, %3
            black_box %3
            black_box %4
        ",
            |m| lvn(m).unwrap(),
            "
          ...
          entry:
            %0: i16 = load_ti 0
            %1: i16 = load_ti 16
            %2: i16 = add %0, %1
            %3: i16 = add %0, %2
            black_box %2
            black_box %3
        ",
        );
    }
}
