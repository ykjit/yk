//! Dead code elimination.

use super::{Inst, Module, Operand};
use vob::Vob;

impl Module {
    /// Eliminate dead code from this module. Note that this does not compact the instructions: it
    /// merely replaces them with [Inst::Tombstone]s.
    fn dead_code_elimination(&mut self) {
        // We perform a simple reverse reachability analysis, tracking what's alive with a single
        // bit.
        let mut used = Vob::from_elem(false, usize::from(self.last_inst_idx()) + 1);
        for iidx in self.iter_all_inst_idxs().rev() {
            let inst = self.inst_all(iidx);
            if inst.always_alive() || used.get(usize::from(iidx)).unwrap() {
                used.set(usize::from(iidx), true);
                inst.map_operands(self, |op| match op {
                    Operand::Const(_) => (),
                    Operand::Local(x) => {
                        used.set(usize::from(x), true);
                    }
                });
            } else {
                self.replace(iidx, Inst::Tombstone);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            black_box %1
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %1: i8 = load_ti 1
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = add %0, %0
            %2: i8 = add %0, %0
            black_box %2
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: i8 = load_ti 0
            %2: i8 = add %0, %0
            black_box %2
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %2: i8 = add %1, %0
            %3: i8 = add %1, %0
            black_box %3
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %3: i8 = add %1, %0
            black_box %3
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %2: i1 = ult %0, %0
            %3: i1 = ult %1, %1
            black_box %3
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %1: i8 = load_ti 1
            %3: i1 = ult %1, %1
            black_box %3
        ",
        );
    }
}
