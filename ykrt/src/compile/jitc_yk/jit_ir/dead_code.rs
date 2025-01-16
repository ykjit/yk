//! Dead code elimination.

use super::{Inst, InstIdx, Module};
use vob::Vob;

impl Module {
    /// Eliminate dead code from this module. Note that this does not compact the instructions: it
    /// merely replaces them with [Inst::Tombstone]s.
    pub(crate) fn dead_code_elimination(&mut self) {
        // We perform a simple reverse reachability analysis, tracking what's alive with a single
        // bit.
        let mut used = Vob::from_elem(false, usize::from(self.last_inst_idx()) + 1);
        let mut tombstone = Vob::from_elem(false, usize::from(self.last_inst_idx()) + 1);
        for (iidx, inst) in self.iter_skipping_insts().rev() {
            if used.get(usize::from(iidx)).unwrap()
                || inst.is_internal_inst()
                || inst.is_guard()
                || inst.has_store_effect(self)
            {
                used.set(usize::from(iidx), true);
                inst.map_operand_vars(self, &mut |x| {
                    used.set(usize::from(x), true);
                });
            } else {
                tombstone.set(usize::from(iidx), true);
            }
        }

        for iidx in tombstone.iter_set_bits(..) {
            self.replace(InstIdx::unchecked_from(iidx), Inst::Tombstone);
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
            %0: i8 = param 0
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
            %0: i8 = param ...
            %2: i8 = add %0, %0
            black_box %2
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
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
            %0: i8 = param ...
            %1: i8 = param ...
            %3: i8 = add %1, %0
            black_box %3
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
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
            %0: i8 = param ...
            %1: i8 = param ...
            %3: i1 = ult %1, %1
            black_box %3
        ",
        );

        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i1 = ult %0, 1i8
            guard true, %1, []
            black_box %1
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i1 = ult %0, 1i8
            guard true, %1, [] ; ...
            black_box %1
        ",
        );

        Module::assert_ir_transform_eq(
            "
          func_decl f(i8)
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            call @f(%0)
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            call @f(%0)
        ",
        );

        Module::assert_ir_transform_eq(
            "
          func_type t1(i8)
          entry:
            %0: ptr = param 0
            %1: i8 = param 1
            %2: i8 = param 2
            icall<t1> %0(%1)
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: ptr = param ...
            %1: i8 = param ...
            %2: i8 = param ...
            icall %0(%1)
        ",
        );

        Module::assert_ir_transform_eq(
            "
          func_type t1(i8)
          entry:
            %0: i8 = param 0
            %1: i8 = %0
            black_box %1
        ",
            |mut m| {
                m.dead_code_elimination();
                m
            },
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }
}
