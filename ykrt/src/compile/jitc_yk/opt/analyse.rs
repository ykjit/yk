//! Analyse a trace and gradually refine what values we know a previous instruction can produce.

use super::{
    super::jit_ir::{GuardInst, Inst, InstIdx, Module, Operand, Predicate},
    Value,
};

/// Ongoing analysis of a trace: what value can a given instruction in the past produce?
///
/// Note that the analysis is forward-looking: just because an instruction's `Value` is (say) a
/// `Const` now does not mean it would be valid to assume that at earlier points it is safe to
/// assume it was also a `Const`.
pub(super) struct Analyse {
    /// For each instruction, what have we learnt about its [Value] so far?
    values: Vec<Value>,
}

impl Analyse {
    pub(super) fn new(m: &Module) -> Analyse {
        Analyse {
            values: vec![Value::Unknown; m.insts_len()],
        }
    }

    /// Map `op` based on our analysis so far. In some cases this will return `op` unchanged, but
    /// in others it may be able to turn what looks like a variable reference into a constant.
    pub(super) fn op_map(&mut self, m: &Module, op: Operand) -> Operand {
        match op {
            Operand::Var(iidx) => match self.values[usize::from(iidx)] {
                Value::Unknown => {
                    // Since we last saw an `ICmp` instruction, we may have gathered new knowledge
                    // that allows us to turn it into a constant.
                    if let (iidx, Inst::ICmp(inst)) = m.inst_decopy(iidx) {
                        let lhs = self.op_map(m, inst.lhs(m));
                        let pred = inst.predicate();
                        let rhs = self.op_map(m, inst.rhs(m));
                        if let (&Operand::Const(lhs_cidx), &Operand::Const(rhs_cidx)) = (&lhs, &rhs)
                        {
                            if pred == Predicate::Equal && m.const_(lhs_cidx) == m.const_(rhs_cidx)
                            {
                                self.values[usize::from(iidx)] = Value::Const(m.true_constidx());
                                return Operand::Const(m.true_constidx());
                            }
                        }
                    }
                    op
                }
                Value::Const(cidx) => Operand::Const(cidx),
            },
            Operand::Const(_) => op,
        }
    }

    /// Update our idea of what value the instruction at `iidx` can produce.
    pub(super) fn set_value(&mut self, iidx: InstIdx, v: Value) {
        self.values[usize::from(iidx)] = v;
    }

    /// Use the guard `inst` to update our knowledge about the variable used as its condition.
    pub(super) fn guard(&mut self, m: &Module, g_inst: GuardInst) {
        if let Operand::Var(iidx) = g_inst.cond(m) {
            if let (_, Inst::ICmp(ic_inst)) = m.inst_decopy(iidx) {
                let lhs = self.op_map(m, ic_inst.lhs(m));
                let pred = ic_inst.predicate();
                let rhs = self.op_map(m, ic_inst.rhs(m));
                match (&lhs, &rhs) {
                    (&Operand::Const(_), &Operand::Const(_)) => {
                        // This will have been handled by icmp/guard optimisations.
                        unreachable!();
                    }
                    (&Operand::Var(iidx), &Operand::Const(cidx))
                    | (&Operand::Const(cidx), &Operand::Var(iidx)) => {
                        if (g_inst.expect && pred == Predicate::Equal)
                            || (!g_inst.expect && pred == Predicate::NotEqual)
                        {
                            self.set_value(iidx, Value::Const(cidx));
                        }
                    }
                    (&Operand::Var(_), &Operand::Var(_)) => (),
                }
            }
        }
    }
}
