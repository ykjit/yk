//! Various utilities for building SIR.
//!
//! This code is intended for consumption by Rust code generator backends. By making these
//! utilities public we avoid the need for code duplication in each backend that wishes to
//! generate SIR.

use indexmap::IndexMap;
use std::{cell::RefCell, convert::TryFrom, default::Default, io};

use crate::{
    BasicBlock, BasicBlockIndex, Body, BodyFlags, CguHash, IRPlace, Local, LocalDecl, Statement,
    Terminator, Ty, TyIndex, TypeId,
};

/// A collection of in-memory SIR data structures to be serialised.
/// Each codegen unit builds one instance of this which is then merged into a "global" instance
/// when the unit completes.
pub struct Sir {
    pub types: RefCell<SirTypes>,
    pub funcs: RefCell<Vec<Body>>,
}

impl Sir {
    pub fn new(cgu_hash: CguHash) -> Self {
        Sir {
            types: RefCell::new(SirTypes {
                cgu_hash,
                map: Default::default(),
                next_idx: TyIndex(0),
            }),
            funcs: Default::default(),
        }
    }

    /// Returns true if there is nothing inside.
    pub fn is_empty(&self) -> bool {
        self.funcs.borrow().len() == 0
    }

    /// Writes a textual representation of the SIR to `w`. Used for `--emit yk-sir`.
    pub fn dump(&self, w: &mut dyn io::Write) -> Result<(), io::Error> {
        for f in self.funcs.borrow().iter() {
            writeln!(w, "{}", f)?;
        }
        Ok(())
    }
}

/// A structure for building the SIR of a function.
pub struct SirBuilder {
    /// The SIR function we are building.
    pub func: Body,
}

impl SirBuilder {
    pub fn new(symbol_name: String, flags: BodyFlags, num_args: usize, block_count: usize) -> Self {
        // Since there's a one-to-one mapping between MIR and SIR blocks, we know how many SIR
        // blocks we will need and can allocate empty SIR blocks ahead of time.
        let blocks = vec![
            BasicBlock {
                stmts: Default::default(),
                term: Terminator::Unreachable,
            };
            block_count
        ];

        Self {
            func: Body {
                symbol_name,
                blocks,
                flags,
                local_decls: Vec::new(),
                num_args,
                layout: (0, 0),
                offsets: Vec::new(),
            },
        }
    }

    /// Returns a zero-offset IRPlace for a new SIR local.
    pub fn new_sir_local(&mut self, sirty: TypeId) -> IRPlace {
        let idx = u32::try_from(self.func.local_decls.len()).unwrap();
        self.func.local_decls.push(LocalDecl {
            ty: sirty,
            referenced: false,
        });
        IRPlace::Val {
            local: Local(idx),
            off: 0,
            ty: sirty,
        }
    }

    /// Tells the tracer codegen that the local `l` is referenced, and that is should be allocated
    /// directly to the stack and not a register. You can't reference registers.
    pub fn notify_referenced(&mut self, l: Local) {
        let idx = usize::try_from(l.0).unwrap();
        let slot = self.func.local_decls.get_mut(idx).unwrap();
        slot.referenced = true;
    }

    /// Returns true if there are no basic blocks.
    pub fn is_empty(&self) -> bool {
        self.func.blocks.len() == 0
    }

    /// Appends a statement to the specified basic block.
    pub fn push_stmt(&mut self, bb: BasicBlockIndex, stmt: Statement) {
        self.func.blocks[usize::try_from(bb).unwrap()]
            .stmts
            .push(stmt);
    }

    /// Sets the terminator of the specified block.
    pub fn set_terminator(&mut self, bb: BasicBlockIndex, new_term: Terminator) {
        let term = &mut self.func.blocks[usize::try_from(bb).unwrap()].term;
        // We should only ever replace the default unreachable terminator assigned at allocation time.
        debug_assert!(*term == Terminator::Unreachable);
        *term = new_term
    }
}

pub struct SirTypes {
    /// A globally unique identifier for the codegen unit.
    pub cgu_hash: CguHash,
    /// Maps types to their index. Ordered by insertion via `IndexMap`.
    pub map: IndexMap<Ty, TyIndex>,
    /// The next available type index.
    next_idx: TyIndex,
}

impl SirTypes {
    /// Get the index of a type. If this is the first time we have seen this type, a new index is
    /// allocated and returned.
    ///
    /// Note that the index is only unique within the scope of the current compilation unit.
    /// To make a globally unique ID, we pair the index with CGU hash (see CguHash).
    pub fn index(&mut self, t: Ty) -> TyIndex {
        let next_idx = &mut self.next_idx.0;
        *self.map.entry(t).or_insert_with(|| {
            let idx = *next_idx;
            *next_idx += 1;
            TyIndex(idx)
        })
    }

    /// Given a type id return the corresponding type.
    pub fn get(&self, tyid: TypeId) -> &Ty {
        self.map
            .get_index(usize::try_from(tyid.idx.0).unwrap())
            .unwrap()
            .0
    }
}
