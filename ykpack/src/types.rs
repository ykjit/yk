// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Types for the Yorick intermediate language.

use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Display},
    marker::PhantomData,
};

pub type CrateHash = u64;
pub type DefIndex = u32;
pub type BasicBlockIndex = u32;
pub type LocalIndex = u32;
pub type VariantIndex = u32;
pub type PromotedIndex = u32;

/// A mirror of the compiler's notion of a "definition ID".
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct DefId {
    pub crate_hash: CrateHash,
    pub def_idx: DefIndex,
}

impl DefId {
    pub fn new(crate_hash: CrateHash, def_idx: DefIndex) -> Self {
        Self {
            crate_hash,
            def_idx,
        }
    }
}

impl Display for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DefId({}, {})", self.crate_hash, self.def_idx)
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct Mir {
    pub def_id: DefId,
    pub item_path_str: String,
    pub blocks: Vec<BasicBlock>,
}

impl Mir {
    pub fn new(def_id: DefId, item_path_str: String, blocks: Vec<BasicBlock>) -> Self {
        Self {
            def_id,
            item_path_str,
            blocks,
        }
    }
}

impl Display for Mir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[Begin TIR for {}]", self.item_path_str)?;
        writeln!(f, "    {}:", self.def_id)?;
        for (i, b) in self.blocks.iter().enumerate() {
            write!(f, "    bb{}:\n{}", i, b)?;
        }
        writeln!(f, "[End TIR for {}]", self.item_path_str)?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct BasicBlock {
    pub stmts: Vec<Statement>,
    pub term: Terminator,
}

impl BasicBlock {
    pub fn new(stmts: Vec<Statement>, term: Terminator) -> Self {
        Self { stmts, term }
    }
}

impl Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for s in self.stmts.iter() {
            write!(f, "        {}", s)?;
        }
        writeln!(f, "        term: {}\n", self.term)
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Statement {
    Nop,
    Assign(Place, Rvalue),
    SetDiscriminant(Place, VariantIndex),
    Unimplemented,
}

impl Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{:?}", self)
    }
}

/// A place for storing things.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Place {
    Base(PlaceBase),
    Projection(PlaceProjection),
}

/// The "base" of a place projection.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum PlaceBase {
    Local(LocalIndex),
    Static(DefId),
    Promoted(PromotedIndex),
}

/// A projection (deref, index, field access, ...).
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct PlaceProjection {
    pub base: Box<Place>,
    pub elem: ProjectionElem<LocalIndex>,
}

/// Describes a projection operation upon a projection base.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum ProjectionElem<V> {
    Unimplemented(PhantomData<V>), // FIXME
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Operand {
    /// In MIR this is either Move or Copy.
    Place(Place),
    Unimplemented, // FIXME constants
}

/// Borrow descriptions.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum BorrowKind {
    Shared,
    Shallow,
    Unique,
    Mut, // FIXME two_phase borrow.
}

/// Things that can appear on the right-hand side of an assignment.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Rvalue {
    Use(Operand),
    Repeat(Operand, u64),
    Ref(BorrowKind, Place), // We do not store the region.
    Len(Place),
    BinaryOp(BinOp, Operand, Operand),
    CheckedBinaryOp(BinOp, Operand, Operand),
    NullaryOp(NullOp),
    UnaryOp(UnOp, Operand),
    Discriminant(Place),
    Aggregate(AggregateKind, Vec<Operand>),
    Unimplemented, // FIXME
}

/// Kinds of aggregate types.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum AggregateKind {
    Array,
    Tuple,
    Closure(DefId),
    Generator(DefId),
    Unimplemented,
}

/// Binary operations.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Offset,
}

// Operations with no arguments.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum NullOp {
    SizeOf,
    Box,
}

// Unary operations.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum UnOp {
    Not,
    Neg,
}

/// A call target.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum CallOperand {
    /// A statically known function identified by its DefId.
    Fn(DefId),
    /// An unknown or unhandled callable.
    Unknown, // FIXME -- Find out what else. Closures jump to mind.
}

/// A basic block terminator.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Terminator {
    Goto {
        target_bb: BasicBlockIndex,
    },
    SwitchInt {
        target_bbs: Vec<BasicBlockIndex>,
    },
    Resume,
    Abort,
    Return,
    Unreachable,
    Drop {
        target_bb: BasicBlockIndex,
        unwind_bb: Option<BasicBlockIndex>,
    },
    DropAndReplace {
        target_bb: BasicBlockIndex,
        unwind_bb: Option<BasicBlockIndex>,
    },
    Call {
        operand: CallOperand,
        cleanup_bb: Option<BasicBlockIndex>,
        ret_bb: Option<BasicBlockIndex>,
    },
    Assert {
        target_bb: BasicBlockIndex,
        cleanup_bb: Option<BasicBlockIndex>,
    },
    Yield {
        resume_bb: BasicBlockIndex,
        drop_bb: Option<BasicBlockIndex>,
    },
    GeneratorDrop,
}

impl Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// The top-level pack type.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Pack {
    Mir(Mir),
}

impl Display for Pack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Pack::Mir(mir) = self;
        write!(f, "{}", mir)
    }
}
