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
use std::fmt::{self, Display};

pub type CrateHash = u64;
pub type DefIndex = u32;
pub type BasicBlockIndex = u32;
pub type LocalIndex = u32;
pub type VariantIndex = u32;
pub type PromotedIndex = u32;
pub type FieldIndex = u32;

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
    Projection(Projection),
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
pub struct Projection(pub Box<Place>, pub ProjectionElem);

/// Describes a projection operation upon a projection base.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum ProjectionElem {
    Deref,
    Field(FieldIndex),
    Index(LocalIndex),
    ConstantIndex {
        offset: u32,
        min_length: u32,
        from_end: bool,
    },
    Subslice {
        from: u32,
        to: u32,
    },
    Downcast(VariantIndex),
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Operand {
    /// In MIR this is either Move or Copy.
    Place(Place),
    Constant(Constant),
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum Constant {
    UnsignedInt(UnsignedInt),
    SignedInt(SignedInt),
    Unimplemented,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum UnsignedInt {
    Usize(usize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128 { hi: u64, lo: u64 },
    Unimplemented,
}

impl UnsignedInt {
    pub fn from_u128(val: u128) -> Self {
        UnsignedInt::U128 {
            hi: (val >> 64) as u64,
            lo: val as u64,
        }
    }

    /// Returns the u128 value from a `Integer::U128`. Errors if the enum is a different variant.
    pub fn u128(&self) -> Result<u128, ()> {
        match self {
            UnsignedInt::U128 { hi, lo } => Ok((*hi as u128) << 64 | *lo as u128),
            _ => Err(()),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub enum SignedInt {
    Isize(isize),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128 { hi: u64, lo: u64 },
    Unimplemented,
}

impl SignedInt {
    pub fn from_i128(val: i128) -> Self {
        SignedInt::I128 {
            hi: (val >> 64) as u64,
            lo: val as u64,
        }
    }

    /// Returns the i128 value from a `Integer::U128`. Errors if the enum is a different variant.
    pub fn i128(&self) -> Result<i128, ()> {
        match self {
            SignedInt::I128 { hi, lo } => Ok((*hi as i128) << 64 | *lo as i128),
            _ => Err(()),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{SignedInt, UnsignedInt};

    #[test]
    fn u128_round_trip() {
        let val = std::u128::MAX - 427819;
        assert_eq!(UnsignedInt::from_u128(val).u128().unwrap(), val);
    }

    #[test]
    fn i128_round_trip() {
        let val = std::i128::MIN + 77;
        assert_eq!(SignedInt::from_i128(val).i128().unwrap(), val);
    }
}
