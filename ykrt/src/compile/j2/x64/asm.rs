//! The X64 assembler. Currently uses [iced_x86] to assemble X64 operations.
//!
//! This module is not quite an MVP, but its design is fairly simplistic. For example: there is
//! surely a more efficient way of laying out the code than using pairs and vecs of vecs;
//! `near_callable` is something of a hack right now; and so on.
//!
//!
//! ## Terminology and indexing
//!
//! To avoid readers confusing HIR instructions with x64 instructions and, in particular, the
//! different indexing system this module uses, we call iced64 instructions "operations" within
//! this module. Operations are indexed by pairs `([BlockIdx], [OpIdx])`.
//!
//!
//! ## Layout of the eventual code
//!
//! Because the first part of a trace we assemble is the trace body, and only then the guard
//! bodies, the natural assembly layout would become `<guard bodies>, <main body>`. This would then
//! confuse a typical branch predictor when executing the main body, which is likely to default to
//! "backward jumps are probably loops and are probably taken". We thus want to lay the eventual
//! machine code out as `<main body>, <guard bodies>`. We therefore store each block individually,
//! and then do the final machine code layout in reverse block order. This is invisible to
//! everything outside this module.

use crate::compile::{
    CompilationError,
    j2::codebuf::{CodeBufInProgress, ExeCodeBuf},
};
use iced_x86::{Code, Encoder, Instruction as Op};
use index_vec::{IndexVec, index_vec};
use std::mem::replace;

/// We guarantee to align the start of blocks to `BLOCK_ALIGNMENT` bytes.
pub(super) static BLOCK_ALIGNMENT: usize = 16;

#[derive(Debug)]
pub(super) struct Asm {
    buf: CodeBufInProgress,
    /// The blocks we are assembling. By definition, the first block will be the main body of the
    /// trace, and any subsequent blocks will be guard bodies.
    blocks: IndexVec<BlockIdx, IndexVec<OpIdx, Op>>,
    /// As a simple optimisation, this is the currently-process block's instructions. When
    /// `block_complete` is called, we will move the entire contents of this into [Self::blocks].
    insts: IndexVec<OpIdx, Op>,
    /// Labels. New labels start with a value of `None`; when they are assigned to an instruction,
    /// this will become `Some(...)`.
    labels: IndexVec<LabelIdx, Option<(BlockIdx, OpIdx)>>,
    /// Operations (CALLs, JMPs), which need relocating. These are not stored in any particular
    /// order.
    relocs: Vec<(BlockIdx, OpIdx, RelocKind)>,
    log: IndexVec<BlockIdx, Vec<(OpIdx, String)>>,
}

impl Asm {
    pub(super) fn new(buf: CodeBufInProgress) -> Self {
        Asm {
            buf,
            blocks: index_vec![],
            insts: index_vec![],
            labels: index_vec![],
            relocs: Vec::new(),
            log: index_vec![vec![]],
        }
    }

    pub(super) fn block_completed(&mut self) {
        self.blocks.push(replace(&mut self.insts, index_vec![]));
        self.log.push(Vec::new());
    }

    pub(super) fn log(&mut self, s: String) {
        self.log.last_mut().unwrap().push((self.insts.len_idx(), s));
    }

    /// Create a new free-floating label: it will only be attached when `attach_label` is called on
    /// the label.
    pub(super) fn mk_label(&mut self) -> LabelIdx {
        self.labels.push(None)
    }

    /// Attach `lidx` to the most recently pushed instruction (i.e. the last instruction `push`ed
    /// before `attach_label` is called).
    pub(super) fn attach_label(&mut self, lidx: LabelIdx) {
        self.labels[lidx] = Some((self.blocks.len_idx(), self.insts.len_idx()));
    }

    /// Push an icedx64 [Op].
    pub(super) fn push_inst(&mut self, op: Result<Op, iced_x86::IcedError>) {
        self.insts.push(op.unwrap());
    }

    /// Push an operation and an associated [RelocKind].
    ///
    /// If `op` is an icedx64 branch / call, it must be a near branch with its address set to 0.
    /// Failing to do so will result in undefined behaviour.
    pub(super) fn push_reloc(
        &mut self,
        op: Result<iced_x86::Instruction, iced_x86::IcedError>,
        reloc: RelocKind,
    ) {
        let x64inst = op.unwrap();
        // Having a branch target of 0 is relied upon in the relocation process.
        assert_eq!(x64inst.near_branch64(), 0);
        let idx = self.insts.push(x64inst);
        self.relocs.push((self.blocks.len_idx(), idx, reloc));
    }

    /// Is `addr` representable as an x64 near call (a signed 32 bit int) relative to where this
    /// trace will be stored in memory?
    pub(super) fn is_near_callable(&self, addr: usize) -> bool {
        // At this point all we know about the trace is its lowest (`self.buf.addr(`)) and highest
        // (`self.buf.addr() + self.buflen`) addresses. We need to make sure that `addr` is at most
        // 2GiB away from the lowest or highest addresses.
        let delta = if addr < self.buf.as_ptr().addr() {
            self.buf.as_ptr().addr() + self.buf.len() - addr
        } else {
            addr - self.buf.as_ptr().addr()
        };
        delta < 0x80000000
    }

    /// Convert this semi-assembled trace into a fully assembled trace, performing relocations etc.
    ///
    /// # Panics
    ///
    /// If `block_completed` has not been called immediately prior to this function.
    pub(super) fn into_exe(
        mut self,
        log: bool,
        labels: &[LabelIdx],
    ) -> Result<(ExeCodeBuf, Option<String>, Vec<usize>), CompilationError> {
        // Convert the operations into a byte sequence, recording byte offsets as we go, which we
        // need for labels and relocations.
        let mut enc = Encoder::new(64);
        let base = u64::try_from(self.buf.as_ptr().addr()).unwrap();
        let mut off: u64 = 0;
        let mut offs = Vec::new();
        let mut blk_offs = IndexVec::with_capacity(self.blocks.len());
        assert!(self.insts.is_empty());
        for b in self.blocks.iter_mut() {
            blk_offs.push(offs.len());
            // Ensure the start of the block is properly aligned.
            while !off.is_multiple_of(u64::try_from(BLOCK_ALIGNMENT).unwrap()) {
                let ip = base + off;
                let lenb = enc.encode(&Op::with(Code::Nopd), ip).unwrap();
                off += u64::try_from(lenb).unwrap();
            }

            for (opidx, inst) in b.iter_mut_enumerated().rev() {
                let ip = base + off;
                inst.set_ip(ip);
                // At this point we don't necessarily know where
                // branch/jump/other-memory-displacement operations should go to: if they have an
                // address of 0, we know we'll be relocating the operation later. However, icedx86
                // will (rightfully) complain about address 0, so we stuff in a dummy jump address
                // (to the operation itself) which we'll patch a little below.
                if (inst.is_call_near() || inst.is_jmp_near() || inst.is_jcc_near())
                    && inst.near_branch64() == 0
                {
                    inst.set_near_branch64(ip);
                }
                if inst.is_ip_rel_memory_operand() && inst.memory_displacement64() == 0 {
                    inst.set_memory_displacement64(ip);
                }
                let lenb = enc
                    .encode(inst, ip)
                    .unwrap_or_else(|e| panic!("At machine {opidx:?} {inst:?}: {e:?}"));
                offs.push((off, lenb));
                off += u64::try_from(lenb).unwrap();
            }
        }
        // Relocations in the final instruction need to know the offset of the "next" instruction.
        offs.push((off, 0));

        // We now have the information we need to create label offsets.
        let label_offs = labels
            .iter()
            .map(|label| {
                let (bidx, opidx) = self.labels[*label].unwrap();
                let off = usize::from(blk_offs[bidx] + self.blocks[bidx].len() - opidx);
                usize::try_from(offs[off].0).unwrap()
            })
            .collect::<Vec<_>>();

        // Perform relocations.
        let mut enc = enc.take_buffer();
        for (bidx, opidx, reloc) in self.relocs.iter().cloned() {
            let inst_off = usize::from(blk_offs[bidx] + self.blocks[bidx].len() - opidx - 1);
            let inst_boff = usize::try_from(offs[inst_off].0).unwrap();
            let next_ip_boff = u64::try_from(inst_boff + offs[inst_off].1).unwrap();
            match reloc {
                RelocKind::BranchWithAddr(_) | RelocKind::RipRelativeWithLabel(_) => {
                    // We now have a not-very-nice hack where we examine the instruction to see
                    // which part of it we write the relocation to.
                    #[allow(clippy::if_same_then_else)]
                    let patch_boff = if enc[inst_boff] == 0xe9 {
                        inst_boff + 1 // JMP
                    } else if enc[inst_boff] == 0x0F
                        && enc[inst_boff + 1] >= 0x80
                        && enc[inst_boff + 1] <= 0x08F
                    {
                        inst_boff + 2 // JCC
                    } else if enc[inst_boff] == 0x66
                        && enc[inst_boff + 1] == 0x0F
                        && (enc[inst_boff + 2] >= 0x5C && enc[inst_boff + 2] <= 0x62)
                    {
                        inst_boff + 4 // PUNPCKLDQ / SUBPD
                    } else if (enc[inst_boff] == 0xF2 || enc[inst_boff] == 0xF3)
                        && enc[inst_boff + 1] == 0x0F
                        && enc[inst_boff + 2] == 0x10
                    {
                        inst_boff + 4 // MOVSD / MOVSS
                    } else {
                        todo!("{:X?}", &enc[inst_boff..inst_boff + 3])
                    };

                    let addr = match reloc {
                        RelocKind::BranchWithAddr(addr) => {
                            let diff = i32::try_from(
                                u64::try_from(addr)
                                    .unwrap()
                                    .checked_signed_diff(base + next_ip_boff)
                                    .unwrap(),
                            )
                            .unwrap();
                            enc[patch_boff..patch_boff + 4].copy_from_slice(&diff.to_le_bytes());
                            u64::try_from(addr).unwrap()
                        }
                        RelocKind::RipRelativeWithLabel(lidx) => {
                            let (lab_bidx, lab_opidx) = self.labels[lidx].unwrap();
                            let to_inst_off = usize::from(
                                blk_offs[lab_bidx] + self.blocks[lab_bidx].len() - lab_opidx,
                            );
                            let to_boff = offs[to_inst_off].0;
                            let diff =
                                i32::try_from(to_boff.checked_signed_diff(next_ip_boff).unwrap())
                                    .unwrap();
                            enc[patch_boff..patch_boff + 4].copy_from_slice(&diff.to_le_bytes());
                            base + to_boff
                        }
                        _ => unreachable!(),
                    };

                    if log {
                        self.blocks[bidx][opidx].set_near_branch64(addr);
                    }
                }
                RelocKind::NearCallWithAddr(addr) => {
                    assert_eq!(enc[inst_boff], 0xE8);
                    let addr = u64::try_from(addr).unwrap();
                    let next_ip = base + next_ip_boff;
                    let diff = i32::try_from(addr.checked_signed_diff(next_ip).unwrap()).unwrap();
                    enc[inst_boff + 1..inst_boff + 1 + 4].copy_from_slice(&diff.to_le_bytes());
                    if log {
                        self.blocks[bidx][opidx].set_near_branch64(addr);
                    }
                }
            }
        }

        // Copy into the executable buffer.
        if enc.len() > self.buf.len() {
            // If we've ended up here, it really suggests that our `buflen` heuristic is too
            // stingy. We _could_, though, restart the whole assembly process with a bigger buffer
            // if we really wanted to.
            todo!();
        }

        let exe = unsafe { self.buf.into_execodebuf(enc.as_ptr(), enc.len()) };

        let log = if log {
            // When we're replacing labels below, we'll be iterating over blocks in order 0..n but
            // op indexes in order n..0. We thus need to sort `self.relocs` according to this need.
            self.relocs
                .sort_by(|(lhs_bidx, lhs_opidx, _), (rhs_bidx, rhs_opidx, _)| {
                    lhs_bidx
                        .cmp(rhs_bidx)
                        .then_with(|| rhs_opidx.cmp(lhs_opidx))
                });

            use iced_x86::Formatter;
            let mut fmtr = iced_x86::NasmFormatter::new();
            fmtr.options_mut().set_branch_leading_zeros(false);
            fmtr.options_mut().set_hex_prefix("0x");
            fmtr.options_mut().set_hex_suffix("");
            fmtr.options_mut().set_rip_relative_addresses(true);
            fmtr.options_mut().set_show_branch_size(false);
            fmtr.options_mut().set_space_after_operand_separator(true);

            let mut relocs_iter = self.relocs.into_iter().peekable();
            let mut logs_iter = self.log.into_iter();
            let mut out = Vec::new();
            for (bidx, b) in self.blocks.into_iter_enumerated() {
                let log = logs_iter.next().unwrap_or_default();
                let mut log_iter = log.into_iter().rev().peekable();
                let _b_len = b.len();
                for (opidx, inst) in b.into_iter_enumerated().rev() {
                    while log_iter.peek().map(|x| x.0) > Some(opidx) {
                        out.push(format!("; {}", log_iter.next().unwrap().1));
                    }
                    if let Some(lidx) = self.labels.position(|x| x == &Some((bidx, opidx + 1))) {
                        out.push(format!("; l{}", usize::from(lidx)));
                    }
                    let mut inst_s = String::new();
                    fmtr.format(&inst, &mut inst_s);
                    if relocs_iter.peek().map(|(x, y, _)| (*x, *y)) == Some((bidx, opidx))
                        && let RelocKind::RipRelativeWithLabel(lidx) = relocs_iter.next().unwrap().2
                    {
                        inst_s.replace_range(
                            inst_s.rfind(' ').unwrap()..,
                            &format!(" l{}", usize::from(lidx)),
                        );
                    }
                    out.push(inst_s);
                }
                out.extend(log_iter.map(|x| format!("; {}", x.1)));
            }

            Some(out.join("\n"))
        } else {
            None
        };

        Ok((exe, log, label_offs))
    }
}

/// A relocation kind: these are all relative to the eventual address of the associated operation.
#[derive(Clone, Debug)]
pub(super) enum RelocKind {
    /// A relative branch to the address at `usize`.
    BranchWithAddr(usize),
    /// An instruction refering to the label at [LabelIdx] relative to RIP. Using this requires
    /// teaching `into_exe` about the specific encoding and the offset of the address in the
    /// instruction. Instructions supported are: JMP, JCC, MOVSD, MOVSS, PUNPCKLDQ, and SUBPD.
    RipRelativeWithLabel(LabelIdx),
    /// A near call to the address at `usize`. That this is possible should have been validated
    /// with [Asm::near_callable], or an exception will ensue.
    NearCallWithAddr(usize),
}

index_vec::define_index_type! {
    struct BlockIdx = u16;
}

index_vec::define_index_type! {
    /// The offset of an operation in a [Block].
    struct OpIdx = u32;
}

index_vec::define_index_type! {
    pub(in crate::compile::j2) struct LabelIdx = u32;
}
