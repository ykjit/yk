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
use iced_x86::{Encoder, Formatter, Instruction as Op, NasmFormatter};
use index_vec::{IndexVec, index_vec};
use std::slice;

pub(super) struct Asm {
    buf: CodeBufInProgress,
    /// The offset from the end of [Self::buf] we are currently at. This starts at
    /// [Self::buf.len] and counts down to zero.
    buf_end_off: u32,
    /// A scratch Icedx86 encoding buffer used solely by [Self::push_inst] to avoid
    /// reallocations.
    enc: Encoder,
    /// A scratch Icedx86 formatter used solely by [Self::push_inst]. Set to `None` if logging is
    /// not enabled.
    fmtr: Option<NasmFormatter>,
    /// Labels. New labels start with a value of `None`; when they are assigned to an instruction,
    /// this will become `Some(...)`.
    labels: IndexVec<LabelIdx, Option<u32>>,
    /// Operations (CALLs, JMPs), which need relocating. These are not stored in any particular
    /// order.
    relocs: Vec<(u32, u8, RelocKind)>,
    /// If `Some(...)`, log instructions.
    log: Option<Vec<String>>,
}

impl Asm {
    pub(super) fn new(buf: CodeBufInProgress, log: bool) -> Self {
        let buf_used = u32::try_from(buf.len()).unwrap();
        let fmtr = if log {
            let mut fmtr = iced_x86::NasmFormatter::new();
            fmtr.options_mut().set_branch_leading_zeros(false);
            fmtr.options_mut().set_hex_prefix("0x");
            fmtr.options_mut().set_hex_suffix("");
            fmtr.options_mut().set_rip_relative_addresses(true);
            fmtr.options_mut().set_show_branch_size(false);
            fmtr.options_mut().set_space_after_operand_separator(true);
            Some(fmtr)
        } else {
            None
        };
        Asm {
            buf,
            buf_end_off: buf_used,
            enc: Encoder::new(64),
            fmtr,
            labels: index_vec![],
            relocs: Vec::new(),
            log: if log { Some(Vec::new()) } else { None },
        }
    }

    pub(super) fn block_completed(&mut self) {}

    pub(super) fn log(&mut self, s: String) {
        if let Some(x) = &mut self.log {
            x.push(format!("; {s}"))
        }
    }

    /// Create a new free-floating label: it will only be attached when `attach_label` is called on
    /// the label.
    pub(super) fn mk_label(&mut self) -> LabelIdx {
        self.labels.push(None)
    }

    /// Attach `lidx` to the most recently pushed instruction (i.e. the last instruction `push`ed
    /// before `attach_label` is called).
    pub(super) fn attach_label(&mut self, lidx: LabelIdx) {
        if let Some(log) = &mut self.log {
            log.push(format!("; l{}", usize::from(lidx)));
        }
        self.labels[lidx] = Some(self.buf_end_off);
    }

    pub(super) fn align(&mut self, align: u32) {
        self.buf_end_off = (self.buf_end_off / align) * align;
    }

    /// Return the current offset, in bytes, from the end of the code buffer. That end is
    /// guaranteed to be aligned to a page boundary.
    pub(super) fn buf_end_off(&self) -> usize {
        usize::try_from(self.buf_end_off).unwrap()
    }

    /// Push `n` bytes of `nop` instructions.
    pub(super) fn push_nops(&mut self, mut n: usize) {
        // From https://en.wikipedia.org/wiki/NOP_(code)
        while n > 0 {
            let bytes: &[u8] = match n {
                1 => &[0x90],
                2 => &[0x66, 0x90],
                3 => &[0x0F, 0x1F, 0x00],
                4 => &[0x0F, 0x1F, 0x40, 0x00],
                5 => &[0x0F, 0x1F, 0x44, 0x00, 0x00],
                6 => &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
                7 => &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
                8 => &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
                _ => &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            };
            self.buf_end_off = self
                .buf_end_off
                .checked_sub(u32::try_from(bytes.len()).unwrap())
                .expect("Would exceed preallocated code buffer");
            unsafe {
                self.buf
                    .as_ptr()
                    .byte_add(usize::try_from(self.buf_end_off).unwrap())
                    .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
            };
            if let Some(log) = &mut self.log {
                log.push(
                    match n {
                        1 => "nop",
                        2 => "xchg ax, ax",
                        3 => "nop dword ptr [rax]",
                        4 => "nop dword ptr [rax+0x0]",
                        5 => "nop dword ptr [rax+rax*1+0x0]",
                        6 => "nop word ptr [rax+rax*1+0x0]",
                        7 => "nop dword ptr [rax+0x0]",
                        8 => "nop dword ptr [rax+rax*1+0x0]",
                        _ => "nop word ptr [rax+rax*1+0x0]",
                    }
                    .to_string(),
                );
            }
            n -= bytes.len();
        }
    }

    /// Push an icedx64 [Op].
    pub(super) fn push_inst(&mut self, op: Result<Op, iced_x86::IcedError>) {
        let mut inst = op.unwrap();

        // At this point we don't necessarily know where
        // branch/jump/other-memory-displacement operations should go to: if they have an
        // address of 0, we know we'll be relocating the operation later. However, icedx86
        // will (rightfully) complain about address 0, so we stuff in a dummy IP that's nearby, and
        // which we'll patch later.
        let ip = u64::try_from(self.buf.as_ptr().addr()).unwrap();
        if (inst.is_call_near() || inst.is_jmp_near() || inst.is_jcc_near())
            && inst.near_branch64() == 0
        {
            inst.set_near_branch64(ip);
        }
        if inst.is_ip_rel_memory_operand() && inst.memory_displacement64() == 0 {
            inst.set_memory_displacement64(ip);
        }
        self.enc.encode(&inst, ip).unwrap();

        let mut enc_buf = self.enc.take_buffer();
        let len = enc_buf.len();
        self.buf_end_off = self
            .buf_end_off
            .checked_sub(u32::try_from(len).unwrap())
            .expect("Would exceed preallocated code buffer");
        unsafe {
            self.buf
                .as_ptr()
                .byte_add(usize::try_from(self.buf_end_off).unwrap())
                .copy_from_nonoverlapping(enc_buf.as_ptr(), len)
        };
        enc_buf.clear();
        self.enc.set_buffer(enc_buf);

        if let Some(log) = &mut self.log {
            let ip = u64::try_from(
                self.buf.as_ptr().addr() + usize::try_from(self.buf_end_off).unwrap(),
            )
            .unwrap();
            if inst.is_ip_rel_memory_operand() && inst.memory_displacement64() == 0 {
                inst.set_memory_displacement64(ip);
            }
            let mut inst_s = String::new();
            self.fmtr.as_mut().unwrap().format(&inst, &mut inst_s);
            log.push(inst_s);
        }
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
        let old_buf_off = self.buf_end_off;
        self.push_inst(op);

        if let Some(log) = &mut self.log {
            let s = log.last_mut().unwrap();
            let off = s.rfind(' ').unwrap();
            match &reloc {
                RelocKind::AbsoluteWithLabel(lidx) | RelocKind::NearWithLabel(lidx) => {
                    s.replace_range(off.., &format!(" l{}", usize::from(*lidx)));
                }
                RelocKind::NearWithAddr(addr) | RelocKind::NearCallWithAddr(addr) => {
                    s.replace_range(off.., &format!(" 0x{addr:X}"));
                }
            }
        }

        self.relocs.push((
            self.buf_end_off,
            u8::try_from(old_buf_off - self.buf_end_off).unwrap(),
            reloc,
        ));
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
        entry_label: LabelIdx,
        labels: &[LabelIdx],
    ) -> Result<(ExeCodeBuf, Option<String>, Vec<usize>), CompilationError> {
        let base = self.buf.as_ptr().addr();
        let bufs = unsafe { slice::from_raw_parts_mut(self.buf.as_ptr(), self.buf.len()) };
        for (off, inst_len, reloc) in self.relocs {
            let off = usize::try_from(off).unwrap();
            let inst_len = usize::from(inst_len);
            let next_ip = unsafe { self.buf.as_ptr().byte_add(off + inst_len) }.addr();
            // We now have a not-very-nice hack where we examine the instruction to see
            // which part of it we write the relocation to.
            match reloc {
                RelocKind::AbsoluteWithLabel(lidx) => {
                    let patch_off = if let 0x48 | 0x49 = bufs[off] {
                        off + 2 // mov r64, imm64
                    } else {
                        todo!("{:?}", &bufs[off..off + 2])
                    };
                    let addr = base + usize::try_from(self.labels[lidx].unwrap()).unwrap();
                    bufs[patch_off..patch_off + 8].copy_from_slice(&addr.to_le_bytes());
                }
                RelocKind::NearWithAddr(_) | RelocKind::NearWithLabel(_) => {
                    #[allow(clippy::if_same_then_else)]
                    let patch_off = if bufs[off] == 0xe9 {
                        off + 1 // JMP
                    } else if bufs[off] == 0x0F && bufs[off + 1] >= 0x80 && bufs[off + 1] <= 0x08F {
                        off + 2 // JCC
                    } else if bufs[off] == 0x66
                        && bufs[off + 1] == 0x0F
                        && (bufs[off + 2] >= 0x5C && bufs[off + 2] <= 0x62)
                    {
                        off + 4 // PUNPCKLDQ / SUBPD
                    } else if (bufs[off] == 0xF2 || bufs[off] == 0xF3)
                        && bufs[off + 1] == 0x0F
                        && bufs[off + 2] == 0x10
                    {
                        off + 4 // MOVSD / MOVSS
                    } else {
                        todo!("{:X?}", &bufs[off..off + 3])
                    };
                    let addr = match reloc {
                        RelocKind::NearWithAddr(addr) => addr,
                        RelocKind::NearWithLabel(lidx) => {
                            base + usize::try_from(self.labels[lidx].unwrap()).unwrap()
                        }
                        _ => unreachable!(),
                    };
                    let diff = i32::try_from(addr.checked_signed_diff(next_ip).unwrap()).unwrap();
                    if bufs[off] == 0x66 {
                        println!("{patch_off} {diff}");
                    }
                    bufs[patch_off..patch_off + 4].copy_from_slice(&diff.to_le_bytes());
                }
                RelocKind::NearCallWithAddr(addr) => {
                    assert_eq!(bufs[off], 0xE8);
                    let diff = i32::try_from(addr.checked_signed_diff(next_ip).unwrap()).unwrap();
                    bufs[off + 1..off + 1 + 4].copy_from_slice(&diff.to_le_bytes());
                }
            }
        }

        let entry_off = usize::try_from(self.labels[entry_label].unwrap()).unwrap();
        let entry_addr = unsafe { self.buf.as_ptr().byte_add(entry_off) };
        let used = self.buf.len() - usize::try_from(self.buf_end_off).unwrap();
        let exe = unsafe { self.buf.into_execodebuf(used, entry_addr) };
        let log = if let Some(mut log) = self.log.take() {
            log.reverse();
            Some(log.join("\n"))
        } else {
            None
        };
        Ok((
            exe,
            log,
            labels
                .iter()
                .map(|lidx| {
                    usize::try_from(self.labels[*lidx].unwrap() - self.buf_end_off).unwrap()
                })
                .collect::<Vec<_>>(),
        ))
    }
}

/// A relocation kind: these are all relative to the eventual address of the associated operation.
#[derive(Clone, Debug)]
pub(super) enum RelocKind {
    /// A relocation to an absolute address referenced by `LabelIdx`.
    AbsoluteWithLabel(LabelIdx),
    /// A relative branch to the address at `usize`.
    NearWithAddr(usize),
    /// An instruction with a near reference, which must have been validated by
    /// [Asm::is_near_callable]. Using this requires teaching `into_exe` about the specific
    /// encoding and the offset of the address in the instruction. Instructions supported are: JMP,
    /// JCC, MOVSD, MOVSS, PUNPCKLDQ, and SUBPD.
    NearWithLabel(LabelIdx),
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
