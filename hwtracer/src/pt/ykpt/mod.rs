//! The Yk PT trace decoder.
//!
//! The decoder works in two modes:
//!
//!  - compiler-assisted decoding
//!  - disassembly-based decoding
//!
//! The former mode is for decoding portions of the trace that are for "native code" (code compiled
//! by ykllvm). Code built with ykllvm gets static control flow information embedded in the end
//! binary in a special section. Along with dynamic information provided by the PT packet stream,
//! we have everything we need to decode these parts of the trace without disassembling
//! instructions. This is the preferred mode of decoding.
//!
//! The latter mode is a fallback for decoding portions of the trace that are for "foreign code"
//! (code not built with ykllvm). For foreign code there is no static control flow edge information
//! available, so to decode these parts of the trace we have to disassemble the instruction stream
//! (like libipt does).
//!
//! You may now be asking: why not just skip the parts of the trace that are for foreign code?
//! After all, if a portion of code wasn't built with ykllvm, then we won't have IR for it anyway,
//! meaning that the JIT is unable to inline it and we'd have to emit a call into the JIT trace.
//! Why bother decoding the bit of the trace we don't actually care about?
//!
//! The problem is, it's not always easy to identify which parts of a PT trace are for native or
//! foreign code: it's easy if the CPU that doesn't implement the deferred TIP optimisation (see
//! the Intel 64 and IA32 Architectures Software Developer's Manual, Vol 3, Section  32.4.2.3).
//!
//! Deferred TIPs mean that TNT decisions can come in "out of order" with TIP updates. For example,
//! a packet stream `[TNT(0,1), TIP(addr), TNT(1,0)]` may be optimised to `[TNT(0, 1, 1, 0),
//! TIP(addr)]`. If the TIP update to `addr` is the return from foreign code, then when we resume
//! compiler-assisted decoding then we need only the last two TNT decisions in the buffer
//! (discarding the first two as we skip over foreign code). The problem is that without successor
//! block information about the foreign code we can't know how how many TNT decisions correspond
//! with the foreign code, and thus how many decisions to discard.
//!
//! We therefore have to disassemble foreign code, popping TNT decisions as we encounter
//! conditional branch instructions. We can still use compiler-assisted decoding for portions of
//! code that are compiled with ykllvm.

mod packets;
mod parser;

use crate::{
    errors::{HWTracerError, TemporaryErrorKind},
    llvm_blockmap::{BlockMapEntry, SuccessorKind, LLVM_BLOCK_MAP},
    Block,
};
use intervaltree::IntervalTree;
use std::{
    collections::VecDeque,
    convert::TryFrom,
    ffi::CString,
    fmt::{self, Debug},
    ops::Range,
    path::{Path, PathBuf},
    ptr, slice,
    sync::LazyLock,
};
use thiserror::Error;
use ykaddr::{
    self,
    obj::{PHDR_MAIN_OBJ, PHDR_OBJECT_CACHE, SELF_BIN_PATH},
};

use packets::{Bitness, Packet, PacketKind};
use parser::PacketParser;

/// The virtual address ranges of segments that we may need to disassemble.
static CODE_SEGS: LazyLock<CodeSegs> = LazyLock::new(|| {
    let mut segs = Vec::new();
    for obj in PHDR_OBJECT_CACHE.iter() {
        let obj_base = obj.addr();
        for hdr in obj.phdrs() {
            if (hdr.flags() & libc::PF_W) == 0 {
                let vaddr = usize::try_from(obj_base + hdr.vaddr()).unwrap();
                let memsz = usize::try_from(hdr.memsz()).unwrap();
                let key = vaddr..(vaddr + memsz);
                segs.push((key.clone(), ()));
            }
        }
    }
    let tree = segs.into_iter().collect::<IntervalTree<usize, ()>>();
    CodeSegs { tree }
});

/// The number of compressed returns that a CPU implementing Intel Processor Trace can keep track
/// of. This is a bound baked into the hardware, but the decoder needs to be aware of it for its
/// compressed return stack.
const PT_MAX_COMPRETS: usize = 64;

/// A data structure providing convenient access to virtual address ranges and memory slices for
/// segments.
///
/// FIXME: For now this assumes that no dlopen()/dlclose() is happening.
struct CodeSegs {
    tree: IntervalTree<usize, ()>,
}

impl CodeSegs {
    /// Obtain the virtual address range and a slice of memory for the segment containing the
    /// specified virtual address.
    fn seg<'s: 'a, 'a>(&'s self, vaddr: usize) -> Segment<'a> {
        let mut hits = self.tree.query(vaddr..(vaddr + 1));
        match hits.next() {
            Some(x) => {
                // Segments can't overlap.
                debug_assert_eq!(hits.next(), None);

                let slice = unsafe {
                    slice::from_raw_parts(x.range.start as *const u8, x.range.end - x.range.start)
                };
                Segment {
                    vaddrs: &x.range,
                    slice,
                }
            }
            None => todo!(), // Has an object been loaded or unloaded at runtime?
        }
    }
}

/// The virtual address range of and memory slice of one ELF segment.
struct Segment<'a> {
    /// The virtual address range of the segment.
    vaddrs: &'a Range<usize>,
    /// A memory slice for the segment.
    slice: &'a [u8],
}

/// Represents a location in the instruction stream of the traced binary.
#[derive(Eq, PartialEq)]
enum ObjLoc {
    /// A known byte offset in the "main binary object" of the program.
    MainObj(u64),
    /// Anything else, as a virtual address (if known).
    OtherObjOrUnknown(Option<usize>),
}

impl Debug for ObjLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MainObj(v) => write!(f, "ObjLoc::MainObj(0x{:x})", v),
            Self::OtherObjOrUnknown(e) => {
                if let Some(e) = e {
                    write!(f, "ObjLoc::OtherObjOrUnknown(0x{:x})", e)
                } else {
                    write!(f, "ObjLoc::OtherObjOrUnknown(???)")
                }
            }
        }
    }
}

/// The return addresses that can appear on the compressed return stack.
#[derive(Debug, Clone)]
enum CompRetAddr {
    /// A regular return address (as a virtual address).
    VAddr(usize),
    /// Return to directly after the callsite at the given offset in the main object binary.
    ///
    /// This exists because when we do compiler-assisted decoding, we don't disassemble the
    /// instruction stream, and thus we don't know how long the call instruction is, and hence nor
    /// the address of the instruction to return to. That's actually OK, because compiler-assisted
    /// decoding needs only to know after which call to continue decoding after.
    AfterCall(u64),
}

/// The compressed return stack (required for the compressed returns optimisation implemented by
/// some Intel CPUs).
///
/// In short, the call-chains of most programs are "well-behaved" in that their calls and returns
/// are properly nested. In such scenarios, it's not necessary for each return from a function to
/// report a fresh target IP (return address) via a PT packet, since the return address can be
/// inferred from an earlier callsite.
///
/// For more information, consult Section 34.4.2.2 of the Intel 64 and IA-32 Architectures Software
/// Developer’s Manual, Volume 3 (under the "Indirect Transfer Compression for Returns")
/// sub-heading.
struct CompressedReturns {
    rets: VecDeque<CompRetAddr>,
}

impl CompressedReturns {
    fn new() -> Self {
        Self {
            rets: VecDeque::new(),
        }
    }

    fn push(&mut self, ret: CompRetAddr) {
        debug_assert!(self.rets.len() <= PT_MAX_COMPRETS);

        // The stack is fixed-size. When the stack is full and a new entry is pushed, the oldest
        // entry is evicted.
        if self.rets.len() == PT_MAX_COMPRETS {
            self.rets.pop_front();
        }

        self.rets.push_back(ret);
    }

    fn pop(&mut self) -> Option<CompRetAddr> {
        self.rets.pop_back()
    }
}

/// Iterate over the blocks of an Intel PT trace using the fast Yk PT decoder.
pub(crate) struct YkPTBlockIterator<'t> {
    /// The packet iterator used to drive the decoding process.
    parser: PacketParser<'t>,
    /// Keeps track of where we are in the traced binary.
    cur_loc: ObjLoc,
    /// A vector of "taken/not-taken" (TNT) decisions. These arrive in batches and get buffered
    /// here in a FIFO fashion (oldest decision at head poistion).
    tnts: VecDeque<bool>,
    /// The compressed return stack.
    comprets: CompressedReturns,
    /// When `true` we have seen one of more `MODE.*` packets that are yet to be bound.
    unbound_modes: bool,
}

impl<'t> YkPTBlockIterator<'t> {
    pub(crate) fn new(trace: &'t [u8]) -> Self {
        Self {
            parser: PacketParser::new(trace),
            cur_loc: ObjLoc::OtherObjOrUnknown(None),
            tnts: VecDeque::new(),
            comprets: CompressedReturns::new(),
            unbound_modes: false,
        }
    }

    /// Convert a file offset to a virtual address.
    fn off_to_vaddr(&self, obj: &Path, off: u64) -> Result<usize, IteratorError> {
        Ok(ykaddr::addr::off_to_vaddr(obj, off).unwrap())
    }

    /// Convert a virtual address to a file offset.
    fn vaddr_to_off(&self, vaddr: usize) -> Result<(PathBuf, u64), IteratorError> {
        Ok(ykaddr::addr::vaddr_to_obj_and_off(vaddr).unwrap())
    }

    /// Looks up the blockmap entry for the given offset in the "main object binary".
    fn lookup_blockmap_entry(
        &self,
        off: u64,
    ) -> Option<&'static intervaltree::Element<u64, BlockMapEntry>> {
        let mut ents = LLVM_BLOCK_MAP.query(off, off + 1);
        if let Some(ent) = ents.next() {
            // A single-address range cannot span multiple blocks.
            debug_assert!(ents.next().is_none());
            Some(ent)
        } else {
            None
        }
    }

    // Lookup a block from an offset in the "main binary" (i.e. not from a shared object).
    fn lookup_block_from_main_bin_offset(&mut self, off: u64) -> Result<Block, IteratorError> {
        if let Some(ent) = self.lookup_blockmap_entry(off) {
            Ok(Block::from_vaddr_range(
                u64::try_from(self.off_to_vaddr(&PHDR_MAIN_OBJ, ent.range.start)?).unwrap(),
                u64::try_from(self.off_to_vaddr(&PHDR_MAIN_OBJ, ent.range.end)?).unwrap(),
            ))
        } else {
            Ok(Block::Unknown)
        }
    }

    /// Use the blockmap entry `ent` to follow the next (after the offset `b_off`) call in the
    /// block (if one exists).
    ///
    /// Returns `Ok(Some(blk))` if there was a call to follow that lands us in the block `blk`.
    ///
    /// Returns `Ok(None)` if there was no call to follow after `b_off`.
    fn maybe_follow_blockmap_call(
        &mut self,
        b_off: u64,
        ent: &BlockMapEntry,
    ) -> Result<Option<Block>, IteratorError> {
        if let Some(call_info) = ent.call_offs().iter().find(|c| c.callsite_off() >= b_off) {
            let target = call_info.target_off();

            if let Some(target_off) = target {
                // This is a direct call.
                //
                // PT won't compress returns from direct calls if the call target is the
                // instruction address immediately after the call.
                //
                // See the Intel Manual, Section 33.4.2.2 for details.
                if target_off != call_info.return_off() {
                    self.comprets
                        .push(CompRetAddr::AfterCall(call_info.callsite_off()));
                }
                self.cur_loc = ObjLoc::MainObj(target_off);
                return Ok(Some(self.lookup_block_from_main_bin_offset(target_off)?));
            } else {
                // This is an indirect call.
                self.comprets
                    .push(CompRetAddr::AfterCall(call_info.callsite_off()));
                self.seek_tip()?;
                return match self.cur_loc {
                    ObjLoc::MainObj(off) => Ok(Some(self.lookup_block_from_main_bin_offset(off)?)),
                    ObjLoc::OtherObjOrUnknown(_) => Ok(Some(Block::Unknown)),
                };
            }
        }

        Ok(None)
    }

    /// Follow the successor of the block described by the blockmap entry `ent`.
    fn follow_blockmap_successor(&mut self, ent: &BlockMapEntry) -> Result<Block, IteratorError> {
        match ent.successor() {
            SuccessorKind::Unconditional { target } => {
                if let Some(target_off) = target {
                    self.cur_loc = ObjLoc::MainObj(*target_off);
                    self.lookup_block_from_main_bin_offset(*target_off)
                } else {
                    // Divergent control flow.
                    todo!();
                }
            }
            SuccessorKind::Conditional {
                taken_target,
                not_taken_target,
            } => {
                // If we don't have any TNT choices buffered, get more.
                if self.tnts.is_empty() {
                    self.seek_tnt()?;
                }

                // Find where to go next based on whether the trace took the branch or not.
                //
                // The `unwrap()` is guaranteed to succeed because the above call to
                // `seek_tnt()` has populated `self.tnts()`.
                let target_off = if self.tnts.pop_front().unwrap() {
                    *taken_target
                } else if let Some(ntt) = not_taken_target {
                    *ntt
                } else {
                    // Divergent control flow.
                    todo!();
                };
                self.cur_loc = ObjLoc::MainObj(target_off);
                self.lookup_block_from_main_bin_offset(target_off)
            }
            SuccessorKind::Return => {
                if self.is_return_compressed()? {
                    // This unwrap cannot fail if the CPU has implemented compressed
                    // returns correctly.
                    self.cur_loc = match self.comprets.pop().unwrap() {
                        CompRetAddr::AfterCall(off) => ObjLoc::MainObj(off + 1),
                        CompRetAddr::VAddr(vaddr) => {
                            let (obj, off) = self.vaddr_to_off(vaddr)?;
                            if obj == *SELF_BIN_PATH {
                                ObjLoc::MainObj(off)
                            } else {
                                ObjLoc::OtherObjOrUnknown(Some(vaddr))
                            }
                        }
                    };
                    if let ObjLoc::MainObj(off) = self.cur_loc {
                        self.lookup_block_from_main_bin_offset(off + 1)
                    } else {
                        Ok(Block::Unknown)
                    }
                } else {
                    // A regular uncompressed return that relies on a TIP update.
                    //
                    // Note that `is_return_compressed()` has already updated
                    // `self.cur_loc()`.
                    match self.cur_loc {
                        ObjLoc::MainObj(off) => Ok(self.lookup_block_from_main_bin_offset(off)?),
                        _ => Ok(Block::Unknown),
                    }
                }
            }
            SuccessorKind::Dynamic => {
                // We can only know the successor via a TIP update in a packet.
                self.seek_tip()?;
                match self.cur_loc {
                    ObjLoc::MainObj(off) => Ok(self.lookup_block_from_main_bin_offset(off)?),
                    _ => Ok(Block::Unknown),
                }
            }
        }
    }

    fn do_next(&mut self) -> Result<Block, IteratorError> {
        // Read as far ahead as we can using static successor info encoded into the blockmap.
        match self.cur_loc {
            ObjLoc::MainObj(b_off) => {
                // We know where we are in the main object binary, so there's a chance that there's
                // a blockmap entry for this location (not all code from the main object binary
                // necessarily has blockmap info. e.g. PLT resolution routines).
                if let Some(ent) = self.lookup_blockmap_entry(b_off.to_owned()) {
                    // If there are calls in the block that come *after* the current position in the
                    // block, then we will need to follow those before we look at the successor info.
                    if let Some(blk) = self.maybe_follow_blockmap_call(b_off, &ent.value)? {
                        Ok(blk)
                    } else {
                        // If we get here, there were no further calls to follow in the block, so we
                        // consult the static successor information.
                        self.follow_blockmap_successor(&ent.value)
                    }
                } else {
                    self.cur_loc =
                        ObjLoc::OtherObjOrUnknown(Some(self.off_to_vaddr(&PHDR_MAIN_OBJ, b_off)?));
                    Ok(Block::Unknown)
                }
            }
            ObjLoc::OtherObjOrUnknown(vaddr) => self.skip_foreign(vaddr),
        }
    }

    /// Returns the target virtual address for a branch instruction.
    fn branch_target_vaddr(&self, inst: &iced_x86::Instruction) -> u64 {
        match inst.op0_kind() {
            iced_x86::OpKind::NearBranch16 => inst.near_branch16().into(),
            iced_x86::OpKind::NearBranch32 => inst.near_branch32().into(),
            iced_x86::OpKind::NearBranch64 => inst.near_branch64(),
            iced_x86::OpKind::FarBranch16 | iced_x86::OpKind::FarBranch32 => panic!(),
            _ => unreachable!(),
        }
    }

    // Determines if a return from a function was compressed in the packet stream.
    //
    // In the event that the return is compressed, the taken decision is popped from `self.tnts`.
    fn is_return_compressed(&mut self) -> Result<bool, IteratorError> {
        let compressed = if !self.tnts.is_empty() {
            // As the Intel manual explains, when a return is *not* compressed, the CPU's TNT
            // buffers are flushed, so if we have any buffered TNT decisions, then this must be a
            // *compressed* return.
            true
        } else {
            // This *may* be a compressed return. If the next event packet carries a TIP update
            // then this was an uncompressed return, otherwise it was compressed.
            let pkt = self.seek_tnt_or_tip()?;
            pkt.tnts().is_some()
        };

        if compressed {
            // If the return was compressed, we must we consume one "taken=true" decision from the
            // TNT buffer. The unwrap cannot fail because the above code ensures that `self.tnts`
            // is not empty.
            let taken = self.tnts.pop_front().unwrap();
            // FIXME: If you re-enable compressed returns (in `collect.c`), once in a blue moon
            // this assertion will fail.
            //
            // More info: https://github.com/ykjit/yk/issues/874
            debug_assert!(taken);
        }

        Ok(compressed)
    }

    fn disassemble(&mut self, start_vaddr: usize) -> Result<Block, IteratorError> {
        let mut seg = CODE_SEGS.seg(start_vaddr);
        let mut dis =
            iced_x86::Decoder::with_ip(64, seg.slice, u64::try_from(seg.vaddrs.start).unwrap(), 0);
        dis.set_ip(u64::try_from(start_vaddr).unwrap());
        dis.set_position(start_vaddr - seg.vaddrs.start).unwrap();
        let mut reposition: bool = false;

        // `as usize` below are safe casts from raw pointer to pointer-sized integer.
        let longjmp_vaddr = {
            let func = CString::new("longjmp").unwrap();
            u64::try_from(unsafe { libc::dlsym(ptr::null_mut(), func.as_ptr()) } as usize).unwrap()
        };
        let us_longjmp_vaddr = {
            let func = CString::new("_longjmp").unwrap();
            u64::try_from(unsafe { libc::dlsym(ptr::null_mut(), func.as_ptr()) } as usize).unwrap()
        };
        let siglongjmp_vaddr = {
            let func = CString::new("siglongjmp").unwrap();
            u64::try_from(unsafe { libc::dlsym(ptr::null_mut(), func.as_ptr()) } as usize).unwrap()
        };

        loop {
            let vaddr = usize::try_from(dis.ip()).unwrap();
            let (obj, off) = self.vaddr_to_off(vaddr)?;

            if obj == *SELF_BIN_PATH {
                let block = self.lookup_block_from_main_bin_offset(off)?;
                if !block.is_unknown() {
                    // We are back to "native code" and can resume compiler-assisted decoding.
                    self.cur_loc = ObjLoc::MainObj(off);
                    return Ok(block);
                }
            }

            if !seg.vaddrs.contains(&vaddr) {
                // The next instruction is outside of the current segment. Switch segment and make
                // a new decoder for it.
                seg = CODE_SEGS.seg(vaddr);
                let seg_start_u64 = u64::try_from(seg.vaddrs.start).unwrap();
                dis = iced_x86::Decoder::with_ip(64, seg.slice, seg_start_u64, 0);
                dis.set_ip(u64::try_from(vaddr).unwrap());
                reposition = true;
            }

            if reposition {
                dis.set_position(vaddr - seg.vaddrs.start).unwrap();
                reposition = false;
            }

            // We can't (yet) handle longjmp in unmapped code. Crash if we spot it.
            //
            // We do this here, as opposed to at the callsite, because the symbols cannot be
            // reliable detected in the face of PLT trampolines (the symbols are lazily
            // resolved upon first use).
            let vaddr = u64::try_from(vaddr).unwrap();
            if (longjmp_vaddr != 0 && vaddr == longjmp_vaddr)
                || (us_longjmp_vaddr != 0 && vaddr == us_longjmp_vaddr)
                || (siglongjmp_vaddr == 0 && vaddr == siglongjmp_vaddr)
            {
                return Err(IteratorError::HWTracerError(HWTracerError::Unrecoverable(
                    "longjmp within traces currently unsupported".to_string(),
                )));
            }

            let inst = dis.decode();
            match inst.flow_control() {
                iced_x86::FlowControl::Next => (),
                iced_x86::FlowControl::Return => {
                    // We don't expect to see any 16-bit far returns.
                    debug_assert!(is_ret_near(&inst));

                    let ret_vaddr = if self.is_return_compressed()? {
                        // This unwrap cannot fail if the CPU correctly implements compressed
                        // returns.
                        match self.comprets.pop().unwrap() {
                            CompRetAddr::VAddr(vaddr) => vaddr,
                            CompRetAddr::AfterCall(off) => {
                                self.off_to_vaddr(&PHDR_MAIN_OBJ, off + 1)?
                            }
                        }
                    } else {
                        match self.cur_loc {
                            ObjLoc::MainObj(off) => self.off_to_vaddr(&PHDR_MAIN_OBJ, off)?,
                            ObjLoc::OtherObjOrUnknown(opt_vaddr) => match opt_vaddr {
                                Some(vaddr) => vaddr,
                                None => unreachable!(),
                            },
                        }
                    };
                    dis.set_ip(u64::try_from(ret_vaddr).unwrap());
                    reposition = true;
                }
                iced_x86::FlowControl::IndirectBranch | iced_x86::FlowControl::IndirectCall => {
                    self.seek_tip()?;
                    let vaddr = match self.cur_loc {
                        ObjLoc::MainObj(off) => self.off_to_vaddr(&PHDR_MAIN_OBJ, off)?,
                        ObjLoc::OtherObjOrUnknown(opt_vaddr) => match opt_vaddr {
                            Some(vaddr) => vaddr,
                            None => unreachable!(),
                        },
                    };

                    if inst.flow_control() == iced_x86::FlowControl::IndirectCall {
                        debug_assert!(!inst.is_call_far());
                        // Indirect calls, even zero-length ones, are always compressed. See
                        // Section 33.4.2.2 of the Intel Manual:
                        //
                        // "push the next IP onto the stack...note that this excludes zero-length
                        // CALLs, which are *direct* near CALLs with displacement zero (to the next
                        // IP)
                        self.comprets
                            .push(CompRetAddr::VAddr(usize::try_from(inst.next_ip()).unwrap()));
                    }

                    dis.set_ip(u64::try_from(vaddr).unwrap());
                    reposition = true;
                }
                iced_x86::FlowControl::ConditionalBranch => {
                    // Ensure we have TNT decisions buffered.
                    if self.tnts.is_empty() {
                        self.seek_tnt()?;
                    }
                    // unwrap() cannot fail as the above code ensures we have decisions buffered.
                    if self.tnts.pop_front().unwrap() {
                        dis.set_ip(self.branch_target_vaddr(&inst));
                        reposition = true;
                    }
                }
                iced_x86::FlowControl::UnconditionalBranch => {
                    dis.set_ip(self.branch_target_vaddr(&inst));
                    reposition = true;
                }
                iced_x86::FlowControl::Call => {
                    // A *direct* call.
                    if inst.code() == iced_x86::Code::Syscall {
                        // Do nothing. We have disabled kernel tracing in hwtracer, so
                        // entering/leaving a syscall will generate packet generation
                        // disable/enable events (`TIP.PGD`/`TIP.PGE` packets) which are handled by
                        // the decoder elsewhere.
                    } else {
                        let target_vaddr = self.branch_target_vaddr(&inst);

                        // Intel PT doesn't compress a direct call to the next instruction.
                        //
                        // Section 33.4.2.2 of the Intel Manual:
                        //
                        // "For near CALLs, push the Next IP onto the stack... Note that this
                        // excludes zero-length CALLs, which are direct near CALLs with
                        // displacement zero (to the next IP).
                        if target_vaddr != inst.next_ip() {
                            self.comprets
                                .push(CompRetAddr::VAddr(usize::try_from(inst.next_ip()).unwrap()));
                        }
                        // We don't expect to see any 16-bit mode far calls in modernity.
                        debug_assert!(!inst.is_call_far());
                        dis.set_ip(target_vaddr);
                        reposition = true;
                    }
                }
                iced_x86::FlowControl::Interrupt => {
                    // It's my understanding that `INT` instructions aren't really used any more.
                    // Interrupt 0x80 used to be used to do system calls, but now there is the
                    // `SYSCALL` instruction which is generally preferred.
                    unreachable!("interrupt");
                }
                iced_x86::FlowControl::XbeginXabortXend => {
                    // Transactions. These are a bit like time machines for the CPU. They can cause
                    // memory and registers to be rewound to a (dynamically decided) past state.
                    //
                    // FIXME: We might be able to handle these by peeking ahead in the trace, but
                    // let's cross that bridge when we come to it.
                    todo!("transaction instruction: {}", inst);
                }
                iced_x86::FlowControl::Exception => {
                    // We were unable to disassemble the instruction stream to a valid x86_64
                    // instruction. This shouldn't happen, and if it does, I want to know about it!
                    unreachable!("invalid instruction encoding");
                }
            }
        }
    }

    /// Skip over "foreign code" for which we have no blockmap info for.
    fn skip_foreign(&mut self, start_vaddr: Option<usize>) -> Result<Block, IteratorError> {
        let start_vaddr = match start_vaddr {
            Some(v) => v,
            None => {
                // We don't statically know where to start, so we rely on a TIP update to tell us.
                self.seek_tip()?;
                match self.cur_loc {
                    ObjLoc::OtherObjOrUnknown(Some(vaddr)) => vaddr,
                    ObjLoc::OtherObjOrUnknown(None) => {
                        // The above `seek_tip()` ensures this can't happen!
                        unreachable!();
                    }
                    ObjLoc::MainObj(off) => self.off_to_vaddr(&PHDR_MAIN_OBJ, off)?,
                }
            }
        };
        self.disassemble(start_vaddr)
    }

    /// Keep decoding packets until we encounter a TNT packet.
    fn seek_tnt(&mut self) -> Result<(), IteratorError> {
        loop {
            let pkt = self.packet()?; // Potentially populates `self.tnts`.
            if pkt.tnts().is_some() {
                return Ok(());
            }
        }
    }

    /// Keep decoding packets until we encounter one with a TIP update.
    fn seek_tip(&mut self) -> Result<(), IteratorError> {
        loop {
            if self.packet()?.kind().encodes_target_ip() {
                // Note that self.packet() will have update `self.cur_loc`.
                return Ok(());
            }
        }
    }

    /// Keep decoding packets until we encounter either a TNT packet or one with a TIP update.
    ///
    /// The packet is returned so that the consumer can determine which kind of packet was
    /// encountered.
    fn seek_tnt_or_tip(&mut self) -> Result<Packet, IteratorError> {
        loop {
            let pkt = self.packet()?;
            if pkt.kind().encodes_target_ip() || pkt.tnts().is_some() {
                return Ok(pkt);
            }
        }
    }

    /// Skip packets up until and including the next `PSBEND` packet. The first packet after the
    /// `PSBEND` is returned.
    fn skip_psb_plus(&mut self) -> Result<Packet, IteratorError> {
        loop {
            if let Some(pkt_or_err) = self.parser.next() {
                if pkt_or_err?.kind() == PacketKind::PSBEND {
                    break;
                }
            } else {
                panic!("No more packets");
            }
        }

        if let Some(pkt_or_err) = self.parser.next() {
            Ok(pkt_or_err?)
        } else {
            Err(IteratorError::NoMorePackets)
        }
    }

    /// Fetch the next packet and update iterator state.
    fn packet(&mut self) -> Result<Packet, IteratorError> {
        if let Some(pkt_or_err) = self.parser.next() {
            let mut pkt = pkt_or_err?;

            if pkt.kind() == PacketKind::OVF {
                return Err(IteratorError::HWTracerError(HWTracerError::Temporary(
                    TemporaryErrorKind::TraceBufferOverflow,
                )));
            }

            if pkt.kind() == PacketKind::FUP && !self.unbound_modes {
                // FIXME: https://github.com/ykjit/yk/issues/593
                //
                // A FUP packet when there are no outstanding MODE packets indicates that
                // regular control flow was interrupted by an asynchronous event (e.g. a signal
                // handler or a context switch). For now we only support the simple case where
                // execution jumps off to some untraceable foreign code for a while, before
                // returning and resuming where we left off. This is characterised by a [FUP,
                // TIP.PGD, TIP.PGE] sequence (with no intermediate TIP or TNT packets). In
                // this case we can simply ignore the interruption. Later we need to support
                // FUPs more generally.
                pkt = self.seek_tnt_or_tip()?;
                if pkt.kind() != PacketKind::TIPPGD {
                    return Err(IteratorError::HWTracerError(HWTracerError::Temporary(
                        TemporaryErrorKind::TraceInterrupted,
                    )));
                }
                pkt = self.seek_tnt_or_tip()?;
                if pkt.kind() != PacketKind::TIPPGE {
                    return Err(IteratorError::HWTracerError(HWTracerError::Temporary(
                        TemporaryErrorKind::TraceInterrupted,
                    )));
                }
                if let Some(pkt_or_err) = self.parser.next() {
                    pkt = pkt_or_err?;
                } else {
                    return Err(IteratorError::NoMorePackets);
                }
            }

            // Section 33.3.7 of the Intel Manual says that packets in a PSB+ sequence:
            //
            //   "should be interpreted as "status only", since they do not imply any change of
            //   state at the time of the PSB, nor are they associated directly with any
            //   instruction or event. Thus, the normal binding and ordering rules that apply to
            //   these packets outside of PSB+ can be ignored..."
            //
            // So we don't let (e.g.) packets carrying a target ip inside a PSB+ update
            // `self.cur_loc`.
            if pkt.kind() == PacketKind::PSB {
                pkt = self.skip_psb_plus()?;

                // FIXME: Why does clearing the compressed return stack here (as we should) cause
                // non-deterministic crashes?
                //
                // Section 33.3.7 of the Intel Manual explains that:
                //
                //   "the decoder should never need to retain any information (e.g., LastIP, call
                //   stack, compound packet event) across a PSB; all compound packet events will be
                //   completed before a PSB, and any compression state will be reset"
                //
                // The "compression state" it refers to is `self.comprets`, yet:
                //
                //   self.comprets.rets.clear();
                //
                // will causes us to to (sometimes) pop from an empty return stack.
            }

            // If it's a MODE packet, remember we've seen it. The meaning of TIP and FUP packets
            // vary depending upon if they were preceded by MODE packets.
            if pkt.kind().is_mode() {
                // This whole codebase assumes 64-bit mode.
                if let Packet::MODEExec(ref mep) = pkt {
                    debug_assert_eq!(mep.bitness(), Bitness::Bits64);
                }
                self.unbound_modes = true;
            }

            // Does this packet bind to prior MODE packets? If so, it "consumes" the packet.
            if pkt.kind().encodes_target_ip() && self.unbound_modes {
                self.unbound_modes = false;
            }

            // Update `self.target_ip` if necessary.
            if let Some(vaddr) = pkt.target_ip() {
                self.cur_loc = match self.vaddr_to_off(vaddr)? {
                    (obj, off) if obj == *SELF_BIN_PATH => ObjLoc::MainObj(off),
                    _ => ObjLoc::OtherObjOrUnknown(Some(vaddr)),
                };
            }

            // Update `self.tnts` if necessary.
            if let Some(bits) = pkt.tnts() {
                self.tnts.extend(bits);
            }

            Ok(pkt)
        } else {
            Err(IteratorError::NoMorePackets)
        }
    }
}

impl<'t> Iterator for YkPTBlockIterator<'t> {
    type Item = Result<Block, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.do_next() {
            Ok(b) => Some(Ok(b)),
            Err(IteratorError::NoMorePackets) => None,
            Err(IteratorError::HWTracerError(e)) => Some(Err(e)),
        }
    }
}

/// An internal-to-this-module struct which allows the block iterator to distinguish "we reached
/// the end of the packet stream in an expected manner" from more serious errors.
#[derive(Debug, Error)]
enum IteratorError {
    #[error("No more packets")]
    NoMorePackets,
    #[error("HWTracerError: {0}")]
    HWTracerError(HWTracerError),
}

impl From<HWTracerError> for IteratorError {
    fn from(e: HWTracerError) -> Self {
        IteratorError::HWTracerError(e)
    }
}

/// iced_x86 should be providing this:
/// https://github.com/icedland/iced/issues/366
fn is_ret_near(inst: &iced_x86::Instruction) -> bool {
    debug_assert_eq!(inst.flow_control(), iced_x86::FlowControl::Return);
    use iced_x86::Code::*;
    match inst.code() {
        Retnd | Retnd_imm16 | Retnq | Retnq_imm16 | Retnw | Retnw_imm16 => true,
        Retfd | Retfd_imm16 | Retfq | Retfq_imm16 | Retfw | Retfw_imm16 => false,
        _ => unreachable!(), // anything else isn't a return instruction.
    }
}

#[cfg(test)]
mod tests {
    use crate::{perf::PerfCollectorConfig, trace_closure, work_loop, TracerBuilder, TracerKind};

    // FIXME: This test won't work until we teach rustc to embed bitcode and emit a basic block
    // section etc.
    #[ignore]
    #[test]
    /// Trace two loops, one 10x larger than the other, then check the proportions match the number
    /// of block the trace passes through.
    fn ten_times_as_many_blocks() {
        let tc = TracerBuilder::new()
            .tracer_kind(TracerKind::PT(PerfCollectorConfig::default()))
            .build()
            .unwrap();

        let trace1 = trace_closure(&tc, || work_loop(10));
        let trace2 = trace_closure(&tc, || work_loop(100));

        let ct1 = trace1.iter_blocks().count();
        let ct2 = trace2.iter_blocks().count();

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        assert!(ct2 > ct1 * 8);
    }
}
