//! The Yk PT trace decoder.

use crate::{
    decode::TraceDecoder,
    errors::HWTracerError,
    llvm_blockmap::{BlockMapEntry, SuccessorKind, LLVM_BLOCK_MAP},
    Block, Trace,
};
use intervaltree;
use std::{
    collections::VecDeque,
    convert::TryFrom,
    fmt::{self, Debug},
    path::PathBuf,
};
use ykutil::{
    addr::{code_vaddr_to_off, off_to_vaddr},
    obj::SELF_BIN_PATH,
};

mod packet_parser;
use packet_parser::{
    packets::{Packet, PacketKind},
    PacketParser,
};

pub(crate) struct YkPTTraceDecoder {}

impl TraceDecoder for YkPTTraceDecoder {
    fn new() -> Self {
        Self {}
    }

    fn iter_blocks<'t>(
        &'t self,
        trace: &'t dyn Trace,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + '_> {
        let itr = YkPTBlockIterator {
            errored: false,
            parser: PacketParser::new(trace.bytes()),
            cur_loc: ObjLoc::OtherObjOrUnknown,
            tnts: VecDeque::new(),
            pge: false,
        };
        Box::new(itr)
    }
}

/// Represents a location in the instruction stream of the traced binary.
#[derive(Eq, PartialEq)]
enum ObjLoc {
    /// A known location in the "main binary object" of the program.
    MainObj(u64),
    /// Either a location not in the "main binary object", or an unknown location.
    OtherObjOrUnknown,
}

impl Debug for ObjLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MainObj(v) => write!(f, "MainObj[0x{:x}]", v),
            Self::OtherObjOrUnknown => write!(f, "OtherObjOrUnknown"),
        }
    }
}

/// Iterate over the blocks of an Intel PT trace using the fast Yk PT decoder.
struct YkPTBlockIterator<'t> {
    /// Set to true when an error has occured.
    errored: bool,
    /// The packet iterator used to drive the decoding process.
    parser: PacketParser<'t>,
    /// Keeps track of where we are in the traced binary.
    cur_loc: ObjLoc,
    /// A vector of "taken/not-taken" (TNT) decisions. These arrive in batches and get buffered
    /// here in a FIFO fashion (oldest decision at head poistion).
    tnts: VecDeque<bool>,
    /// When true, packet generation is enabled (we've seen a `TIP.PGE` packet, but no
    /// corresponding `TIP.PGD` yet).
    pge: bool,
}

impl<'t> YkPTBlockIterator<'t> {
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
    fn lookup_block_from_main_bin_offset(&self, off: u64) -> Block {
        if let Some(ent) = self.lookup_blockmap_entry(off) {
            Block::from_vaddr_range(
                u64::try_from(off_to_vaddr(&PathBuf::from(""), ent.range.start).unwrap()).unwrap(),
                u64::try_from(off_to_vaddr(&PathBuf::from(""), ent.range.end).unwrap()).unwrap(),
            )
        } else {
            Block::unknown()
        }
    }

    fn do_next(&mut self) -> Result<Option<Block>, HWTracerError> {
        // Read as far ahead as we can using static successor info encoded into the blockmap.
        if let ObjLoc::MainObj(b_off) = self.cur_loc {
            // We know where we are in the main object binary, so there's a chance that there's a
            // blockmap entry for this location.
            if let Some(ent) = self.lookup_blockmap_entry(b_off.to_owned()) {
                // If there are calls in the block that come *after* the current position in the
                // block, then we will need to follow those before we look at the successor info.
                if let Some(call_info) = ent
                    .value
                    .call_offs()
                    .iter()
                    .find(|c| c.callsite_off() >= b_off)
                {
                    let target = call_info.target_off();
                    if let Some(target_off) = target {
                        self.cur_loc = ObjLoc::MainObj(target_off);
                        return Ok(Some(self.lookup_block_from_main_bin_offset(target_off)));
                    } else {
                        // Call target isn't known statically.
                        self.cur_loc = ObjLoc::OtherObjOrUnknown;
                        return Ok(Some(Block::unknown()));
                    }
                }

                // If we get here, there were no further calls to follow in the block, so we
                // consult the static successor information.
                match ent.value.successor() {
                    SuccessorKind::Unconditional { target } => {
                        if let Some(target_off) = target {
                            self.cur_loc = ObjLoc::MainObj(*target_off);
                            return Ok(Some(self.lookup_block_from_main_bin_offset(*target_off)));
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
                        if self.tnts.is_empty() && !self.seek_tnt()? {
                            // The packet stream was exhausted.
                            return Ok(None);
                        }
                        // Find where to go next based on whether the trace took the branch or not.
                        //
                        // The `unwrap()` is guaranteed to succeed because the above call to
                        // `seek_tnt()` has populated `self.tnts()`.
                        let target_off = if self.tnts.pop_front().unwrap() {
                            *taken_target
                        } else {
                            if let Some(ntt) = not_taken_target {
                                *ntt
                            } else {
                                // Divergent control flow.
                                todo!();
                            }
                        };
                        self.cur_loc = ObjLoc::MainObj(target_off);
                        return Ok(Some(self.lookup_block_from_main_bin_offset(target_off)));
                    }
                    SuccessorKind::Dynamic => {
                        // We can only know the successor via a TIP update in a packet. Fall
                        // through to `self.seek_tip()`.
                        self.cur_loc = ObjLoc::OtherObjOrUnknown;
                    }
                }
            } else {
                // In the absence of blockmap info, we cannot know the successor block.
                self.cur_loc = ObjLoc::OtherObjOrUnknown;
                return Ok(Some(Block::unknown()));
            }
        }

        // If we get here then we can't statically figure out where to go next. A packet is going
        // to have to tell us what to do.
        self.seek_tip()
    }

    /// Keep decoding packets until we have some TNT decisions buffered.
    fn seek_tnt(&mut self) -> Result<bool, HWTracerError> {
        while self.tnts.is_empty() {
            let pkt = self.packet()?; // Potentially populates `self.tnts`.
            if pkt.is_none() {
                return Ok(false); // Packet stream exhausted.
            }
        }
        Ok(true)
    }

    /// Keep decoding packets until we receive a TIP update.
    fn seek_tip(&mut self) -> Result<Option<Block>, HWTracerError> {
        debug_assert_eq!(self.cur_loc, ObjLoc::OtherObjOrUnknown);
        loop {
            if self.packet()?.is_some() {
                if let ObjLoc::MainObj(off) = self.cur_loc {
                    return Ok(Some(self.lookup_block_from_main_bin_offset(off)));
                }
            } else {
                return Ok(None); // No more packets.
            }
        }
    }

    /// Fetch the next packet and update iterator state.
    fn packet(&mut self) -> Result<Option<Packet>, HWTracerError> {
        if let Some(pkt_or_err) = self.parser.next() {
            let pkt = pkt_or_err?;

            // Update `self.pge` if necessary.
            match pkt.kind() {
                PacketKind::TIPPGE => {
                    debug_assert!(!self.pge);
                    self.pge = true;
                }
                PacketKind::TIPPGD => {
                    debug_assert!(self.pge);
                    self.pge = false;
                }
                _ => (),
            }

            // Update `self.target_ip` if necessary.
            if let Some(vaddr) = pkt.target_ip() {
                if self.pge {
                    match code_vaddr_to_off(vaddr) {
                        Some((obj, off)) if obj == *SELF_BIN_PATH => {
                            self.cur_loc = ObjLoc::MainObj(off)
                        }
                        _ => self.cur_loc = ObjLoc::OtherObjOrUnknown,
                    }

                    // At this point any remaining buffered TNTs would be ones for past control
                    // flow decisions that we were unable to make use of due to imperfect
                    // knowledge of the control flow. E.g. execution passed through code that
                    // wasn't compiled by ykllvm, so we get TNT packets, but no static successor
                    // info to use them against.
                    self.tnts.clear();
                }
            }

            // Update `self.tnts` if necessary.
            if let Some(bits) = pkt.tnts() {
                self.tnts.extend(bits);
            }

            Ok(Some(pkt))
        } else {
            Ok(None)
        }
    }
}

impl<'t> Iterator for YkPTBlockIterator<'t> {
    type Item = Result<Block, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.errored {
            match self.do_next() {
                Ok(Some(blk)) => Some(Ok(blk)),
                Ok(None) => None,
                Err(e) => {
                    self.errored = true;
                    Some(Err(e))
                }
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collect::TraceCollectorBuilder,
        decode::{test_helpers, TraceDecoderKind},
    };

    #[ignore] // FIXME
    #[test]
    fn ten_times_as_many_blocks() {
        let tc = TraceCollectorBuilder::new().build().unwrap();
        test_helpers::ten_times_as_many_blocks(tc, TraceDecoderKind::YkPT);
    }
}
