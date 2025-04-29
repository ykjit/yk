//! A packet parser for the Yk PT trace decoder.

use crate::errors::HWTracerError;
use deku::prelude::*;
use std::cmp::min;

use super::packets::*;

#[derive(Clone, Copy, Debug)]
enum PacketParserState {
    /// The "normal" decoding state.
    Normal,
    /// We are decoding a PSB+ sequence.
    PSBPlus,
}

impl PacketParserState {
    /// Returns the kinds of packet that are valid for the state.
    fn valid_packets(&self) -> &'static [PacketKind] {
        // Note that the parser will attempt to match packet kinds in the order that they appear in
        // the returned slice. For best performance, the returned slice should be sorted, most
        // frequently expected packet kinds first.
        //
        // OPT: The order below is a rough guess based on what limited traces I've seen. Benchmark
        // and optimise.
        match self {
            Self::Normal => &[
                PacketKind::ShortTNT,
                PacketKind::PAD,
                PacketKind::FUP,
                PacketKind::TIP,
                PacketKind::CYC,
                PacketKind::LongTNT,
                PacketKind::PSB,
                PacketKind::MODEExec,
                PacketKind::MODETSX,
                PacketKind::CBR,
                PacketKind::TIPPGE,
                PacketKind::TIPPGD,
                PacketKind::EXSTOP,
                PacketKind::OVF,
            ],
            Self::PSBPlus => &[
                PacketKind::PAD,
                PacketKind::CBR,
                PacketKind::FUP,
                PacketKind::MODEExec,
                PacketKind::MODETSX,
                PacketKind::PSBEND,
                PacketKind::OVF,
                PacketKind::VMCS,
            ],
        }
    }

    /// Check if the parser needs to transition to a new state as a result of parsing a certain
    /// kind of packet.
    fn transition(&mut self, pkt_kind: PacketKind) {
        let new = match (*self, pkt_kind) {
            (Self::Normal, PacketKind::PSB) => Self::PSBPlus,
            (Self::PSBPlus, PacketKind::PSBEND) => Self::Normal,
            _ => return, // No state transition.
        };
        *self = new;
    }
}

pub(super) struct PacketParser<'t> {
    /// The remaining raw bytes of the PT trace we need to parse.
    ///
    /// This slice is updated in-place after a packet's worth of bytes is consumed.
    pt_bytes: &'t [u8],
    /// The parser operates as a state machine. This field keeps track of which state we are in.
    state: PacketParserState,
    /// The most recent Target IP (TIP) value that we've seen. This is needed because updated TIP
    /// values are sometimes compressed using bits from the previous TIP value.
    prev_tip: usize,
}

/// Attempt to read the packet of type `$packet` using deku. On success wrap the packet up into the
/// corresponding discriminant of `Packet`.
macro_rules! read_to_packet {
    ($packet: ty, $pt_bytes: expr, $discr: expr) => {
        <$packet>::from_bytes(($pt_bytes, 0)).and_then(|(r, p)| Ok((r, $discr(p))))
    };
}

/// Same as `read_to_packet!`, but with extra logic for dealing with packets which encode a TIP.
macro_rules! read_to_packet_tip {
    ($packet: ty, $pt_bytes: expr, $discr: expr, $prev_tip: expr) => {
        <$packet>::from_bytes(($pt_bytes, 0)).and_then(|(r, p)| {
            let ret = if p.needs_prev_tip() {
                Ok((r, $discr(p, Some($prev_tip))))
            } else {
                Ok((r, $discr(p, None)))
            };
            ret
        })
    };
}

impl<'t> PacketParser<'t> {
    pub(super) fn new(bytes: &'t [u8]) -> Self {
        Self {
            pt_bytes: bytes,
            state: PacketParserState::Normal,
            prev_tip: 0,
        }
    }

    /// Attempt to parse a packet of the specified `PacketKind`.
    fn parse_kind(&mut self, kind: PacketKind) -> Option<Packet> {
        let parse_res = match kind {
            PacketKind::PSB => {
                read_to_packet!(PSBPacket, self.pt_bytes, Packet::PSB)
            }
            PacketKind::CBR => read_to_packet!(CBRPacket, self.pt_bytes, Packet::CBR),
            PacketKind::PSBEND => {
                read_to_packet!(PSBENDPacket, self.pt_bytes, Packet::PSBEND)
            }
            PacketKind::PAD => read_to_packet!(PADPacket, self.pt_bytes, Packet::PAD),
            PacketKind::MODEExec => {
                read_to_packet!(MODEExecPacket, self.pt_bytes, Packet::MODEExec)
            }
            PacketKind::MODETSX => {
                read_to_packet!(MODETSXPacket, self.pt_bytes, Packet::MODETSX)
            }
            PacketKind::TIPPGE => {
                read_to_packet_tip!(TIPPGEPacket, self.pt_bytes, Packet::TIPPGE, self.prev_tip)
            }
            PacketKind::TIPPGD => {
                read_to_packet_tip!(TIPPGDPacket, self.pt_bytes, Packet::TIPPGD, self.prev_tip)
            }
            PacketKind::ShortTNT => {
                read_to_packet!(ShortTNTPacket, self.pt_bytes, Packet::ShortTNT)
            }
            PacketKind::LongTNT => {
                read_to_packet!(LongTNTPacket, self.pt_bytes, Packet::LongTNT)
            }
            PacketKind::TIP => {
                read_to_packet_tip!(TIPPacket, self.pt_bytes, Packet::TIP, self.prev_tip)
            }
            PacketKind::FUP => {
                read_to_packet_tip!(FUPPacket, self.pt_bytes, Packet::FUP, self.prev_tip)
            }
            PacketKind::CYC => read_to_packet!(CYCPacket, self.pt_bytes, Packet::CYC),
            PacketKind::EXSTOP => {
                read_to_packet!(EXSTOPPacket, self.pt_bytes, Packet::EXSTOP)
            }
            PacketKind::OVF => read_to_packet!(OVFPacket, self.pt_bytes, Packet::OVF),
            PacketKind::VMCS => read_to_packet!(VMCSPacket, self.pt_bytes, Packet::VMCS),
        };
        if let Ok((remain, pkt)) = parse_res {
            // PT packets are always byte-aligned, so the bit offset returned should always be
            // zero.
            assert_eq!(remain.1, 0);
            // Update `self.pt_bytes` with the remaining, yet-to-be-consumed bytes.
            self.pt_bytes = remain.0;
            Some(pkt)
        } else {
            None
        }
    }

    /// Attempt to parse a packet for the current parser state.
    fn parse_state(&mut self) -> Result<Packet, HWTracerError> {
        for kind in self.state.valid_packets() {
            if let Some(pkt) = self.parse_kind(*kind) {
                if *kind == PacketKind::PSBEND {
                    self.state = PacketParserState::Normal;
                }
                return Ok(pkt);
            }
        }
        panic!(
            "In state {:?}, failed to parse packet from bytes: {}",
            self.state,
            self.byte_stream_str(8, ", ")
        );
    }

    /// Returns a string showing a binary formatted peek at the next `nbytes` bytes of
    /// `self.bytes`. Bytes in the output are separated by `sep`.
    ///
    /// This is used to format error messages, but is also useful when debugging.
    fn byte_stream_str(&self, nbytes: usize, sep: &str) -> String {
        let nbytes = min(nbytes, self.pt_bytes.len());
        let mut bytes = self.pt_bytes.iter();
        let mut vals = Vec::new();
        for _ in 0..nbytes {
            vals.push(format!("{:08b}", bytes.next().unwrap()));
        }

        if bytes.len() > nbytes {
            vals.push("...".to_owned());
        }

        vals.join(sep)
    }

    /// Attempt to parse a packet.
    fn parse_packet(&mut self) -> Result<Packet, HWTracerError> {
        // Attempt to parse a packet.
        let pkt = self.parse_state()?;

        // If the packet contains an updated TIP, then cache it.
        if let Some(tip) = pkt.target_ip() {
            self.prev_tip = tip;
        }

        // See if the packet we just parsed triggers a state transition.
        self.state.transition(pkt.kind());

        Ok(pkt)
    }
}

impl Iterator for PacketParser<'_> {
    type Item = Result<Packet, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.pt_bytes.is_empty() {
            let p = self.parse_packet();
            Some(p)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::packets::*, PacketParser};
    use crate::{trace_closure, work_loop, TracerBuilder, TracerKind};

    /// Parse the packets of a small trace, checking the basic structure of the decoded trace.
    #[test]
    fn parse_small_trace() {
        let tc = TracerBuilder::new()
            .tracer_kind(TracerKind::PT(crate::perf::PerfCollectorConfig::default()))
            .build()
            .unwrap();
        let trace = trace_closure(&tc, || work_loop(3));

        #[derive(Clone, Copy, Debug)]
        enum TestState {
            /// Start here.
            Init,
            /// Saw the start of the PSB+ sequence.
            SawPSBPlusStart,
            /// Saw the end of the PSB+ sequence.
            SawPSBPlusEnd,
            /// Saw the packet generation enable packet.
            SawPacketGenEnable,
            /// Saw a TNT packet.
            SawTNT,
            /// Saw the packet generation disable packet.
            SawPacketGenDisable,
        }

        let mut ts = TestState::Init;
        for pkt in PacketParser::new(trace.bytes()) {
            ts = match (ts, pkt.unwrap().kind()) {
                (TestState::Init, PacketKind::PSB) => TestState::SawPSBPlusStart,
                (TestState::SawPSBPlusStart, PacketKind::PSBEND) => TestState::SawPSBPlusEnd,
                (TestState::SawPSBPlusEnd, PacketKind::TIPPGE) => TestState::SawPacketGenEnable,
                (TestState::SawPacketGenEnable, PacketKind::ShortTNT)
                | (TestState::SawPacketGenEnable, PacketKind::LongTNT) => TestState::SawTNT,
                (TestState::SawTNT, PacketKind::TIPPGD) => TestState::SawPacketGenDisable,
                (ts, _) => ts,
            };
        }
        assert!(matches!(ts, TestState::SawPacketGenDisable));
    }

    /// Test target IP decompression when the `IPBytes = 0b000`.
    #[test]
    fn ipbytes_decompress_000() {
        let ipbytes0 = IPBytes::new(0b000);
        assert_eq!(
            TargetIP::from_bits(0, 0).decompress(ipbytes0, Some(0xdeafcafedeadcafe)),
            None
        );
    }

    /// Test target IP decompression when the `IPBytes = 0b001`.
    #[test]
    fn ipbytes_decompress_001() {
        let ipb = IPBytes::new(0b001);
        assert_eq!(
            TargetIP::from_bits(16, 0x000000000000cccc).decompress(ipb, Some(0xa1a2a3a4a5a69999)),
            Some(0xa1a2a3a4a5a6cccc)
        );
    }

    /// Test target IP decompression when the `IPBytes = 0b010`.
    #[test]
    fn ipbytes_decompress_010() {
        let ipb = IPBytes::new(0b010);
        assert_eq!(
            TargetIP::from_bits(32, 0x00000000bbbbbbbb).decompress(ipb, Some(0xcccccccc99999999)),
            Some(0xccccccccbbbbbbbb)
        );
    }

    /// Test target IP decompression when the `IPBytes = 0b011`.
    #[test]
    fn ipbytes_decompress_011() {
        let ipb = IPBytes::new(0b011);

        // Bit 47 zero-extend.
        assert_eq!(TargetIP::from_bits(48, 0).decompress(ipb, None), Some(0));
        assert_eq!(
            TargetIP::from_bits(48, 0x0000010203040506).decompress(ipb, None),
            Some(0x0000010203040506)
        );

        // Bit 47 one-extend.
        assert_eq!(
            TargetIP::from_bits(48, 1 << 47).decompress(ipb, None),
            Some(0xffff800000000000)
        );
        assert_eq!(
            TargetIP::from_bits(48, 0x0000887766554433).decompress(ipb, None),
            Some(0xffff887766554433)
        );
    }
}
