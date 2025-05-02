//! Intel PT packets and their constituents.

use deku::prelude::*;

/// The `IPBytes` field common to all IP packets.
///
/// This tells us what kind of compression was used for a `TargetIP`.
#[derive(Clone, Copy, Debug, DekuRead)]
pub(super) struct IPBytes {
    #[deku(bits = "3")]
    val: u8,
}

impl IPBytes {
    #[cfg(test)]
    pub(super) fn new(val: u8) -> Self {
        debug_assert!(val >> 3 == 0);
        Self { val }
    }

    /// Returns `true` if we need the previous TIP value to make sense of the new one.
    pub(super) fn needs_prev_tip(&self) -> bool {
        matches!(self.val, 0b001 | 0b010 | 0b100)
    }
}

/// The `TargetIP` fields in packets which update the TIP.
///
/// This is a variable-width field depending upon the value if `IPBytes` in the containing packet.
#[derive(Debug, DekuRead)]
#[deku(id = "ip_bytes_val", ctx = "ip_bytes_val: u8")]
pub(super) enum TargetIP {
    #[deku(id = "0b000")]
    OutOfContext,
    #[deku(id = "0b001")]
    Ip16(u16),
    #[deku(id = "0b010")]
    Ip32(u32),
    #[deku(id_pat = "0b011 | 0b100")]
    Ip48(#[deku(bits = "48")] u64),
    #[deku(id = "0b110")]
    Ip64(u64),
}

impl TargetIP {
    #[cfg(test)]
    pub(super) fn from_bits(bits: u8, val: u64) -> Self {
        match bits {
            0 => Self::OutOfContext,
            16 => Self::Ip16(u16::try_from(val).unwrap()),
            32 => Self::Ip32(u32::try_from(val).unwrap()),
            48 => Self::Ip48(val),
            64 => Self::Ip64(val),
            _ => panic!(),
        }
    }

    /// Decompress a `TargetIP` and `IPBytes` pair into an instruction pointer address.
    ///
    /// Returns `None` if the target IP was "out of context".
    pub(super) fn decompress(&self, ip_bytes: IPBytes, prev_tip: Option<usize>) -> Option<usize> {
        let res = match ip_bytes.val {
            0b000 => {
                debug_assert!(matches!(self, Self::OutOfContext));
                return None;
            }
            0b001 => {
                // The result is bytes 63..=16 from `prev_tip` and bytes 15..=0 from `ip`.
                if let Self::Ip16(v) = self {
                    prev_tip.unwrap() & 0xffffffffffff0000 | usize::from(*v)
                } else {
                    unreachable!();
                }
            }
            0b010 => {
                // The result is bytes 63..=32 from `prev_tip` and bytes 31..=0 from `ip`.
                if let Self::Ip32(v) = self {
                    prev_tip.unwrap() & 0xffffffff00000000 | usize::try_from(*v).unwrap()
                } else {
                    unreachable!();
                }
            }
            0b011 => {
                // The result is bits 0..=47 from the IP, with the remaining high-order bits
                // extended with the value of bit 47.
                if let Self::Ip48(v) = self {
                    debug_assert!(v >> 48 == 0);
                    // Extract the value of bit 47.
                    let b47 = (v & (1 << 47)) >> 47;
                    // Copy the value of bit 47 across all 64 bits.
                    let all = u64::wrapping_sub(!b47 & 0x1, 1);
                    // Restore bits 47..=0 to arrive at the result.
                    usize::try_from(all & 0xffff000000000000 | v).unwrap()
                } else {
                    unreachable!();
                }
            }
            0b100 => todo!(),
            0b101 => unreachable!(), // reserved by Intel.
            0b110 => {
                // Uncompressed IP.
                if let Self::Ip64(v) = self {
                    usize::try_from(*v).unwrap()
                } else {
                    unreachable!();
                }
            }
            0b111 => unreachable!(), // reserved by Intel.
            _ => todo!("IPBytes: {:03b}", ip_bytes.val),
        };
        Some(res)
    }
}

/// Packet Stream Boundary (PSB) packet.
#[derive(Debug, PartialEq, DekuRead)]
#[deku(magic = b"\x02\x82\x02\x82\x02\x82\x02\x82\x02\x82\x02\x82\x02\x82\x02\x82")]
pub(super) struct PSBPacket {}

/// Core Bus Ratio (CBR) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(magic = b"\x02\x03")]
pub(super) struct CBRPacket {
    #[deku(temp)]
    unused: u16,
}

/// End of PSB+ sequence (PSBEND) packet.
#[derive(Debug, DekuRead)]
#[deku(magic = b"\x02\x23")]
pub(super) struct PSBENDPacket {}

/// Padding (PAD) packet.
#[derive(Debug, DekuRead)]
#[deku(magic = b"\x00")]
pub(super) struct PADPacket {}

#[derive(Debug, Eq, PartialEq)]
pub enum Bitness {
    Bits16,
    Bits32,
    Bits64,
}

/// Mode (MODE.Exec) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(magic = b"\x99")]
pub(super) struct MODEExecPacket {
    #[deku(bits = "3", assert = "*magic1 == 0x0", temp)]
    magic1: u8,
    #[deku(bits = "2", temp)]
    reserved: u8,
    #[deku(bits = "1", temp)]
    if_: u8,
    #[deku(bits = "1")]
    csd: u8,
    #[deku(bits = "1")]
    csl_lma: u8,
}

impl MODEExecPacket {
    pub fn bitness(&self) -> Bitness {
        match (self.csd, self.csl_lma) {
            (0, 1) => Bitness::Bits64,
            (1, 0) => Bitness::Bits32,
            (0, 0) => Bitness::Bits16,
            _ => unreachable!(),
        }
    }
}

/// Mode (MODE.TSX) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(magic = b"\x99")]
pub(super) struct MODETSXPacket {
    #[deku(bits = "3", assert = "*magic1 == 0x1", temp)]
    magic1: u8,
    #[deku(bits = "5", temp)]
    unused: u8,
}

/// Packet Generation Enable (TIP.PGE) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct TIPPGEPacket {
    ip_bytes: IPBytes,
    #[deku(bits = "5", assert = "*magic & 0x1f == 0x11", temp)]
    magic: u8,
    #[deku(ctx = "ip_bytes.val")]
    target_ip: TargetIP,
}

impl TIPPGEPacket {
    fn target_ip(&self, prev_tip: Option<usize>) -> Option<usize> {
        self.target_ip.decompress(self.ip_bytes, prev_tip)
    }

    pub(super) fn needs_prev_tip(&self) -> bool {
        self.ip_bytes.needs_prev_tip()
    }
}

/// Short Taken/Not-Taken (TNT) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct ShortTNTPacket {
    /// Bits encoding the branch decisions **and** a stop bit.
    ///
    /// The deku first part of the assertion here (`branches != 0x1`) is subtle: we know that the
    /// `branches` field must contain a stop bit terminating the field, but if the stop bit appears
    /// in place of the first branch, then this is not a short TNT packet at all; it's a long TNT
    /// packet.
    ///
    /// The second part of the assertion (`branches != 0x0`) prevents a pad packet (`0x0`) from
    /// being interpreted as a short TNT with no stop bit.
    #[deku(bits = "7", assert = "*branches != 0x1 && *branches != 0x0")]
    branches: u8,
    #[deku(bits = "1", assert = "!*magic", temp)]
    magic: bool,
}

impl ShortTNTPacket {
    pub(super) fn tnts(&self) -> Vec<bool> {
        let mut push = false;
        let mut tnts = Vec::new();
        for i in (0..7).rev() {
            let bit = (self.branches >> i) & 0x1;
            if !push && bit == 1 {
                // We are witnessing the stop bit. Push from now on.
                push = true;
            } else if push {
                tnts.push(bit == 1)
            }
        }
        assert!(push); // or we didn't see a stop bit!
        assert_ne!(tnts.len(), 0); // No such thing as an empty TNT packet.
        tnts
    }
}

/// Long Taken/Not-Taken (TNT) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(magic = b"\x02\xa3")]
pub(super) struct LongTNTPacket {
    /// Bits encoding the branch decisions **and** a stop bit.
    ///
    /// FIXME: marked `temp` until we actually use the field.
    #[deku(bits = "48", temp)]
    branches: u64,
}

impl LongTNTPacket {
    pub(super) fn tnts(&self) -> Vec<bool> {
        todo!();
    }
}

/// Target IP (TIP) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct TIPPacket {
    ip_bytes: IPBytes,
    #[deku(bits = "5", assert = "*magic & 0x1f == 0x0d", temp)]
    magic: u8,
    #[deku(ctx = "ip_bytes.val")]
    target_ip: TargetIP,
}

impl TIPPacket {
    fn target_ip(&self, prev_tip: Option<usize>) -> Option<usize> {
        self.target_ip.decompress(self.ip_bytes, prev_tip)
    }

    pub(super) fn needs_prev_tip(&self) -> bool {
        self.ip_bytes.needs_prev_tip()
    }
}

/// Packet Generation Disable (TIP.PGD) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct TIPPGDPacket {
    ip_bytes: IPBytes,
    #[deku(bits = "5", assert = "*magic & 0x1f == 0x1", temp)]
    magic: u8,
    #[deku(ctx = "ip_bytes.val")]
    target_ip: TargetIP,
}

impl TIPPGDPacket {
    fn target_ip(&self, prev_tip: Option<usize>) -> Option<usize> {
        self.target_ip.decompress(self.ip_bytes, prev_tip)
    }

    pub(super) fn needs_prev_tip(&self) -> bool {
        self.ip_bytes.needs_prev_tip()
    }
}

/// Flow Update (FUP) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct FUPPacket {
    ip_bytes: IPBytes,
    #[deku(bits = "5", assert = "*magic & 0x1f == 0b11101", temp)]
    magic: u8,
    #[deku(ctx = "ip_bytes.val")]
    target_ip: TargetIP,
}

impl FUPPacket {
    fn target_ip(&self, prev_tip: Option<usize>) -> Option<usize> {
        self.target_ip.decompress(self.ip_bytes, prev_tip)
    }

    pub(super) fn needs_prev_tip(&self) -> bool {
        self.ip_bytes.needs_prev_tip()
    }
}

/// Cycle count (CYC) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct CYCPacket {
    #[deku(bits = "5", temp)]
    unused: u8,
    #[deku(bits = "1", temp)]
    exp: bool,
    #[deku(bits = "2", assert = "*magic & 0x3 == 0b11", temp)]
    magic: u8,
    /// A CYC packet is variable length and has 0 or more "extended" bytes.
    #[deku(bits = 8, cond = "*exp", until = "|e: &u8| e & 0x01 != 0x01", temp)]
    extended: Vec<u8>,
}

/// Execution Stop (EXSTOP) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct EXSTOPPacket {
    #[deku(bits = "8", assert = "*magic1 == 0x2", temp)]
    magic1: u8,
    #[deku(bits = "1", temp)]
    ip: u8,
    #[deku(bits = "7", assert = "*magic2 == 0x2", temp)]
    magic2: u8,
}

/// Overflow (OVF) packet.
#[derive(Debug, PartialEq, DekuRead)]
#[deku(magic = b"\x02\xf3")]
pub(super) struct OVFPacket {}

/// Virtual Machine Control Structure (VMCS) packet.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(super) struct VMCSPacket {
    #[deku(bits = "8", assert = "*magic1 == 0x2", temp)]
    magic1: u8,
    #[deku(bits = "8", assert = "*magic2 == 0b11001000", temp)]
    magic2: u8,
    #[deku(temp)]
    unused: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PacketKind {
    PSB,
    CBR,
    PSBEND,
    PAD,
    MODEExec,
    MODETSX,
    TIPPGE,
    TIPPGD,
    ShortTNT,
    LongTNT,
    TIP,
    FUP,
    CYC,
    EXSTOP,
    OVF,
    VMCS,
}

impl PacketKind {
    /// Returns true if this kind of packet has a field for encoding a target IP.
    pub fn encodes_target_ip(&self) -> bool {
        match self {
            Self::TIPPGE | Self::TIPPGD | Self::TIP | Self::FUP => true,
            Self::PSB
            | Self::CBR
            | Self::PSBEND
            | Self::PAD
            | Self::MODEExec
            | Self::MODETSX
            | Self::ShortTNT
            | Self::LongTNT
            | Self::CYC
            | Self::EXSTOP
            | Self::OVF
            | Self::VMCS => false,
        }
    }

    /// Returns `true` if this packet kind is one of the `MODE.*` packets.
    pub fn is_mode(&self) -> bool {
        match self {
            Self::MODEExec | Self::MODETSX => true,
            Self::CBR
            | Self::CYC
            | Self::FUP
            | Self::LongTNT
            | Self::PAD
            | Self::PSB
            | Self::PSBEND
            | Self::ShortTNT
            | Self::TIP
            | Self::TIPPGD
            | Self::TIPPGE
            | Self::EXSTOP
            | Self::OVF
            | Self::VMCS => false,
        }
    }
}

/// The top-level representation of an Intel Processor Trace packet.
///
/// Variants with an `Option<usize>` may cache the previous TIP value (at the time the packet was
/// created). This may be needed to get the updated TIP value from the packet.
#[derive(Debug)]
pub(super) enum Packet {
    PSB(PSBPacket),
    CBR(CBRPacket),
    PSBEND(PSBENDPacket),
    PAD(PADPacket),
    MODEExec(MODEExecPacket),
    MODETSX(MODETSXPacket),
    TIPPGE(TIPPGEPacket, Option<usize>),
    TIPPGD(TIPPGDPacket, Option<usize>),
    ShortTNT(ShortTNTPacket),
    LongTNT(LongTNTPacket),
    TIP(TIPPacket, Option<usize>),
    FUP(FUPPacket, Option<usize>),
    CYC(CYCPacket),
    EXSTOP(EXSTOPPacket),
    OVF(OVFPacket),
    VMCS(VMCSPacket),
}

impl Packet {
    /// If the packet contains a (non "out of context") TIP update, return the IP value.
    pub(super) fn target_ip(&self) -> Option<usize> {
        match self {
            Self::TIPPGE(p, prev_tip) => p.target_ip(*prev_tip),
            Self::TIPPGD(p, prev_tip) => p.target_ip(*prev_tip),
            Self::TIP(p, prev_tip) => p.target_ip(*prev_tip),
            Self::FUP(p, prev_tip) => p.target_ip(*prev_tip),
            Self::PSB(_)
            | Self::CBR(_)
            | Self::PSBEND(_)
            | Self::PAD(_)
            | Self::MODEExec(_)
            | Self::MODETSX(_)
            | Self::ShortTNT(_)
            | Self::LongTNT(_)
            | Self::CYC(_)
            | Self::EXSTOP(_)
            | Self::OVF(_)
            | Self::VMCS(_) => None,
        }
    }

    pub(super) fn kind(&self) -> PacketKind {
        match self {
            Self::PSB(_) => PacketKind::PSB,
            Self::CBR(_) => PacketKind::CBR,
            Self::PSBEND(_) => PacketKind::PSBEND,
            Self::PAD(_) => PacketKind::PAD,
            Self::MODEExec(_) => PacketKind::MODEExec,
            Self::MODETSX(_) => PacketKind::MODETSX,
            Self::TIPPGE(..) => PacketKind::TIPPGE,
            Self::TIPPGD(..) => PacketKind::TIPPGD,
            Self::ShortTNT(_) => PacketKind::ShortTNT,
            Self::LongTNT(_) => PacketKind::LongTNT,
            Self::TIP(..) => PacketKind::TIP,
            Self::FUP(..) => PacketKind::FUP,
            Self::CYC(_) => PacketKind::CYC,
            Self::EXSTOP(_) => PacketKind::EXSTOP,
            Self::OVF(_) => PacketKind::OVF,
            Self::VMCS(_) => PacketKind::VMCS,
        }
    }

    /// Extract the taken/not-taken decisions from a TNT packet.
    ///
    /// Returns `None` if the packet is not a TNT packet.
    ///
    /// OPT: Use a bit-field instead of a vector.
    pub(super) fn tnts(&self) -> Option<Vec<bool>> {
        match self {
            Self::ShortTNT(p) => Some(p.tnts()),
            Self::LongTNT(p) => Some(p.tnts()),
            Self::PSB(_)
            | Self::CBR(_)
            | Self::PSBEND(_)
            | Self::PAD(_)
            | Self::MODEExec(_)
            | Self::MODETSX(_)
            | Self::TIPPGE(_, _)
            | Self::TIPPGD(_, _)
            | Self::TIP(_, _)
            | Self::FUP(_, _)
            | Self::CYC(_)
            | Self::EXSTOP(_)
            | Self::OVF(_)
            | Self::VMCS(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ShortTNTPacket;

    #[test]
    fn short_tnt() {
        let mk = |branches: u8| ShortTNTPacket { branches };

        debug_assert_eq!(mk(0b1000000).tnts(), vec![false; 6]);
        debug_assert_eq!(mk(0b1111111).tnts(), vec![true; 6]);
        debug_assert_eq!(mk(0b0000011).tnts(), vec![true]);
        debug_assert_eq!(mk(0b0000010).tnts(), vec![false]);
        debug_assert_eq!(
            mk(0b1001001).tnts(),
            vec![false, false, true, false, false, true]
        );
    }
}
