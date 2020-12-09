//! Loading and tracing of Serialised Intermediate Representation (SIR).

use fallible_iterator::FallibleIterator;
use fxhash::FxHashMap;
use memmap::Mmap;
use object::{Object, ObjectSection};
use std::{
    convert::TryFrom,
    env,
    fmt::{self, Debug, Display, Write},
    fs::File,
    io::{Cursor, Seek, SeekFrom},
    iter::Iterator
};
use ykpack::{self, Body, BodyFlags, CguHash, Decoder, Pack, SirHeader, SirOffset};

lazy_static! {
    pub static ref EXE_MMAP: Mmap =
        unsafe { Mmap::map(&File::open(&env::current_exe().unwrap()).unwrap()).unwrap() };
    pub static ref SIR: Sir<'static> = Sir::new(&*EXE_MMAP).unwrap();
}

/// An interface to the serialised IR of an executable.
///
/// One of these structures is generated in the above `lazy_static` and is then shared immutably
/// across all threads. Only the headers of each SIR section are eagerly loaded. For performance
/// reasons, the actual IR is loaded on-demand.
#[derive(Debug)]
pub struct Sir<'m> {
    /// The SIR section headers.
    /// Maps a codegen unit hash to a `(section-name, header, header-size)` tuple.
    hdrs: FxHashMap<CguHash, (String, SirHeader, SirOffset)>,
    /// The current executable's ELF information.
    exe_obj: object::File<'m>,
    /// Section cache to avoid expensive `object::File::section_by_name()` calls.
    sec_cache: FxHashMap<String, &'m [u8]>
}

impl<'m> Sir<'m> {
    pub fn new(mmap: &'m Mmap) -> Result<Self, ()> {
        // SAFETY: Not really, we hope that nobody changes the file underneath our feet.
        let mut hdrs = FxHashMap::default();
        let mut sec_cache = FxHashMap::default();
        let exe_obj = object::File::parse(&*mmap).unwrap();
        for sec in exe_obj.sections() {
            let sec_name = sec.name().unwrap();
            let sec_data = sec.data().unwrap();
            sec_cache.insert(sec_name.to_owned(), sec_data);
            if sec_name.starts_with(ykpack::SIR_SECTION_PREFIX) {
                let mut curs = Cursor::new(sec_data);
                let mut dec = Decoder::from(&mut curs);
                let hdr = if let Pack::Header(hdr) = dec.next().unwrap().unwrap() {
                    hdr
                } else {
                    panic!("missing sir header");
                };
                let hdr_size = usize::try_from(curs.seek(SeekFrom::Current(0)).unwrap()).unwrap();
                hdrs.insert(hdr.cgu_hash, (sec_name.to_owned(), hdr, hdr_size));
            }
        }
        Ok(Self {
            hdrs,
            exe_obj,
            sec_cache
        })
    }

    fn cursor_for_section(&self, sec_name: &str) -> Cursor<&[u8]> {
        Cursor::new(self.sec_cache[sec_name])
    }

    /// Decode a type in a named section, at an absolute offset from the beginning of that section.
    fn decode_ty(&self, sec_name: &str, off: SirOffset) -> ykpack::Ty {
        let mut curs = self.cursor_for_section(&sec_name);
        curs.seek(SeekFrom::Start(u64::try_from(off).unwrap()))
            .unwrap();
        let mut dec = Decoder::from(&mut curs);
        if let Ok(Some(Pack::Type(t))) = dec.next() {
            t
        } else {
            panic!("Failed to deserialize SIR type");
        }
    }

    /// Get the type data for the given type ID.
    pub fn ty(&self, tyid: &ykpack::TypeId) -> ykpack::Ty {
        let (cgu, tidx) = tyid;
        let (ref sec_name, ref hdr, hdr_size) = SIR.hdrs[cgu];
        let off = hdr.types[usize::try_from(*tidx).unwrap()];
        self.decode_ty(sec_name, hdr_size + off)
    }

    /// Decode a body in a named section, at an absolute offset from the beginning of that section.
    fn decode_body(&self, sec_name: &str, off: SirOffset) -> ykpack::Body {
        let mut curs = self.cursor_for_section(&sec_name);
        curs.seek(SeekFrom::Start(u64::try_from(off).unwrap()))
            .unwrap();
        let mut dec = Decoder::from(&mut curs);
        if let Ok(Some(Pack::Body(body))) = dec.next() {
            body
        } else {
            panic!("Failed to deserialize SIR body");
        }
    }

    /// Get the body data for the given symbol name.
    /// Returns None if not found.
    pub fn body(&self, body_sym: &str) -> Option<Body> {
        for (sec_name, hdr, hdr_size) in SIR.hdrs.values() {
            if let Some(off) = hdr.bodies.get(body_sym) {
                return Some(self.decode_body(sec_name, hdr_size + off));
            }
        }
        None
    }
}

impl<'m> Display for Sir<'m> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (sec_name, hdr, hdr_size) in self.hdrs.values() {
            writeln!(
                f,
                "# Types for CGU {} in section {}:",
                hdr.cgu_hash, sec_name
            )?;
            for (idx, off) in hdr.types.iter().enumerate() {
                writeln!(
                    f,
                    "  {} => {}",
                    idx,
                    self.decode_ty(sec_name, hdr_size + off)
                )?;
            }
            writeln!(
                f,
                "# Bodies for CGU {} in section {}",
                hdr.cgu_hash, sec_name
            )?;
            for off in hdr.bodies.values() {
                let txt = format!("{}\n", self.decode_body(sec_name, hdr_size + off));
                let lines = txt.lines();
                for line in lines {
                    writeln!(f, "  {}", line)?;
                }
            }
            writeln!(f, "\n")?;
        }
        Ok(())
    }
}

/// The same as core::SirLoc, just with a String representation of the symbol name and with the
/// traits we were disallowed from using in libcore.
#[derive(Debug, Hash, Eq, PartialEq)]
pub struct SirLoc {
    pub symbol_name: String,
    pub bb_idx: u32,
    // Virtual address of this location.
    pub addr: Option<u64>
}

impl SirLoc {
    pub fn new(symbol_name: String, bb_idx: u32, addr: Option<u64>) -> Self {
        Self {
            symbol_name,
            bb_idx,
            addr
        }
    }
}

/// Generic representation of a trace of SIR block locations.
pub trait SirTrace: Debug + Send {
    /// Returns the length of the *raw* (untrimmed) trace, measured in SIR locations.
    fn raw_len(&self) -> usize;

    /// Returns the SIR location at index `idx` in the *raw* (untrimmed) trace.
    fn raw_loc(&self, idx: usize) -> &SirLoc;
}

impl<'a> IntoIterator for &'a dyn SirTrace {
    type Item = &'a SirLoc;
    type IntoIter = SirTraceIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SirTraceIterator::new(self)
    }
}

/// Returns a string containing the textual representation of a SIR trace.
pub fn sir_trace_str(sir: &Sir, trace: &dyn SirTrace, trimmed: bool, show_blocks: bool) -> String {
    let locs: Vec<&SirLoc> = match trimmed {
        false => (0..(trace.raw_len())).map(|i| trace.raw_loc(i)).collect(),
        true => trace.into_iter().collect()
    };

    let mut res = String::new();
    let res_r = &mut res;

    for loc in locs {
        write!(res_r, "[{}] bb={}, flags=[", loc.symbol_name, loc.bb_idx).unwrap();

        let body = sir.body(&loc.symbol_name);
        if let Some(ref body) = body {
            if body.flags.contains(BodyFlags::INTERP_STEP) {
                write!(res_r, "INTERP_STEP ").unwrap();
            }
        }
        writeln!(res_r, "]").unwrap();

        if show_blocks {
            if let Some(body) = body {
                writeln!(
                    res_r,
                    "{}:",
                    body.blocks[usize::try_from(loc.bb_idx).unwrap()]
                )
                .unwrap();
            } else {
                writeln!(res_r, "    <no sir>").unwrap();
            }
        }
    }
    res
}

impl Display for dyn SirTrace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", sir_trace_str(&*SIR, self, false, true))
    }
}

/// An iterator over a trimmed SIR trace.
pub struct SirTraceIterator<'a> {
    trace: &'a dyn SirTrace,
    next_idx: usize
}

impl<'a> SirTraceIterator<'a> {
    pub fn new(trace: &'a dyn SirTrace) -> Self {
        SirTraceIterator { trace, next_idx: 0 }
    }
}

impl<'a> Iterator for SirTraceIterator<'a> {
    type Item = &'a SirLoc;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx < self.trace.raw_len() {
            let ret = self.trace.raw_loc(self.next_idx);
            self.next_idx += 1;
            Some(ret)
        } else {
            None // No more locations.
        }
    }
}
