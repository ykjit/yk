//! Loading and tracing of Serialised Intermediate Representation (SIR).

use fallible_iterator::FallibleIterator;
use fxhash::FxHashMap;
use memmap2::Mmap;
use object::{Object, ObjectSection};
use std::{
    convert::TryFrom,
    env,
    fmt::{self, Debug, Display, Write},
    fs::File,
    io::{Cursor, Seek, SeekFrom},
    iter::Iterator,
    sync::{Arc, RwLock}
};
use ykpack::{self, Body, BodyFlags, CguHash, Decoder, Local, Pack, SirHeader, SirOffset, Ty};

// The return local is always $0.
pub const RETURN_LOCAL: Local = Local(0);
// In TIR traces, the argument to the interp_step is always local $1.
pub const INTERP_STEP_ARG: Local = Local(1);

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
    sec_cache: FxHashMap<String, &'m [u8]>,
    /// Body cache, to avoid repeated decodings.
    body_cache: RwLock<FxHashMap<String, Option<Arc<Body>>>>,
    /// Type cache, to avoid repeated decodings.
    ty_cache: RwLock<FxHashMap<ykpack::TypeId, Arc<Ty>>>
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
            sec_cache,
            body_cache: Default::default(),
            ty_cache: Default::default()
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
    pub fn ty(&self, tyid: &ykpack::TypeId) -> Arc<ykpack::Ty> {
        {
            let rd = self.ty_cache.read().unwrap();
            if let Some(ty) = rd.get(tyid) {
                // Cache hit, return a reference to the previously decoded body.
                return ty.clone();
            }
        } // Drop the RwLock's read() to prevent deadlocking.

        // Cache miss. Decode the type and update the cache.
        let (ref sec_name, ref hdr, hdr_size) = SIR.hdrs[&tyid.cgu];
        let off = hdr.types[usize::try_from(tyid.idx.0).unwrap()];
        let ty = self.decode_ty(sec_name, hdr_size + off);
        let mut wr = self.ty_cache.write().unwrap();
        let arc = Arc::new(ty);
        wr.insert(tyid.to_owned(), arc.clone());
        arc
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
    pub fn body(&self, body_sym: &str) -> Option<Arc<Body>> {
        {
            let rd = self.body_cache.read().unwrap();
            if let Some(body) = rd.get(body_sym) {
                // Cache hit, return a reference to the previously decoded body.
                return body.clone();
            }
        } // Drop the RwLock's read() to prevent deadlocking.

        // Cache miss. Decode the body and update the cache.
        for (sec_name, hdr, hdr_size) in SIR.hdrs.values() {
            if let Some(off) = hdr.bodies.get(body_sym) {
                let body = self.decode_body(sec_name, hdr_size + off);
                let mut wr = self.body_cache.write().unwrap();
                let arc = Arc::new(body);
                wr.insert(body_sym.to_owned(), Some(arc.clone()));
                return Some(arc);
            }
        }

        // The body is absent. Update the cache with a `None` to prevent repeated searches.
        self.body_cache
            .write()
            .unwrap()
            .insert(body_sym.to_owned(), None);
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
    pub symbol_name: &'static str,
    pub bb_idx: u32,
    // Virtual address of this location.
    pub addr: Option<u64>
}

impl SirLoc {
    pub fn new(symbol_name: &'static str, bb_idx: u32, addr: Option<u64>) -> Self {
        Self {
            symbol_name,
            bb_idx,
            addr
        }
    }
}

/// Generic representation of a trace of SIR block locations.
pub struct SirTrace(Vec<SirLoc>);

impl SirTrace {
    pub fn new(locs: Vec<SirLoc>) -> Self {
        SirTrace(locs)
    }
}

impl std::ops::Deref for SirTrace {
    type Target = [SirLoc];

    fn deref(&self) -> &[SirLoc] {
        &*self.0
    }
}

/// Returns a string containing the textual representation of a SIR trace.
pub fn sir_trace_str(sir: &Sir, trace: &SirTrace, show_blocks: bool) -> String {
    let mut res = String::new();
    let res_r = &mut res;

    for loc in &**trace {
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

impl Display for SirTrace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", sir_trace_str(&*SIR, self, true))
    }
}
