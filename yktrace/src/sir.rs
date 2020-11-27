//! Loading and tracing of Serialised Intermediate Representation (SIR).

use fallible_iterator::FallibleIterator;
use memmap::Mmap;
use object::{Object, ObjectSection};
use std::{
    collections::HashMap,
    convert::TryFrom,
    env,
    fmt::{self, Debug, Display, Write},
    fs::File,
    io::Cursor,
    iter::Iterator,
    path::Path
};
use ykpack::{self, Body, BodyFlags, CguHash, Decoder, Pack, Ty};

/// The serialised IR loaded in from disk. One of these structures is generated in the above
/// `lazy_static` and is shared immutably for all threads.
#[derive(Debug)]
pub struct Sir {
    /// Lets us map a symbol name to a SIR body.
    pub bodies: HashMap<String, Body>,
    /// SIR Local variable types, keyed by codegen unit hash.
    pub types: HashMap<CguHash, Vec<Ty>>
}

impl Sir {
    pub fn ty(&self, id: &ykpack::TypeId) -> &ykpack::Ty {
        &self.types[&id.0][usize::try_from(id.1).unwrap()]
    }
}

lazy_static! {
    pub static ref SIR: Sir = Sir::read_file(&env::current_exe().unwrap()).unwrap();
}

impl Sir {
    pub fn read_file(file: &Path) -> Result<Sir, ()> {
        // SAFETY: Not really, we hope that nobody changes the file underneath our feet.
        let data = unsafe { Mmap::map(&File::open(file).unwrap()).unwrap() };
        let object = object::File::parse(&*data).unwrap();

        // We iterate over ELF sections, looking for ones which contain SIR and loading them into
        // memory.
        let mut bodies = HashMap::new();
        let mut types = HashMap::new();
        for sec in object.sections() {
            if sec.name().unwrap().starts_with(".yksir_") {
                let mut curs = Cursor::new(sec.data().unwrap());
                let mut dec = Decoder::from(&mut curs);

                while let Some(pack) = dec.next().unwrap() {
                    match pack {
                        Pack::Body(body) => {
                            // Due to the way Rust compiles stuff, duplicates may exist. Where
                            // duplicates exist, the functions will be identical, but may have
                            // different (but equivalent) types. This is because types too may be
                            // duplicated (for example in a different crate).
                            bodies
                                .entry(body.symbol_name.clone())
                                .or_insert_with(|| body);
                        }
                        Pack::Types(ts) => {
                            let old = types.insert(ts.cgu_hash, ts.types);
                            debug_assert!(old.is_none()); // There's one `Types` pack per codegen unit.
                        }
                    }
                }
            }
        }

        Ok(Sir { bodies, types })
    }
}

impl Display for Sir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for body in self.bodies.values() {
            writeln!(f, "{}", body)?;
        }

        for (cgu_hash, types) in self.types.iter() {
            writeln!(f, "TYPES OF {}", cgu_hash)?;
            for ty in types {
                writeln!(f, "{}", ty)?;
            }
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
pub trait SirTrace: Debug {
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

        let body = sir.bodies.get(&loc.symbol_name);
        if let Some(body) = body {
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
