//! Loading and tracing of Serialised Intermediate Representation (SIR).

use elf;
use fallible_iterator::FallibleIterator;
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    env,
    fmt::{self, Debug, Display, Write},
    io::Cursor,
    iter::Iterator,
    path::Path
};
use ykpack::{bodyflags, Body, Decoder, Local, Pack, Ty}; // FIXME kill.

/// The serialised IR loaded in from disk. One of these structures is generated in the above
/// `lazy_static` and is shared immutably for all threads.
#[derive(Debug)]
pub struct Sir {
    /// Lets us map a symbol name to a SIR body.
    pub bodies: HashMap<String, Body>,
    // Interesting locations that we need quick access to.
    pub markers: SirMarkers,
    /// SIR Local variable types, keyed by crate hash.
    pub types: HashMap<u64, Vec<Ty>>,
    /// Thread tracer type IDs.
    pub thread_tracers: HashSet<ykpack::TypeId>
}

impl Sir {
    pub fn ty(&self, id: &ykpack::TypeId) -> &ykpack::Ty {
        &self.types[&id.0][usize::try_from(id.1).unwrap()]
    }

    pub fn is_thread_tracer_ty(&self, id: &ykpack::TypeId) -> bool {
        self.thread_tracers.contains(id)
    }
}

/// Records interesting locations required for trace manipulation.
#[derive(Debug)]
pub struct SirMarkers {
    /// Functions which start tracing and whose suffix gets trimmed off the top of traces.
    /// Although you'd expect only one such function, (i.e. `yktrace::start_tracing`), in fact
    /// the location which appears in the trace can vary according to how Rust compiles the
    /// program (this happens even if `yktracer::start_tracing()` is marked `#[inline(never)]`).
    /// For this reason, we mark few different places as potential heads.
    ///
    /// We will only see the suffix of these functions in traces, as trace recording will start
    /// somewhere in the middle of them.
    ///
    /// The compiler is made aware of this location by the `#[trace_head]` annotation.
    pub trace_heads: Vec<String>,
    /// Similar to `trace_heads`, functions which stop tracing and whose prefix gets trimmed off
    /// the bottom of traces.
    ///
    /// The compiler is made aware of these locations by the `#[trace_tail]` annotation.
    pub trace_tails: Vec<String>
}

lazy_static! {
    pub static ref SIR: Sir = Sir::read_file(&env::current_exe().unwrap()).unwrap();
}

impl Sir {
    pub fn read_file(file: &Path) -> Result<Sir, ()> {
        let ef = elf::File::open_path(file).unwrap();

        // We iterate over ELF sections, looking for ones which contain SIR and loading it into
        // memory.
        let mut bodies = HashMap::new();
        let mut types = HashMap::new();
        let mut trace_heads = Vec::new();
        let mut trace_tails = Vec::new();
        let mut thread_tracers = HashSet::new();
        for sec in &ef.sections {
            if sec.shdr.name.starts_with(".yksir_") {
                let mut curs = Cursor::new(&sec.data);
                let mut dec = Decoder::from(&mut curs);

                while let Some(pack) = dec.next().unwrap() {
                    match pack {
                        Pack::Body(body) => {
                            // Cache some locations that we need quick access to.
                            if body.flags & bodyflags::TRACE_HEAD != 0 {
                                trace_heads.push(body.symbol_name.clone());
                            }

                            if body.flags & bodyflags::TRACE_TAIL != 0 {
                                trace_tails.push(body.symbol_name.clone());
                            }

                            // Due to the way Rust compiles stuff, duplicates may exist. Where
                            // duplicates exist, the functions will be identical, but may have
                            // different (but equivalent) types. This is because types too may be
                            // duplicated using a different crate hash.
                            bodies
                                .entry(body.symbol_name.clone())
                                .or_insert_with(|| body);
                        }
                        Pack::Types(ts) => {
                            let old = types.insert(ts.crate_hash, ts.types);
                            debug_assert!(old.is_none()); // There's one `Types` pack per crate.
                            for idx in ts.thread_tracers {
                                thread_tracers.insert((ts.crate_hash, idx));
                            }
                        }
                    }
                }
            }
        }

        assert!(!trace_heads.is_empty(), "no trace heads found!");
        assert!(!trace_tails.is_empty(), "no trace tails found!");
        let markers = SirMarkers {
            trace_heads,
            trace_tails
        };

        Ok(Sir {
            bodies,
            markers,
            types,
            thread_tracers
        })
    }
}

impl Display for Sir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for body in self.bodies.values() {
            writeln!(f, "{}", body)?;
        }

        for head in &self.markers.trace_heads {
            writeln!(f, "HEAD {}", head)?;
        }

        for tail in &self.markers.trace_tails {
            writeln!(f, "TAIL {}", tail)?;
        }

        for (crate_hash, types) in self.types.iter() {
            writeln!(f, "TYPES OF {}", crate_hash)?;
            for ty in types {
                writeln!(f, "{}", ty)?;
            }
        }

        for thread_tracer in self.thread_tracers.iter() {
            writeln!(f, "THREAD TRACER {}:{}", thread_tracer.0, thread_tracer.1)?;
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

    /// Returns the local variable containing the trace inputs tuple.
    fn input(&self) -> Local;
}

impl<'a> IntoIterator for &'a dyn SirTrace {
    type Item = &'a SirLoc;
    type IntoIter = SirTraceIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SirTraceIterator::new(&*SIR, self)
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

    write!(res_r, "Trace input local: {}\n\n", trace.input()).unwrap();
    for loc in locs {
        write!(res_r, "[{}] bb={}, flags=[", loc.symbol_name, loc.bb_idx).unwrap();

        let body = sir.bodies.get(&loc.symbol_name);
        if let Some(body) = body {
            if body.flags & bodyflags::TRACE_HEAD != 0 {
                write!(res_r, "HEAD ").unwrap();
            }
            if body.flags & bodyflags::TRACE_TAIL != 0 {
                write!(res_r, "TAIL ").unwrap();
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
    sir: &'a Sir,
    trace: &'a dyn SirTrace,
    next_idx: usize
}

impl<'a> SirTraceIterator<'a> {
    pub fn new(sir: &'a Sir, trace: &'a dyn SirTrace) -> Self {
        // We are going to present a "trimmed trace", so we do a backwards scan looking for the end
        // of the code that starts the tracer.
        let mut begin_idx = None;
        for blk_idx in (0..trace.raw_len()).rev() {
            let sym = &trace.raw_loc(blk_idx).symbol_name;
            if sir.markers.trace_heads.contains(sym) {
                begin_idx = Some(blk_idx + 1);
                break;
            }
        }

        SirTraceIterator {
            sir,
            trace,
            next_idx: begin_idx.expect("Couldn't find the end of the code that starts the tracer")
        }
    }
}

impl<'a> Iterator for SirTraceIterator<'a> {
    type Item = &'a SirLoc;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx < self.trace.raw_len() {
            let sym = &self.trace.raw_loc(self.next_idx).symbol_name;
            if self.sir.markers.trace_tails.contains(sym) {
                // Stop when we find the start of the code that stops the tracer, thus trimming the
                // end of the trace. By setting the next index to one above the last one in the
                // trace, we ensure the iterator will return `None` forever more.
                self.next_idx = self.trace.raw_len();
                None
            } else {
                let ret = self.trace.raw_loc(self.next_idx);
                self.next_idx += 1;
                Some(ret)
            }
        } else {
            None // No more locations.
        }
    }
}
