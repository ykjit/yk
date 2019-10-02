// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Debugging utilities.

use crate::{tir::SIR, SirLoc, SirTrace};
use elf;
use fallible_iterator::FallibleIterator;
use std::{collections::HashMap, env, io::Cursor, iter::IntoIterator};
pub use ykpack::{bodyflags, Statement};
use ykpack::{Decoder, DefId, Pack};

// The SIR Debug Map lets us map a DefId to a definition path.
// This is read from the .yk_sir ELF section.
lazy_static! {
    static ref SIR_DEBUG_MAP: HashMap<DefId, String> = {
        let ef = elf::File::open_path(env::current_exe().unwrap()).unwrap();
        let sec = ef.get_section(".yk_sir").expect("Can't find SIR section");
        let mut curs = Cursor::new(&sec.data);
        let mut dec = Decoder::from(&mut curs);

        let mut sir_dbg_map = HashMap::new();
        while let Some(pack) = dec.next().unwrap() {
            match pack {
                Pack::Body(_) => (),
                Pack::Debug(d) => {
                    let old = sir_dbg_map.insert(d.def_id().clone(), String::from(d.def_path()));
                    debug_assert!(old.is_none()); // should be no duplicates.
                },
            }
        }
        sir_dbg_map
    };
}

/// Given a DefId, get the definition path.
pub fn def_path(def_id: &DefId) -> Option<&str> {
    SIR_DEBUG_MAP
        .get(def_id)
        .as_ref()
        .map(|s| String::as_str(s))
}

/// Prints a SIR trace to stdout for debugging purposes.
pub fn print_sir_trace(trace: &dyn SirTrace, trimmed: bool) {
    let locs: Vec<&SirLoc> = match trimmed {
        false => (0..(trace.raw_len())).map(|i| trace.raw_loc(i)).collect(),
        true => trace.into_iter().collect()
    };

    println!("---[ BEGIN SIR TRACE DUMP ]---");
    for loc in locs {
        let def_id = DefId::from_sir_loc(&loc);
        let def_path_s = match def_path(&def_id) {
            Some(s) => s,
            None => "<unknown>"
        };

        print!(
            "[{}] crate={}, index={}, bb={}, flags=[",
            def_path_s,
            loc.crate_hash(),
            loc.def_idx(),
            loc.bb_idx(),
        );

        let body = SIR.bodies.get(&def_id);
        if let Some(body) = body {
            if body.flags & bodyflags::TRACE_HEAD != 0 {
                print!("HEAD ");
            }
            if body.flags & bodyflags::TRACE_TAIL != 0 {
                print!("TAIL ");
            }
        }
        println!("]");
    }
    println!("---[ END SIR TRACE DUMP ]---");
}

#[cfg(test)]
mod tests {
    use super::SIR_DEBUG_MAP;

    /// This just checks loading the SIR debug map doesn't crash.
    #[test]
    fn test_load_debug_map() {
        let _ = SIR_DEBUG_MAP.iter().len();
    }
}
