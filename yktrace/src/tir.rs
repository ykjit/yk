// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Conceptually this module takes an ordered collection of SIR block locations and converts it
//! into a Tracing IR (TIR) Trace using the SIR found in the `.yk_sir` section of the currently
//! running executable.

use super::SirTrace;
use elf;
use fallible_iterator::FallibleIterator;
use std::{collections::HashMap, env, io::Cursor};
use ykpack::{Body, Decoder, DefId, Pack};

// The SIR Map lets us look up a SIR body from the SIR DefId.
// The map is unique to the executable binary being traced (i.e. shared for all threads).
lazy_static! {
    static ref SIR_MAP: HashMap<DefId, Body> = {
        let ef = elf::File::open_path(env::current_exe().unwrap()).unwrap();
        let sec = ef.get_section(".yk_tir").expect("Can't find TIR section");
        let mut curs = Cursor::new(&sec.data);
        let mut dec = Decoder::from(&mut curs);

        let mut sir_map = HashMap::new();
        while let Some(pack) = dec.next().unwrap() {
            let Pack::Body(body) = pack;
            sir_map.insert(body.def_id.clone(), body);
        }
        sir_map
    };
}

/// A TIR trace is conceptually a straight-line path through the SIR with guarded speculation.
pub struct TirTrace {
    _stmts: Vec<_TirOp>
}

impl TirTrace {
    pub(crate) fn new(_trace: &'_ dyn SirTrace) -> Self {
        // FIXME: Use the SIR_MAP to convert the SirTrace to a TirTrace.
        unimplemented!()
    }
}

/// A TIR operation. A collection of these makes a TIR trace.
pub enum _TirOp {
    // FIXME: implement innards.
}
