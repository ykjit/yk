// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Specialising TIR for a MIR trace.
//!
//! This module takes a trace of MIR locations and converts it into a specialised "TIR Fragment"
//! using the TIR found in the `.yk_tir` section of the currently running executable.

use super::{MirTrace};
use fallible_iterator::FallibleIterator;
use ykpack::{Body, DefId, Decoder, Pack};
use elf;
use std::collections::HashMap;
use std::env;
use std::io::Cursor;

/// A TIR fragment is a chunk of TIR specialised to a particular trace of MIR locations. Each
/// fragment contains only the TIR blocks touched by the MIR trace. Branches not taken manifest as
/// guards in the fragment.
/// FIXME the exact representation needs to be decided.
pub struct TirFrag {}

// The TirMap lets us look up a TIR body from the MIR DefId.
// The map is immutable and unique to the executable binary being traced.
lazy_static! {
    static ref TIR_MAP: HashMap<DefId, Body> = {
        let ef = elf::File::open_path(env::current_exe().unwrap()).unwrap();
        let sec = ef.get_section(".yk_tir").expect("Can't find TIR section");
        let mut curs = Cursor::new(&sec.data);
        let mut dec = Decoder::from(&mut curs);

        let mut tir_map = HashMap::new();
        while let Some(pack) = dec.next().unwrap() {
            let Pack::Body(body) = pack;
            tir_map.insert(body.def_id.clone(), body);
        }
        tir_map
    };
}

/// The TIR Specialiser takes a trace of MIR locations and returns a TIR fragment.
pub struct TirSpecialiser<'t> {
    _trace: &'t dyn MirTrace,
}

impl<'t> TirSpecialiser<'t> {
    pub (crate) fn new(trace: &'t dyn MirTrace) -> Self {
        Self { _trace: trace }
    }

    pub (crate) fn specialise(&self) -> TirFrag {
        unimplemented!() // FIXME: Use the TIR_MAP to make a TirFrag.
    }
}
