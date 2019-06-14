// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tiri -- TIR interpreter.
//!
//! No effort has been made to make this fast.

use elf;
use fallible_iterator::FallibleIterator;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use ykpack::{BasicBlockIndex, Body, Decoder, DefId, Pack, Statement, StatementIndex, Terminator};

/// The interpreter maintains a stack of these to keep track of calls.
struct Frame<'p> {
    /// A reference to the TIR body corresponding with this `Frame`.
    body: &'p Body,
    /// The current execution position in the above TIR body.
    pc: (BasicBlockIndex, StatementIndex),
}

impl<'p> Frame<'p> {
    /// Create a new frame and position its program counter at the beginning of the first block.
    fn new(body: &'p Body) -> Self {
        Self { body, pc: (0, 0) }
    }
}

/// The interpreter itself.
/// The struct itself holds only immutable program information.
struct Interp {
    /// Maps a `DefId` to the corresponding TIR body.
    tir_map: HashMap<DefId, Body>,
    /// The `DefId` of the `main()` function, which serves as the entry point.
    main_defid: DefId,
}

impl Interp {
    /// Create a new interpreter, using the TIR found in the `.yk_tir` section of the binary `bin`.
    fn new(bin: &Path) -> Self {
        let ef = elf::File::open_path(&PathBuf::from(bin)).unwrap();
        let sec = ef.get_section(".yk_tir").unwrap();
        let mut curs = Cursor::new(&sec.data);
        let mut dec = Decoder::from(&mut curs);

        let mut tir_map = HashMap::new();
        let mut main_defid = None;
        while let Some(pack) = dec.next().unwrap() {
            let Pack::Body(body) = pack;

            if body.def_path_str == "main" {
                main_defid = Some(body.def_id.clone());
            }
            tir_map.insert(body.def_id.clone(), body);
        }

        Self {
            tir_map: tir_map,
            main_defid: main_defid.expect("Couldn't find main()"),
        }
    }

    /// Start interpreting TIR from the `main()` function.
    fn run(&self) {
        let mut stack = Vec::new();
        stack.push(Frame::new(&self.tir_map[&self.main_defid]));

        // The main interpreter loop.
        loop {
            let cur_frame = stack.last().unwrap();
            let cur_block = &cur_frame.body.blocks[usize::try_from(cur_frame.pc.0).unwrap()];
            let block_len = cur_block.stmts.len();
            let pc_stmt_usize = usize::try_from(cur_frame.pc.1).unwrap();

            if pc_stmt_usize < block_len {
                let stmt = &cur_block.stmts[pc_stmt_usize];
                self.interp_stmt(stmt);
            } else if pc_stmt_usize == block_len {
                // We take statement index one past the end to mean the block terminator.
                let term = &cur_block.term;
                self.interp_term(term);
            } else {
                unreachable!();
            }
        }
    }

    /// Interpret the specified statement.
    fn interp_stmt(&self, _stmt: &Statement) {
        unimplemented!();
    }

    /// Interpret the specified terminator.
    fn interp_term(&self, _term: &Terminator) {
        unimplemented!();
    }
}

fn main() {
    let bin = std::env::args().skip(1).next().unwrap();
    Interp::new(&PathBuf::from(bin)).run();
}
