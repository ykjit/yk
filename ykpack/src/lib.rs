// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ykpack -- Serialiser and deserialiser for carrying data from compile-time to run-time.
//!
//! This crate allows ykrustc to serialise various compile-time information for later
//! deserialisation by the Yorick runtime.
//!
//! The encoder and decoder API is structured in such a way that each item -- or "Pack" -- can be
//! streamed to/from the serialised format one item at a time. This helps to reduce memory
//! consumption.
//!
//! The MIR data is serialised in the msgpack format in the following form:
//!
//!  -----------
//!  pack_0:             \
//!  ...                  - Packs.
//!  pack_n              /
//!  sentinel           -- End of packs marker.
//!  -----------
//!
//!  Where each pack_i is an instance of `Some(Pack)` and the sentinel is a `None`.
//!
//!  The version field is automatically written and checked by the `Encoder` and `Decoder`
//!  respectively.

mod decode;
mod encode;
mod types;

pub use decode::Decoder;
pub use encode::Encoder;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::{
        BasicBlock, Constant, Decoder, DefId, Encoder, Local, Operand, Pack, Rvalue, Statement,
        Terminator, Tir, UnsignedInt,
    };
    use fallible_iterator::{self, FallibleIterator};
    use std::io::{Cursor, Seek, SeekFrom};

    // Get a cursor to serialise to and deserialise from. For real, we'd be reading from a file,
    // but for tests we use a vector of bytes.
    fn get_curs() -> Cursor<Vec<u8>> {
        let buf: Vec<u8> = Vec::new();
        Cursor::new(buf)
    }

    // Rewind a cursor to the beginning.
    fn rewind_curs(curs: &mut Cursor<Vec<u8>>) {
        curs.seek(SeekFrom::Start(0)).unwrap();
    }

    // Makes some sample stuff to round trip test.
    fn get_sample_packs() -> Vec<Pack> {
        let dummy_term = Terminator::Abort;

        let stmts1_b1 = vec![Statement::Nop; 16];
        let stmts1_b2 = vec![Statement::Nop; 3];
        let blocks1 = vec![
            BasicBlock::new(stmts1_b1, dummy_term.clone()),
            BasicBlock::new(stmts1_b2, dummy_term.clone()),
        ];
        let tir1 = Pack::Tir(Tir::new(DefId::new(1, 2), String::from("item1"), blocks1));

        let stmts2_b1 = vec![Statement::Nop; 7];
        let stmts2_b2 = vec![Statement::Nop; 200];
        let stmts2_b3 = vec![Statement::Nop; 1];
        let blocks2 = vec![
            BasicBlock::new(stmts2_b1, dummy_term.clone()),
            BasicBlock::new(stmts2_b2, dummy_term.clone()),
            BasicBlock::new(stmts2_b3, dummy_term.clone()),
        ];
        let tir2 = Pack::Tir(Tir::new(DefId::new(4, 5), String::from("item2"), blocks2));

        vec![tir1, tir2]
    }

    // Check serialising and deserialising works for zero packs.
    #[test]
    fn test_empty() {
        let mut curs = get_curs();

        let enc = Encoder::from(&mut curs);
        enc.done().unwrap();

        rewind_curs(&mut curs);
        let mut dec = Decoder::from(&mut curs);
        assert!(dec.next().unwrap().is_none());
    }

    // Check a typical serialising and deserialising session.
    #[test]
    fn test_basic() {
        let inputs = get_sample_packs();
        let mut curs = get_curs();

        let mut enc = Encoder::from(&mut curs);
        for md in &inputs {
            enc.serialise(md.clone()).unwrap();
        }
        enc.done().unwrap();

        rewind_curs(&mut curs);
        let dec = Decoder::from(&mut curs);

        // Obtain two fallible iterators, so we can zip them.
        let expect_iter = fallible_iterator::convert(inputs.into_iter().map(|e| Ok(e)));

        let mut itr = dec.zip(expect_iter);
        while let Some((got, expect)) = itr.next().unwrap() {
            assert_eq!(expect, got);
        }
    }

    #[test]
    #[should_panic(expected = "not marked done")]
    fn test_encode_not_done() {
        let inputs = get_sample_packs();
        let mut curs = get_curs();

        let mut enc = Encoder::from(&mut curs);
        for md in &inputs {
            enc.serialise(md.clone()).unwrap();
        }
        // We expect this to panic, as the encoder wasn't finalised with a call to `enc.done()`.
    }

    #[test]
    fn test_text_dump() {
        let stmts_t1_b0 = vec![
            Statement::Assign(
                Local::new(0, 0),
                Rvalue::Operand(Operand::Local(Local::new(1, 0))),
            ),
            Statement::Assign(Local::new(2, 0), Rvalue::GetField(Local::new(3, 0), 4)),
            Statement::Assign(
                Local::new(4, 0),
                Rvalue::Operand(Operand::Constant(Constant::UnsignedInt(UnsignedInt::U8(
                    10,
                )))),
            ),
            Statement::Nop,
        ];
        let term_t1_b0 = Terminator::Abort;
        let stmts_t1_b1 = vec![
            Statement::Assign(Local::new(5, 0), Rvalue::Load(Local::new(6, 0))),
            Statement::Store(Local::new(5, 0), Operand::Local(Local::new(4, 0))),
            Statement::Assign(
                Local::new(7, 0),
                Rvalue::Add(
                    Operand::Local(Local::new(8, 0)),
                    Operand::Local(Local::new(9, 0)),
                ),
            ),
            Statement::Assign(
                Local::new(7, 0),
                Rvalue::Sub(
                    Operand::Local(Local::new(9, 0)),
                    Operand::Local(Local::new(10, 0)),
                ),
            ),
            Statement::Assign(Local::new(11, 0), Rvalue::Alloca(0)),
        ];
        let term_t1_b1 = Terminator::Goto { target_bb: 50 };

        let blocks_t1 = vec![
            BasicBlock::new(stmts_t1_b0, term_t1_b0),
            BasicBlock::new(stmts_t1_b1, term_t1_b1),
        ];

        let tirs = vec![
            Pack::Tir(Tir::new(DefId::new(1, 2), String::from("item1"), blocks_t1)),
            Pack::Tir(Tir::new(
                DefId::new(3, 4),
                String::from("item2"),
                vec![BasicBlock::new(
                    vec![Statement::Unimplemented],
                    Terminator::Unreachable,
                )],
            )),
        ];

        let mut got = String::new();
        for pack in tirs {
            got.push_str(&format!("{}", pack));
        }
        let got_lines = got.split("\n");

        let expect = "[Begin TIR for item1]\n\
    DefId(1, 2):
    bb0:
        $0: t0 = $1: t0
        $2: t0 = get_field($3: t0, 4)
        $4: t0 = U8(10)
        nop
        Abort
    bb1:
        $5: t0 = load($6: t0)
        store($5: t0, $4: t0)
        $7: t0 = int_add($8: t0, $9: t0)
        $7: t0 = int_sub($9: t0, $10: t0)
        $11: t0 = alloca(0)
        Goto { target_bb: 50 }
[End TIR for item1]
[Begin TIR for item2]
    DefId(3, 4):
    bb0:
        unimplemented
        Unreachable
[End TIR for item2]\n";

        let expect_lines = expect.split("\n");

        assert_eq!(got_lines.clone().count(), expect_lines.clone().count());
        for (got, expect) in got_lines.zip(expect_lines) {
            assert_eq!(got.trim(), expect.trim());
        }
    }
}
