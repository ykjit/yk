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

/// The prefix used in `DILabel` names for blocks.
pub const BLOCK_LABEL_PREFIX: &str = "__YK_BLK";

// ELF sections with this prefix contain SIR.
pub const SIR_SECTION_PREFIX: &str = ".yksir_";

// The ELF section with this name contains label mapping information derived from DWARF DILabels.
pub const YKLABELS_SECTION: &str = ".yklabels";

#[cfg(test)]
mod tests {
    use super::{BasicBlock, Body, BodyFlags, Decoder, Encoder, Pack, Statement, Terminator};
    use fallible_iterator::{self, FallibleIterator};
    use std::io::Cursor;

    // Makes some sample stuff to round trip test.
    fn get_sample_packs() -> Vec<Pack> {
        let dummy_term = Terminator::Goto(10);

        let stmts1_b1 = vec![Statement::Nop; 16];
        let stmts1_b2 = vec![Statement::Nop; 3];
        let blocks1 = vec![
            BasicBlock::new(stmts1_b1, dummy_term.clone()),
            BasicBlock::new(stmts1_b2, dummy_term.clone()),
        ];
        let sir1 = Pack::Body(Body {
            symbol_name: String::from("symbol1"),
            blocks: blocks1,
            flags: BodyFlags::empty(),
            local_decls: Vec::new(),
            num_args: 0,
            layout: (0, 0),
            offsets: Vec::new(),
        });

        let stmts2_b1 = vec![Statement::Nop; 7];
        let stmts2_b2 = vec![Statement::Nop; 200];
        let stmts2_b3 = vec![Statement::Nop; 1];
        let blocks2 = vec![
            BasicBlock::new(stmts2_b1, dummy_term.clone()),
            BasicBlock::new(stmts2_b2, dummy_term.clone()),
            BasicBlock::new(stmts2_b3, dummy_term.clone()),
        ];
        let sir2 = Pack::Body(Body {
            symbol_name: String::from("symbol2"),
            blocks: blocks2,
            flags: BodyFlags::empty(),
            local_decls: Vec::new(),
            num_args: 0,
            layout: (0, 0),
            offsets: Vec::new(),
        });

        vec![sir1, sir2]
    }

    // Check a typical serialising and deserialising session.
    #[test]
    fn basic() {
        let inputs = get_sample_packs();
        let num_packs = inputs.len();
        let mut buf = Vec::new();
        let mut enc = Encoder::from(&mut buf);
        for md in &inputs {
            enc.serialise(md.clone()).unwrap();
        }

        let mut curs = Cursor::new(&mut buf);
        let dec = Decoder::from(&mut curs);

        // Obtain two fallible iterators, so we can zip them.
        let expect_iter = fallible_iterator::convert(inputs.into_iter().map(|e| Ok(e)));

        let mut itr = dec.zip(expect_iter);
        for _ in 0..num_packs {
            let (got, expect) = itr.next().unwrap().unwrap();
            assert_eq!(expect, got);
        }

        // We've consumed everything, so attempting to decode another pack should fail.
        assert!(itr.next().is_err());
    }
}
