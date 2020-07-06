//! Debugging utilities.

use crate::{tir::SIR, SirLoc, SirTrace};
use std::{convert::TryFrom, fmt::Write, iter::IntoIterator, string::String};
pub use ykpack::{bodyflags, Statement};

/// Prints a SIR trace to stdout for debugging purposes.
pub fn sir_trace_str<'a>(trace: &'a dyn SirTrace, trimmed: bool, show_blocks: bool) -> String {
    let locs: Vec<&SirLoc> = match trimmed {
        false => (0..(trace.raw_len())).map(|i| trace.raw_loc(i)).collect(),
        true => trace.into_iter().collect()
    };

    let mut res = String::new();
    let res_r = &mut res;

    write!(res_r, "Trace input local: {}\n\n", trace.input()).unwrap();
    for loc in locs {
        write!(res_r, "[{}] bb={}, flags=[", loc.symbol_name, loc.bb_idx).unwrap();

        let body = SIR.bodies.get(&loc.symbol_name);
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
