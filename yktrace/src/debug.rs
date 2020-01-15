//! Debugging utilities.

use crate::{tir::SIR, SirLoc, SirTrace};
use std::{convert::TryFrom, iter::IntoIterator};
pub use ykpack::{bodyflags, Statement};

/// Prints a SIR trace to stdout for debugging purposes.
pub fn print_sir_trace(trace: &dyn SirTrace, trimmed: bool, show_blocks: bool) {
    let locs: Vec<&SirLoc> = match trimmed {
        false => (0..(trace.raw_len())).map(|i| trace.raw_loc(i)).collect(),
        true => trace.into_iter().collect()
    };

    println!("---[ BEGIN SIR TRACE DUMP ]---");
    for loc in locs {
        print!("[{}] bb={}, flags=[", loc.symbol_name, loc.bb_idx,);

        let body = SIR.bodies.get(&loc.symbol_name);
        if let Some(body) = body {
            if body.flags & bodyflags::TRACE_HEAD != 0 {
                print!("HEAD ");
            }
            if body.flags & bodyflags::TRACE_TAIL != 0 {
                print!("TAIL ");
            }
        }
        println!("]");

        if show_blocks {
            if let Some(body) = body {
                println!("{}:", body.blocks[usize::try_from(loc.bb_idx).unwrap()]);
            } else {
                println!("    <no sir>");
            }
        }
    }
    println!("---[ END SIR TRACE DUMP ]---");
}
