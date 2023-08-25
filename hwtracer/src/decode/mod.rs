//! Trace decoders.

#[cfg(decoder_ykpt)]
pub(crate) mod ykpt;

/// Decoder agnostic tests  and helper routines live here.
#[cfg(test)]
mod test_helpers {
    use crate::{
        collect::{test_helpers::trace_closure, Tracer},
        work_loop,
    };
    use std::sync::Arc;

    /// Trace two loops, one 10x larger than the other, then check the proportions match the number
    /// of block the trace passes through.
    pub fn ten_times_as_many_blocks(tc: Arc<dyn Tracer>) {
        let trace1 = trace_closure(&tc, || work_loop(10));
        let trace2 = trace_closure(&tc, || work_loop(100));

        let ct1 = trace1.iter_blocks().count();
        let ct2 = trace2.iter_blocks().count();

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        assert!(ct2 > ct1 * 8);
    }
}
