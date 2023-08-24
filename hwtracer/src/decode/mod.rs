//! Trace decoders.

use crate::{errors::HWTracerError, Block, Trace};

#[cfg(decoder_ykpt)]
mod ykpt;
#[cfg(decoder_ykpt)]
use ykpt::YkPTTraceDecoder;

pub trait TraceDecoder {
    /// Create the trace decoder.
    fn new(trace: Box<dyn Trace>) -> Self
    where
        Self: Sized;

    /// Iterate over the blocks of the trace.
    fn iter_blocks<'a>(&'a self) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'a>;
}

/// Returns the default trace decoder for this configuration.
pub fn default_decoder(trace: Box<dyn Trace>) -> Result<Box<dyn TraceDecoder>, HWTracerError> {
    #[cfg(decoder_ykpt)]
    return Ok(Box::new(YkPTTraceDecoder::new(trace)));
    #[cfg(not(decoder_ykpt))]
    return Err(HWTracerError::DecoderUnavailable(self.kind));
}

/// Decoder agnostic tests  and helper routines live here.
#[cfg(test)]
mod test_helpers {
    use super::default_decoder;
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

        let ct1 = default_decoder(trace1).unwrap().iter_blocks().count();
        let ct2 = default_decoder(trace2).unwrap().iter_blocks().count();

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        assert!(ct2 > ct1 * 8);
    }
}
