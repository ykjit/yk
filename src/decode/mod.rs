//! Trace decoders.

use crate::{errors::HWTracerError, Block, Trace};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[cfg(decoder_libipt)]
pub(crate) mod libipt;
#[cfg(decoder_libipt)]
use libipt::LibIPTTraceDecoder;

#[derive(Clone, Copy, Debug, EnumIter)]
pub enum TraceDecoderKind {
    LibIPT,
}

impl TraceDecoderKind {
    /// Returns the default kind of decoder for the current platform.
    fn default_for_platform() -> Option<Self> {
        for kind in Self::iter() {
            if Self::match_platform(&kind).is_ok() {
                return Some(kind);
            }
        }
        None
    }

    /// Returns `Ok` if the this decoder kind is appropriate for the current platform.
    fn match_platform(&self) -> Result<(), HWTracerError> {
        match self {
            Self::LibIPT => {
                #[cfg(not(decoder_libipt))]
                return Err(HWTracerError::DecoderUnavailable(Self::LibIPT));
                #[cfg(decoder_libipt)]
                return Ok(());
            }
        }
    }
}

pub trait TraceDecoder {
    /// Create the trace decoder.
    fn new() -> Self
    where
        Self: Sized;

    /// Iterate over the blocks of the trace.
    fn iter_blocks<'t>(
        &'t self,
        trace: &'t dyn Trace,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + '_>;
}

pub struct TraceDecoderBuilder {
    kind: TraceDecoderKind,
}

impl TraceDecoderBuilder {
    /// Create a new TraceDecoderBuilder using an appropriate defaults.
    pub fn new() -> Self {
        Self {
            kind: TraceDecoderKind::default_for_platform().unwrap(),
        }
    }

    /// Select the kind of trace decoder.
    pub fn kind(mut self, kind: TraceDecoderKind) -> Self {
        self.kind = kind;
        self
    }

    /// Build the trace decoder.
    ///
    /// An error is returned if the requested decoder is inappropriate for the platform or the
    /// requested decoder was not compiled in to hwtracer.
    pub fn build(self) -> Result<Box<dyn TraceDecoder>, HWTracerError> {
        self.kind.match_platform()?;
        match self.kind {
            TraceDecoderKind::LibIPT => Ok(Box::new(LibIPTTraceDecoder::new())),
        }
    }
}

/// Decoder agnostic tests  and helper routines live here.
#[cfg(test)]
mod test_helpers {
    use super::{TraceDecoder, TraceDecoderBuilder, TraceDecoderKind};
    use crate::{
        collect::{test_helpers::trace_closure, ThreadTraceCollector},
        test_helpers::work_loop,
        Block, Trace,
    };
    use std::slice::Iter;

    /// Helper to check an expected list of blocks matches what we actually got.
    pub fn test_expected_blocks(
        trace: Box<dyn Trace>,
        decoder_kind: TraceDecoderKind,
        mut expect_iter: Iter<Block>,
    ) {
        let dec = TraceDecoderBuilder::new()
            .kind(decoder_kind)
            .build()
            .unwrap();
        let mut got_iter = dec.iter_blocks(&*trace);
        loop {
            let expect = expect_iter.next();
            let got = got_iter.next();
            if expect.is_none() || got.is_none() {
                break;
            }
            assert_eq!(
                got.unwrap().unwrap().first_instr(),
                expect.unwrap().first_instr()
            );
        }
        // Check that both iterators were the same length.
        assert!(expect_iter.next().is_none());
        assert!(got_iter.next().is_none());
    }

    /// Trace two loops, one 10x larger than the other, then check the proportions match the number
    /// of block the trace passes through.
    pub fn ten_times_as_many_blocks(
        thr_col: &mut dyn ThreadTraceCollector,
        decoder_kind: TraceDecoderKind,
    ) {
        let trace1 = trace_closure(thr_col, || work_loop(10));
        let trace2 = trace_closure(thr_col, || work_loop(100));

        let dec: Box<dyn TraceDecoder> = TraceDecoderBuilder::new()
            .kind(decoder_kind)
            .build()
            .unwrap();

        let ct1 = dec.iter_blocks(&*trace1).count();
        let ct2 = dec.iter_blocks(&*trace2).count();

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        assert!(ct2 > ct1 * 9);
    }
}
