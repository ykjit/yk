//! Trace decoders.

use crate::{errors::HWTracerError, Block, Trace};
#[cfg(feature = "yk_testing")]
use std::env;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[cfg(decoder_ykpt)]
mod ykpt;
#[cfg(decoder_ykpt)]
use ykpt::YkPTTraceDecoder;

#[derive(Clone, Copy, Debug, EnumIter)]
#[repr(u8)]
pub enum TraceDecoderKind {
    YkPT,
}

impl TraceDecoderKind {
    /// Returns the default kind of decoder for the current platform or `None` if this platform
    /// does not support tracing.
    pub fn default_for_platform() -> Option<Self> {
        Self::iter().find(|&kind| Self::match_platform(&kind).is_ok())
    }

    /// Returns `Ok` if the this decoder kind is appropriate for the current platform.
    fn match_platform(&self) -> Result<(), HWTracerError> {
        match self {
            Self::YkPT => {
                #[cfg(decoder_ykpt)]
                return Ok(());
                #[cfg(not(decoder_ykpt))]
                return Err(HWTracerError::DecoderUnavailable(Self::YkPT));
            }
        }
    }

    #[cfg(feature = "yk_testing")]
    fn from_str(name: &str) -> Self {
        match name {
            "ykpt" => Self::YkPT,
            _ => panic!(),
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
    pub fn build(mut self) -> Result<Box<dyn TraceDecoder>, HWTracerError> {
        #[cfg(feature = "yk_testing")]
        {
            if let Ok(val) = env::var("YKD_FORCE_TRACE_DECODER") {
                self.kind = TraceDecoderKind::from_str(&val);
            }
        }
        self.kind.match_platform()?;
        match self.kind {
            TraceDecoderKind::YkPT => {
                #[cfg(decoder_ykpt)]
                return Ok(Box::new(YkPTTraceDecoder::new()));
                #[cfg(not(decoder_ykpt))]
                return Err(HWTracerError::DecoderUnavailable(self.kind));
            }
        }
    }
}

/// Decoder agnostic tests  and helper routines live here.
#[cfg(test)]
mod test_helpers {
    use super::{TraceDecoder, TraceDecoderBuilder, TraceDecoderKind};
    use crate::{
        collect::{test_helpers::trace_closure, TraceCollector},
        test_helpers::work_loop,
    };

    /// Trace two loops, one 10x larger than the other, then check the proportions match the number
    /// of block the trace passes through.
    pub fn ten_times_as_many_blocks(tc: TraceCollector, decoder_kind: TraceDecoderKind) {
        let trace1 = trace_closure(&tc, || work_loop(10));
        let trace2 = trace_closure(&tc, || work_loop(100));

        let dec: Box<dyn TraceDecoder> = TraceDecoderBuilder::new()
            .kind(decoder_kind)
            .build()
            .unwrap();

        let ct1 = dec.iter_blocks(&*trace1).count();
        let ct2 = dec.iter_blocks(&*trace2).count();

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        assert!(ct2 > ct1 * 8);
    }
}
