//! The Yk PT trace decoder.

use crate::{decode::TraceDecoder, errors::HWTracerError, Block, Trace};

mod packet_parser;
use packet_parser::PacketParser;

pub(crate) struct YkPTTraceDecoder {}

impl TraceDecoder for YkPTTraceDecoder {
    fn new() -> Self {
        Self {}
    }

    fn iter_blocks<'t>(
        &'t self,
        trace: &'t dyn Trace,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + '_> {
        let itr = YkPTBlockIterator {
            errored: false,
            parser: PacketParser::new(trace.bytes()),
        };
        Box::new(itr)
    }
}

/// Iterate over the blocks of an Intel PT trace using the fast Yk PT decoder.
struct YkPTBlockIterator<'t> {
    /// Set to true when an error has occured.
    errored: bool,
    /// PT packet iterator.
    parser: PacketParser<'t>,
}

impl<'t> Iterator for YkPTBlockIterator<'t> {
    type Item = Result<Block, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.errored {
            // FIXME: For now this is dummy code to prevent dead-code warnings. Later block binding
            // logic will go here.
            self.parser.next().unwrap().unwrap();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collect::TraceCollectorBuilder,
        decode::{test_helpers, TraceDecoderKind},
    };

    #[ignore] // FIXME
    #[test]
    fn ten_times_as_many_blocks() {
        let tc = TraceCollectorBuilder::new().build().unwrap();
        test_helpers::ten_times_as_many_blocks(tc, TraceDecoderKind::YkPT);
    }
}
