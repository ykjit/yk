use crate::trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction};

pub(crate) struct SWTraceIterator {
    trace: std::vec::IntoIter<TraceAction>,
}

impl SWTraceIterator {
    pub(crate) fn new(trace: Vec<TraceAction>) -> SWTraceIterator {
        return SWTraceIterator {
            trace: trace.into_iter(),
        };
    }
}

impl Iterator for SWTraceIterator {
    type Item = Result<TraceAction, AOTTraceIteratorError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.trace.next().map(|x| Ok(x))
    }
}

impl AOTTraceIterator for SWTraceIterator {}
