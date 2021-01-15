use crate::Pack;
use fallible_iterator::FallibleIterator;
use std::io::Read;

/// The pack decoder.
/// Offers a simple iterator interface to serialised packs.
pub struct Decoder<'a> {
    from: &'a mut dyn Read,
}

impl<'a> Decoder<'a> {
    /// Returns a new decoder which will deserialise from `read_from`.
    pub fn from(read_from: &'a mut dyn Read) -> Self {
        Self { from: read_from }
    }
}

impl<'a> FallibleIterator for Decoder<'a> {
    type Item = Pack;
    type Error = bincode::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        bincode::deserialize_from(&mut *self.from)
    }
}
