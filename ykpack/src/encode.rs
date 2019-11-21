use crate::Pack;
use rmp_serde::{encode, Serializer};
use serde::Serialize;
use std::io::prelude::*;

/// The pack encoder.
///
/// Packs are written using the `serialise()` method. Once all of the desired packs is serialised,
/// the consumer must call `done()`.
pub struct Encoder<'a> {
    ser: Serializer<&'a mut dyn Write>,
    done: bool,
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder which serialises `Pack` into the writable `write_into`.
    pub fn from(write_into: &'a mut dyn Write) -> Self {
        let ser = Serializer::new(write_into);
        Self { ser, done: false }
    }

    /// Serialises a pack.
    pub fn serialise(&mut self, md: Pack) -> Result<(), encode::Error> {
        Some(md).serialize(&mut self.ser)
    }

    /// Finalises the serialisation and writes a sentinel.
    pub fn done(mut self) -> Result<(), encode::Error> {
        None::<Option<Pack>>.serialize(&mut self.ser)?;
        self.done = true;
        Ok(())
    }
}
