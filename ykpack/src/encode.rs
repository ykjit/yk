use crate::Pack;
use std::io::Write;

/// The pack encoder.
///
/// Packs are written using the `serialise()` method. Once all of the desired packs is serialised,
/// the consumer must call `done()`.
pub struct Encoder<'a> {
    to: &'a mut dyn Write,
    done: bool,
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder which serialises `Pack` into the writable `write_into`.
    pub fn from(write_into: &'a mut dyn Write) -> Self {
        Self {
            to: write_into,
            done: false,
        }
    }

    /// Serialises a pack.
    pub fn serialise(&mut self, md: Pack) -> Result<(), bincode::Error> {
        assert!(!self.done);
        bincode::serialize_into(&mut *self.to, &Some(md))
    }

    /// Finalises the serialisation and writes a sentinel.
    pub fn done(mut self) -> Result<(), bincode::Error> {
        assert!(!self.done);
        bincode::serialize_into(&mut *self.to, &None::<Pack>)?;
        self.done = true;
        Ok(())
    }
}
