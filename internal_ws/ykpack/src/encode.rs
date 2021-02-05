//! XXX needs a top-level doc comment

use crate::Pack;

/// The pack encoder.
pub struct Encoder<'a> {
    to: &'a mut Vec<u8>,
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder which serialises `Pack` into the writable `write_into`.
    pub fn from(write_into: &'a mut Vec<u8>) -> Self {
        Self { to: write_into }
    }

    /// Serialises a pack.
    pub fn serialise(&mut self, md: Pack) -> Result<(), bincode::Error> {
        bincode::serialize_into(&mut *self.to, &Some(md))
    }

    /// Return the number of bytes encoded so far.
    pub fn tell(&mut self) -> usize {
        self.to.len()
    }
}
