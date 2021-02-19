//! The pack encoder.
//!
//! This is used by ykrustc to encode SIR elements into the end binary.

use crate::Pack;

pub struct Encoder<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> Encoder<'a> {
    /// Creates an encoder which serialises into the vector `buf`.
    pub fn from(buf: &'a mut Vec<u8>) -> Self {
        Self { buf }
    }

    /// Serialises a pack.
    pub fn serialise(&mut self, md: Pack) -> Result<(), bincode::Error> {
        bincode::serialize_into(&mut *self.buf, &Some(md))
    }

    /// Return the number of bytes encoded so far.
    pub fn tell(&mut self) -> usize {
        self.buf.len()
    }
}
