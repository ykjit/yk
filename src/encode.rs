// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::Pack;
use rmp_serde::{encode, Serializer};
use serde::Serialize;
use std::{io::prelude::*, ops::Drop};

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

impl<'a> Drop for Encoder<'a> {
    fn drop(&mut self) {
        if !self.done {
            panic!("Encoder not marked done()");
        }
    }
}
