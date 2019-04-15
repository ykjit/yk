// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::Pack;
use fallible_iterator::FallibleIterator;
use rmp_serde::{
    decode::{self, ReadReader},
    Deserializer,
};
use serde::Deserialize;
use std::io::Read;

/// The pack decoder.
/// Offers a simple iterator interface to serialised packs.
pub struct Decoder<'a> {
    deser: Deserializer<ReadReader<&'a mut dyn Read>>,
}

impl<'a> Decoder<'a> {
    /// Returns a new decoder which will deserialise from `read_from`.
    pub fn from(read_from: &'a mut dyn Read) -> Self {
        let deser = Deserializer::new(read_from);
        Self { deser }
    }
}

impl<'a> FallibleIterator for Decoder<'a> {
    type Item = Pack;
    type Error = decode::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        Option::<Pack>::deserialize(&mut self.deser)
    }
}
