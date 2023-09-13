//! Yk's AOT IR deserialiser.
//!
//! This is a parser for the on-disk (in the ELF binary) IR format used to express the
//! ahead-of-time compiled interpreter.
//!
//! The `Display` implementations convert the in-memory data structures into a human-readable
//! textual format.

use deku::prelude::*;
use std::{
    error::Error,
    ffi::CStr,
    fmt::{self, Display},
};

/// A magic number that all bytecode payloads begin with.
const MAGIC: u32 = 0xedd5f00d;
/// The version of the bytecode format.
const FORMAT_VERSION: u32 = 0;

fn deserialise_string(v: Vec<u8>) -> Result<String, DekuError> {
    let err = Err(DekuError::Parse("failed to parse string".to_owned()));
    match CStr::from_bytes_until_nul(v.as_slice()) {
        Ok(c) => match c.to_str() {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => err,
        },
        _ => err,
    }
}

/// An instruction opcode.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(type = "u8")]
pub(crate) enum Opcode {
    Nop = 0,
}

/// A bytecode instruction.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Instruction {
    opcode: Opcode,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.opcode)
    }
}

/// A basic block containing bytecode instructions.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Block {
    #[deku(temp)]
    num_instrs: usize,
    #[deku(count = "num_instrs")]
    instrs: Vec<Instruction>,
}

impl Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in &self.instrs {
            writeln!(f, "    {}", i)?;
        }
        Ok(())
    }
}

/// A function.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Function {
    #[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")]
    name: String,
    #[deku(temp)]
    num_blocks: usize,
    #[deku(count = "num_blocks")]
    blocks: Vec<Block>,
}

impl Function {
    fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_declaration() {
            // declarations have no body, so print it as such.
            write!(f, "func {};", self.name)
        } else {
            writeln!(f, "func {} {{", self.name)?;
            for (i, b) in self.blocks.iter().enumerate() {
                write!(f, "  bb{}:\n{}", i, b)?;
            }
            write!(f, "}}")
        }
    }
}

/// A bytecode module.
///
/// This is the top-level container for the bytecode.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct AOTModule {
    #[deku(assert = "*magic == MAGIC", temp)]
    magic: u32,
    #[deku(assert = "*version == FORMAT_VERSION")]
    version: u32,
    #[deku(temp)]
    num_funcs: usize,
    #[deku(count = "num_funcs")]
    funcs: Vec<Function>,
}

impl Display for AOTModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "# IR format version: {}", self.version)?;
        for func in &self.funcs {
            writeln!(f, "\n{}", func)?;
        }
        Ok(())
    }
}

/// Deserialise an AOT module from the slice `data`.
pub(crate) fn deserialise_module(data: &[u8]) -> Result<AOTModule, Box<dyn Error>> {
    match AOTModule::from_bytes((data, 0)) {
        Ok(((_, _), modu)) => Ok(modu),
        Err(e) => Err(e.to_string().into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{deserialise_module, deserialise_string, Opcode, FORMAT_VERSION, MAGIC};
    use byteorder::{NativeEndian, WriteBytesExt};
    use std::ffi::CString;

    #[cfg(target_pointer_width = "64")]
    fn write_native_usize(d: &mut Vec<u8>, v: usize) {
        d.write_u64::<NativeEndian>(v as u64).unwrap(); // `as` is safe: u64 == usize
    }

    fn write_str(d: &mut Vec<u8>, s: &str) {
        d.extend(s.as_bytes());
        d.push(0); // null terminator.
    }

    #[test]
    fn deser_and_display() {
        let mut data = Vec::new();

        // magic
        data.write_u32::<NativeEndian>(MAGIC).unwrap();
        // version
        data.write_u32::<NativeEndian>(FORMAT_VERSION).unwrap();

        // num_functions
        write_native_usize(&mut data, 2);

        // funcs[0].name
        write_str(&mut data, "foo");
        // funcs[0].num_blocks
        write_native_usize(&mut data, 2);
        // funcs[0].blocks[0].num_instrs
        write_native_usize(&mut data, 2);
        // funcs[0].blocks[0].instrs[0].opcode
        data.write_u8(Opcode::Nop as u8).unwrap();
        // funcs[0].blocks[0].instrs[1].opcode
        data.write_u8(Opcode::Nop as u8).unwrap();
        // funcs[0].blocks[1].num_instrs
        write_native_usize(&mut data, 1);
        // funcs[0].blocks[1].instrs[0].opcode
        data.write_u8(Opcode::Nop as u8).unwrap();

        // funcs[1].name
        write_str(&mut data, "bar");
        // funcs[0].num_blocks
        write_native_usize(&mut data, 0);

        let test_mod = deserialise_module(data.as_slice()).unwrap();
        let string_mod = format!("{}", test_mod);

        println!("{}", string_mod);
        let expect = "\
# IR format version: 0

func foo {
  bb0:
    Nop
    Nop
  bb1:
    Nop
}

func bar;
";
        assert_eq!(expect, string_mod);
    }

    #[test]
    fn string_deser() {
        let check = |s: &str| {
            assert_eq!(
                &deserialise_string(CString::new(s).unwrap().into_bytes_with_nul()).unwrap(),
                s
            );
        };
        check("testing");
        check("the quick brown fox jumped over the lazy dog");
        check("");
        check("         ");
    }
}
