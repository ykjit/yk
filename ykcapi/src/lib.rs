//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![feature(bench_black_box)]
#![feature(c_variadic)]
#![feature(once_cell)]

use std::convert::TryInto;
use std::ffi::c_void;
use std::process;
use std::{ptr, slice};
use ykrt::{HotThreshold, Location, MT};
use yksmp::{Location as SMLocation, StackMapParser};

mod sginterp;
use sginterp::{SGInterp, SGValue};

/// The first three locations of an LLVM stackmap record, according to the source, are CC, Flags,
/// Num Deopts, which need to be skipped when mapping the stackmap values back to AOT variables.
const SM_REC_HEADER: usize = 3;

mod llvmapihelper;

// The "dummy control point" that is replaced in an LLVM pass.
#[no_mangle]
pub extern "C" fn yk_control_point(_loc: *mut Location) {
    // Intentionally empty.
}

// The "real" control point, that is called once the interpreter has been patched by ykllvm.
#[no_mangle]
pub extern "C" fn __ykrt_control_point(loc: *mut Location, ctrlp_vars: *mut c_void) {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let mt = MT::global();
        let loc = unsafe { &*loc };
        mt.control_point(loc, ctrlp_vars);
    }
}

#[no_mangle]
pub extern "C" fn yk_set_hot_threshold(hot_threshold: HotThreshold) {
    let mt = MT::global();
    mt.set_hot_threshold(hot_threshold);
}

#[no_mangle]
pub extern "C" fn yk_location_new() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_location_drop(loc: Location) {
    drop(loc)
}

/// Reads out registers spilled to the stack of the previous frame during the deoptimisation
/// routine. The order of the registers are in accordance to the DWARF register number mapping
/// referenced in the SystemV ABI manual (https://uclibc.org/docs/psABI-x86_64.pdf).
struct Registers {
    addr: *const usize,
}

impl Registers {
    /// Creates a Registers struct from a given a pointer on the stack containing spilled
    /// registers.
    fn from_ptr(ptr: *const c_void) -> Registers {
        Registers {
            addr: ptr as *const usize,
        }
    }

    /// Read the spilled register value at the offset `off` from the previous stack frame.
    unsafe fn read_from_stack(&self, off: isize) -> usize {
        ptr::read::<usize>(self.addr.offset(off))
    }

    /// Retrieve the previous frame's register value given by its DWARF register number `id`. This
    /// number additionally functions as an offset into the the spilled stack to find that
    /// register's value.
    #[cfg(target_arch = "x86_64")]
    unsafe fn get(&self, id: u16) -> usize {
        if id == 6 {
            unreachable!("We currently have no way to access RBP of the previous stackframe.")
        }
        if id > 7 {
            unreachable!(
                "Register #{} currently not saved during deoptimisation.",
                id
            )
        }
        let mut val = self.read_from_stack(id.try_into().unwrap());
        // Due to the return address being pushed to the stack before we store RSP, its value is
        // off by 8 bytes.
        if id == 7 {
            val += std::mem::size_of::<usize>();
        }
        val
    }
}

#[derive(Debug)]
#[repr(C)]
struct AOTVar {
    bbidx: u32,
    instridx: u32,
    fname: *const i8,
}

#[derive(Debug)]
#[repr(C)]
pub struct AOTMap {
    addr: *const c_void,
    size: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct CurPos {
    bbidx: u32,
    instridx: u32,
    fname: *const i8,
}

/// Parses the stackmap and saved registers from the given address (i.e. the return address of the
/// deoptimisation call).
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub extern "C" fn yk_stopgap(
    sm_addr: *const c_void,
    sm_size: usize,
    aotmap: &AOTMap,
    curpos: &CurPos,
    retaddr: usize,
    rsp: *const c_void,
) {
    // FIXME: remove once we have a stopgap interpreter.
    eprintln!("jit-state: stopgap");

    // Parse AOTMap.
    let aotmap = unsafe { slice::from_raw_parts(aotmap.addr as *const AOTVar, aotmap.size) };

    // Restore saved registers from the stack.
    let registers = Registers::from_ptr(rsp);

    let mut sginterp = unsafe {
        SGInterp::new(
            curpos.bbidx,
            curpos.instridx,
            std::ffi::CStr::from_ptr(curpos.fname),
        )
    };

    // Parse the stackmap.
    let slice = unsafe { slice::from_raw_parts(sm_addr as *mut u8, sm_size) };
    let map = StackMapParser::parse(slice).unwrap();
    let locs = map.get(&retaddr.try_into().unwrap()).unwrap();

    // Extract live values from the stackmap.
    // Skip first 3 locations as they don't relate to any of our live variables.
    for (i, l) in locs.iter().skip(SM_REC_HEADER).enumerate() {
        match l {
            SMLocation::Register(reg, size) => {
                // FIXME: remove once we have a stopgap interpreter.
                let val = unsafe { registers.get(*reg) };
                eprintln!("Register: {} ({} {})", val, reg, size);
            }
            SMLocation::Direct(reg, off, size) => {
                // When using `llvm.experimental.deoptimize` then direct locations should always be
                // in relation to RSP.
                assert_eq!(*reg, 7);
                let addr = unsafe { registers.get(*reg) + (*off as usize) };
                let v = match *size {
                    1 => unsafe { ptr::read::<u8>(addr as *mut u8) as u64 },
                    2 => unsafe { ptr::read::<u16>(addr as *mut u16) as u64 },
                    4 => unsafe { ptr::read::<u32>(addr as *mut u32) as u64 },
                    8 => unsafe { ptr::read::<u64>(addr as *mut u64) as u64 },
                    _ => unreachable!(),
                };
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Direct: {} ({} {})", v, reg, off);
            }
            SMLocation::Indirect(reg, off, size) => {
                let addr = unsafe { registers.get(*reg) + (*off as usize) };
                let v = match *size {
                    1 => unsafe { ptr::read::<u8>(addr as *mut u8) as u64 },
                    2 => unsafe { ptr::read::<u16>(addr as *mut u16) as u64 },
                    4 => unsafe { ptr::read::<u32>(addr as *mut u32) as u64 },
                    8 => unsafe { ptr::read::<u64>(addr as *mut u64) as u64 },
                    _ => unreachable!(),
                };
                let aot = &aotmap[i];
                unsafe {
                    sginterp.init_live(
                        aot.bbidx,
                        aot.instridx,
                        std::ffi::CStr::from_ptr(aot.fname),
                        SGValue::U64(v),
                    );
                }
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Indirect: {} ({} {} {})", v, reg, off, size);
            }
            SMLocation::Constant(v) => {
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Constant: {}", v);
                let aot = &aotmap[i];
                unsafe {
                    sginterp.init_live(
                        aot.bbidx,
                        aot.instridx,
                        std::ffi::CStr::from_ptr(aot.fname),
                        SGValue::U32(*v),
                    );
                }
            }
            SMLocation::LargeConstant(v) => {
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Large constant: {}", v);
            }
        }
    }
    unsafe { sginterp.interpret() };
    process::exit(0);
}

/// The following module contains exports only used for testing from external C code.
/// These symbols are not shipped as part of the main API.
#[cfg(feature = "c_testing")]
mod c_testing {
    use yktrace::BlockMap;

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_new() -> *mut BlockMap {
        Box::into_raw(Box::new(BlockMap::new()))
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_free(bm: *mut BlockMap) {
        unsafe { Box::from_raw(bm) };
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_len(bm: *mut BlockMap) -> usize {
        unsafe { &*bm }.len()
    }
}
