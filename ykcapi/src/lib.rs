//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![feature(bench_black_box)]
#![feature(c_variadic)]
#![feature(naked_functions)]
#![feature(once_cell)]

#[cfg(feature = "yk_testing")]
mod testing;

use std::arch::asm;
use std::convert::TryInto;
use std::ffi::c_void;
use std::process;
use std::{ptr, slice};
use ykrt::{print_jit_state, HotThreshold, Location, MT};
use yksmp::{Location as SMLocation, StackMapParser};

#[no_mangle]
pub extern "C" fn yk_mt_new() -> *mut MT {
    let mt = Box::new(MT::new());
    Box::into_raw(mt)
}

#[no_mangle]
pub extern "C" fn yk_mt_drop(mt: *mut MT) {
    unsafe { Box::from_raw(mt) };
}

// The "dummy control point" that is replaced in an LLVM pass.
#[no_mangle]
pub extern "C" fn yk_control_point(_mt: *mut MT, _loc: *mut Location) {
    // Intentionally empty.
}

// The "real" control point, that is called once the interpreter has been patched by ykllvm.
#[no_mangle]
pub extern "C" fn __ykrt_control_point(mt: *mut MT, loc: *mut Location, ctrlp_vars: *mut c_void) {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let mt = unsafe { &*mt };
        let loc = unsafe { &*loc };
        mt.control_point(loc, ctrlp_vars);
    }
}

#[no_mangle]
pub extern "C" fn yk_hot_threshold_set(mt: &MT, hot_threshold: HotThreshold) {
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

/// Parses the stackmap and saved registers from the given address (i.e. the return address of the
/// deoptimisation call).
/// FIXME: Until we have the stopgap interpreter we simply print out the values we find.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub extern "C" fn yk_stopgap(
    sm_addr: *const c_void,
    sm_size: usize,
    retaddr: usize,
    rsp: *const c_void,
) {
    // FIXME: remove once we have a stopgap interpreter.
    #[cfg(feature = "jit_state_debug")]
    print_jit_state("stopgap");
    // Restore saved registers from the stack.
    let registers = Registers::from_ptr(rsp);

    // Parse the stackmap.
    let slice = unsafe { slice::from_raw_parts(sm_addr as *mut u8, sm_size) };
    let map = StackMapParser::parse(slice).unwrap();
    let locs = map.get(&retaddr.try_into().unwrap()).unwrap();

    // Extract live values from the stackmap.
    for l in locs {
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
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Indirect: {} ({} {})", v, reg, off);
            }
            SMLocation::Constant(v) => {
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Constant: {}", v);
            }
            SMLocation::LargeConstant(v) => {
                // FIXME: remove once we have a stopgap interpreter.
                eprintln!("Large constant: {}", v);
            }
        }
    }
    // FIXME: Initialise stopgap interpreter here.
    process::exit(0);
}

/// The `__llvm__deoptimize()` function required by `llvm.experimental.deoptimize` intrinsic, that
/// we use for exiting to the stop-gap interpreter on guard failure.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
pub extern "C" fn __llvm_deoptimize(addr: *const c_void, size: usize) {
    // Push all registers to the stack before they can be clobbered, so that we can find their
    // values after parsing in the stackmap. The order in which we push the registers is equivalent
    // to the Sys-V x86_64 ABI, which the stackmap format uses as well. This function has the
    // "naked" attribute to keep the optimiser from generating the function prologue which messes
    // with the RSP value of the previous stack frame (this value is often referenced by the
    // stackmap).
    unsafe {
        asm!(
            // Save registers to the stack.
            // FIXME: Add other registers that may be referenced by the stackmap.
            "push rsp",
            "push rbp",
            "push rdi",
            "push rsi",
            "push rbx",
            "push rcx",
            "push rdx",
            "push rax",
            // Now we need to call yk_stopgap. The arguments need to be in RDI,
            // RSI, RDX, and RCX. The first two arguments (stackmap address and
            // stackmap size) are already where they need to be as we are just
            // forwarding them from the current function's arguments. The remaining
            // arguments (return address and current stack pointer) need to be in
            // RDX and RCX. The return address was at [RSP] before the above
            // pushes, so to find it we need to offset 8 bytes per push.
            "mov rdx, [rsp+64]",
            "mov rcx, rsp",
            "sub rsp, 8",
            "call yk_stopgap",
            "add rsp, 64",
            "ret",
            options(noreturn)
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("__llvm_deoptimize() not yet implemented for this platform");
