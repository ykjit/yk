//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![feature(c_variadic)]
#![feature(naked_functions)]
#![feature(once_cell)]

#[cfg(feature = "yk_testing")]
mod testing;

use std::arch::asm;
use std::convert::{TryFrom, TryInto};
use std::ffi::c_void;
use std::{ptr, slice};
use ykfr::{self, FrameReconstructor};
#[cfg(feature = "yk_jitstate_debug")]
use ykrt::print_jit_state;
use ykrt::{HotThreshold, Location, MT};
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
pub extern "C" fn yk_mt_control_point(_mt: *mut MT, _loc: *mut Location) {
    // Intentionally empty.
}

// The "real" control point, that is called once the interpreter has been patched by ykllvm.
// Returns the address of a reconstructed stack or null if there wasn#t a guard failure.
#[no_mangle]
pub extern "C" fn __ykrt_control_point(
    mt: *mut MT,
    loc: *mut Location,
    ctrlp_vars: *mut c_void,
    // Frame address of caller.
    frameaddr: *mut c_void,
) -> *const c_void {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let mt = unsafe { &*mt };
        let loc = unsafe { &*loc };
        return mt.control_point(loc, ctrlp_vars, frameaddr);
    }
    std::ptr::null()
}

#[no_mangle]
pub extern "C" fn yk_mt_hot_threshold_set(mt: &MT, hot_threshold: HotThreshold) {
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
        if id > 7 {
            unreachable!(
                "Register #{} currently not saved during deoptimisation.",
                id
            )
        }
        let val = self.read_from_stack(id.try_into().unwrap());
        // Due to the return address being pushed to the stack before we store RSP, its value is
        // off by 8 bytes.
        if id == 7 {
            todo!(); // Check this is still true now that llvm_deoptimize is a naked function.
        }
        val
    }
}

/// Location in terms of basic block index, instruction index, and function name, of a
/// variable in the AOT module. Mirrors the LLVM struct defined in ykllvmwrap/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
struct AOTVar {
    bbidx: usize,
    instridx: usize,
    fname: *const i8,
    sfidx: usize,
}

/// Address and length of a vector. Mirrors the struct defined in
/// ykllvmwrap/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
pub struct CVec {
    // FIXME rename to PtrLen
    addr: *const c_void,
    length: usize,
}

/// Address, offset, and length of the live AOT values for this guard failure. Mirrors the struct
/// defined in ykllvmwrap/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
pub struct LiveAOTVals {
    addr: *const c_void,
    offset: usize,
    length: usize,
}

/// After a guard failure, reconstructs the stack frames and registers and then jumps back to the
/// AOT code from where to continue.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
pub extern "C" fn __ykrt_reconstruct_frames(newframesptr: *const c_void) {
    unsafe {
        asm!(
            // The first 8 bytes of the new frames is the size of the map, needed for copying it
            // over. Move it into RDX, but reduce it by 8 bytes, since we'll also adjust RDI next
            // to make it point to the actual beginning of the new frames (jumping over the length
            // value stored at the beginning of RDI).
            "mov rdx, [rdi]",
            "sub rdx, 8",
            // Then adjust the address to where the new stack actually starts.
            "add rdi, 8",
            // Make space for the new stack, but use 8 bytes less in order to overwrite this
            // function's return address since we won't be returning there.
            "add rsp, 8",
            "sub rsp, rdx",
            // Copy over the new stack frames.
            "mov rsi, rdi", // 2nd arg: src
            "mov rdi, rsp", // 1st arg: dest
            "call memcpy",
            // Now move the source (i.e. the heap allocated frames) into the first argument and its
            // size into the second. Then free the memory.
            "mov rdi, rsi",
            // Adjust rdi back to beginning of `newframesptr`.
            "sub rdi, 8",
            "mov rsi, rdx",
            // Adjust length back to full size to free entire mmap.
            "add rsi, 8",
            "call munmap",
            // Restore registers.
            // FIXME: Add other registers that may need restoring (e.g. floating point).
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop r11",
            "pop r10",
            "pop r9",
            "pop r8",
            "pop rbp",
            "pop rdi",
            "pop rsi",
            "pop rbx",
            "pop rcx",
            "pop rdx",
            "pop rax",
            // Load new return address from the stack and jump to it.
            "add rsp, 8",
            "jmp [rsp-8]",
            options(noreturn)
        )
    }
}

/// Called when a guard failure occurs. Handles the reading of stackmaps, matching of JIT to AOT
/// variables, etc., in order to reconstruct the stack.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
pub extern "C" fn yk_stopgap(
    stackmap: &CVec,
    aotvals: &LiveAOTVals,
    actframes: &CVec,
    frameaddr: *mut c_void,
    retaddr: usize,
    rsp: *const c_void,
) -> *const c_void {
    #[cfg(feature = "yk_jitstate_debug")]
    print_jit_state("deoptimise");

    // Parse the live AOT values.
    let aotvalsptr =
        unsafe { (aotvals.addr as *const u8).offset(isize::try_from(aotvals.offset).unwrap()) };
    let aotvals = unsafe { slice::from_raw_parts(aotvalsptr as *const AOTVar, aotvals.length) };

    // Parse active frames vector.
    // Note that the memory behind this slice is allocated on the stack (of the compiled trace) and
    // thus doesn't need to be freed.
    let activeframes = unsafe {
        slice::from_raw_parts(actframes.addr as *const ykfr::FrameInfo, actframes.length)
    };

    // Restore saved registers from the stack.
    let registers = Registers::from_ptr(rsp);

    let mut framerec = unsafe { FrameReconstructor::new(activeframes) };

    // Parse the stackmap of the JIT module.
    // OPT: Parsing the stackmap and initialising `framerec` is slow and could be heavily reduced
    // by caching the result.
    let slice = unsafe { slice::from_raw_parts(stackmap.addr as *mut u8, stackmap.length) };
    let map = StackMapParser::parse(slice).unwrap();
    let live_vars = map.get(&retaddr.try_into().unwrap()).unwrap();

    // Extract live values from the stackmap.
    // Skip first live variable that contains 3 unrelated locations (CC, Flags, Num Deopts).
    for (i, locs) in live_vars.iter().skip(1).enumerate() {
        // The stopgap interpreter assumes that each live value has at most one location. This
        // isn't always true. Fixing it could be involved, but since we are planning on deleting
        // the stopgap interpreter, let's just add an assertion for now.
        assert!(locs.len() == 1);
        let l = locs.get(0).unwrap();
        match l {
            SMLocation::Register(reg, _size, _off) => {
                let _val = unsafe { registers.get(*reg) };
                todo!();
            }
            SMLocation::Direct(reg, off, _size) => {
                // When using `llvm.experimental.deoptimize` then direct locations should always be
                // in relation to RBP.
                assert_eq!(*reg, 6);
                let addr = unsafe { registers.get(*reg) as *mut u8 };
                let addr = unsafe { addr.offset(isize::try_from(*off).unwrap()) };
                let aot = &aotvals[i];
                unsafe {
                    framerec.var_init(
                        aot.bbidx,
                        aot.instridx,
                        std::ffi::CStr::from_ptr(aot.fname),
                        aot.sfidx,
                        addr as u64,
                    );
                }
            }
            SMLocation::Indirect(reg, off, size) => {
                let addr = unsafe { registers.get(*reg) as *mut u8 };
                let addr = unsafe { addr.offset(isize::try_from(*off).unwrap()) };
                let v = match *size {
                    1 => unsafe { ptr::read::<u8>(addr as *mut u8) as u64 },
                    2 => unsafe { ptr::read::<u16>(addr as *mut u16) as u64 },
                    4 => unsafe { ptr::read::<u32>(addr as *mut u32) as u64 },
                    8 => unsafe { ptr::read::<u64>(addr as *mut u64) },
                    _ => unreachable!(),
                };
                let aot = &aotvals[i];
                unsafe {
                    framerec.var_init(
                        aot.bbidx,
                        aot.instridx,
                        std::ffi::CStr::from_ptr(aot.fname),
                        aot.sfidx,
                        v,
                    );
                }
            }
            SMLocation::Constant(v) => {
                let aot = &aotvals[i];
                unsafe {
                    framerec.var_init(
                        aot.bbidx,
                        aot.instridx,
                        std::ffi::CStr::from_ptr(aot.fname),
                        aot.sfidx,
                        *v as u64,
                    );
                }
            }
            SMLocation::LargeConstant(_v) => {
                todo!();
            }
        }
    }

    framerec.reconstruct_frames(frameaddr)
}

/// The `__llvm__deoptimize()` function required by `llvm.experimental.deoptimize` intrinsic, that
/// we use for exiting to the stop-gap interpreter on guard failure.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
pub extern "C" fn __llvm_deoptimize(
    stackmap: *const c_void,
    aotvals: *const c_void,
    frames: *const c_void,
    retval: *mut c_void,
) -> *const c_void {
    // Push all registers to the stack before they can be clobbered, so that we can find their
    // values after parsing in the stackmap. The order in which we push the registers is equivalent
    // to the Sys-V x86_64 ABI, which the stackmap format uses as well. This function has the
    // "naked" attribute to keep the optimiser from generating the function prologue which messes
    // with the RSP value of the previous stack frame (this value is often referenced by the
    // stackmap).
    unsafe {
        asm!(
            // Save registers that may be referenced by the stackmap to the stack before they get
            // overwritten, so that we read their values later during stopgapping.
            // FIXME: Add other registers that may be referenced by the stackmap.
            "push rsp",
            "push rbp",
            "push rdi",
            "push rsi",
            "push rbx",
            "push rcx",
            "push rdx",
            "push rax",
            // Now we need to call yk_stopgap. The arguments need to be in RDI, RSI, RDX,
            // RCX, R8, and R9. The first four arguments (stackmap
            // live variable map, frames, and return value pointer) are already where they
            // need to be as we are just forwarding them from the current function's
            // arguments. The remaining arguments (return address and current stack
            // pointer) need to be in R8 and R9. The return address was at [RSP] before
            // the above pushes, so to find it we need to offset 8 bytes per push.
            "mov r8, [rsp+64]",
            "mov r9, rsp",
            "sub rsp, 8", // Alignment
            "call yk_stopgap",
            "add rsp, 72",
            // FIXME: Don't rely on RBP being pushed. Use frame size retrieved from
            // stackmap instead.
            "mov rsp, rbp",
            "pop rbp",
            "ret",
            options(noreturn)
        )
    }
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("__llvm_deoptimize() not yet implemented for this platform");
