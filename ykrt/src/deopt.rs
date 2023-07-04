//! Run-time deoptimisation support: when a guard fails, this module restores the state necessary
//! to resume interpreter execution.

use crate::frame::{FrameInfo, FrameReconstructor};
#[cfg(feature = "yk_jitstate_debug")]
use crate::print_jit_state;
use crate::trace::CompiledTrace;
use std::{arch::asm, ffi::c_void, ptr, slice, sync::Arc};
use yksmp::{Location as SMLocation, StackMapParser};

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
/// variable in the AOT module. Mirrors the LLVM struct defined in yktracec/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
struct AOTVar {
    bbidx: usize,
    instridx: usize,
    fname: *const i8,
    sfidx: usize,
}

/// Address and length of a vector. Mirrors the struct defined in
/// yktracec/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
struct CVec {
    // FIXME rename to PtrLen
    addr: *const c_void,
    length: usize,
}

/// Address, offset, and length of the live AOT values for this guard failure. Mirrors the struct
/// defined in yktracec/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
struct LiveAOTVals {
    offset: usize,
    length: usize,
}

/// Struct containing pointers needed for frame reconstruction.
#[derive(Debug)]
#[repr(C)]
pub struct NewFramesInfo {
    // Address of the new stackframes in memory.
    src: *const c_void,
    // Address into the current stack we want to write `src` to.
    dst: *const c_void,
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("__llvm_deoptimize() not yet implemented for this platform");

/// The `__llvm__deoptimize()` function required by `llvm.experimental.deoptimize` intrinsic, that
/// is called during a guard failure.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
extern "C" fn __llvm_deoptimize(
    ctr: *const CompiledTrace,
    frameaddr: *mut c_void,
    aotvals: *const c_void,
    actframes: *const c_void,
    guardid: usize,
) -> ! {
    // Push all registers to the stack before they can be clobbered, so that we can find their
    // values after parsing in the stackmap. The order in which we push the registers is equivalent
    // to the Sys-V x86_64 ABI, which the stackmap format uses as well. This function has the
    // "naked" attribute to keep the optimiser from generating the function prologue which messes
    // with the RSP value of the previous stack frame (this value is often referenced by the
    // stackmap).

    unsafe {
        asm!(
            // Save registers that may be referenced by the stackmap to the stack before they get
            // overwritten, so that we can read their values later during deoptimisation.
            // FIXME: Add other registers that may be referenced by the stackmap.
            "push rsp",
            "push rbp",
            "push rdi",
            "push rsi",
            "push rbx",
            "push rcx",
            "push rdx",
            "push rax",
            // Since we are just passing through the arguments of this function to `__ykrt_deopt`,
            // we don't need do anything here. However, we need to add some extra arguments to the
            // call.

            // The return address was at [RSP] before the above pushes, so to find it we need to
            // offset 8 bytes per push.
            "mov r9, [rsp+64]",
            "push rsp",                       // Current stack pointer.
            "call __ykrt_deopt",              // Returns NewFramesInfo
            "mov rdi, rax",                   // Pass NewFramesInfo.src as 1st argument.
            "mov rsi, rdx",                   // Pass NewFramesInfo.dst as 2nd argument.
            "call __ykrt_reconstruct_frames", // This doesn't return.
            "ret",
            options(noreturn)
        )
    }
}

/// Called when a guard failure occurs. Handles the reading of stackmaps, matching of JIT to AOT
/// variables, etc., in order to reconstruct the stack.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
unsafe extern "C" fn __ykrt_deopt(
    ctr: *const CompiledTrace,
    // Address of the control point's frame.
    frameaddr: *mut c_void,
    // Struct describing the location of the AOT live variables.
    aotvals: &LiveAOTVals,
    // Address and size of vector holding active AOT frame information needed to recreate them.
    actframes: &CVec,
    // ID of the failing guard.
    guardid: usize,
    // Return address of deoptimize call. Used to find the correct stackmap record.
    retaddr: usize,
    // Current stack pointer. Needed to read spilled register values from the stack.
    rsp: *const c_void,
) -> NewFramesInfo {
    #[cfg(feature = "yk_jitstate_debug")]
    print_jit_state("deoptimise");

    // Put the CompiledTrace back into an Arc, so it is dropped properly.
    let ctr = Arc::from_raw(ctr);

    // FIXME: Check here if we have a side trace and execute it. Otherwise just increment the guard
    // failure counter.

    // Parse the live AOT values.
    let aotvalsptr =
        unsafe { (ctr.aotvals as *const u8).offset(isize::try_from(aotvals.offset).unwrap()) };
    let aotvals = unsafe { slice::from_raw_parts(aotvalsptr as *const AOTVar, aotvals.length) };

    // Parse active frames vector.
    // Note that the memory behind this slice is allocated on the stack (of the compiled trace) and
    // thus doesn't need to be freed.
    let activeframes =
        unsafe { slice::from_raw_parts(actframes.addr as *const FrameInfo, actframes.length) };

    // Restore saved registers from the stack.
    let registers = Registers::from_ptr(rsp);

    let mut framerec = unsafe { FrameReconstructor::new(activeframes) };

    // Parse the stackmap of the JIT module.
    // OPT: Parsing the stackmap and initialising `framerec` is slow and could be heavily reduced
    // by caching the result.
    let slice = unsafe { slice::from_raw_parts(ctr.smptr as *mut u8, ctr.smsize) };
    let map = StackMapParser::parse(slice).unwrap();
    let live_vars = map.get(&retaddr.try_into().unwrap()).unwrap();

    // Extract live values from the stackmap.
    // Skip first live variable that contains 3 unrelated locations (CC, Flags, Num Deopts).
    for (i, locs) in live_vars.iter().skip(1).enumerate() {
        // We currently assume that live variables have at most one location. We currently encode
        // extra locations inside a single location. But we are running out of space so we might
        // need to extend stackmaps later to allow for multiple locations per variable.
        assert!(locs.len() == 1);
        let l = locs.get(0).unwrap();
        match l {
            SMLocation::Register(reg, _size, _off, _extra) => {
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
                    1 => unsafe { ptr::read::<u8>(addr) as u64 },
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

    let (src, btmframesize) = unsafe { framerec.reconstruct_frames(frameaddr) };
    // Calculate the offset on the stack we want to write the new frames to: immediately after the
    // frame containing the control point.
    let newframesdst = frameaddr.sub(btmframesize);
    NewFramesInfo {
        src,
        dst: newframesdst,
    }
}

/// After a guard failure, reconstructs the stack frames and registers and then jumps back to the
/// AOT code from where to continue.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
extern "C" fn __ykrt_reconstruct_frames(
    newframesptr: *const c_void,
    newframedst: *const c_void,
) -> ! {
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
            // Reset RSP to the end of the control point's caller's frame (this doesn't include the
            // return address which will thus be overwritten in the process)
            "mov rsp, rsi",
            // Make space for the new frame.
            "sub rsp, rdx",
            // Copy over the new stack frames.
            "mov rsi, rdi", // 2nd arg: src
            "mov rdi, rsp", // 1st arg: dest
            "call memcpy",
            // Now move the source (i.e. the heap allocated frames) into the first argument and its
            // size into the second.
            "mov rdi, rsi",
            // Adjust rdi back to beginning of `newframesptr`.
            "sub rdi, 8",
            // Free the malloced memory.
            "call free",
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
            // At this point the address we want to deoptimise to is on top of the stack. The `ret`
            // instruction will pop it from the stack and jump to it.
            "ret",
            options(noreturn)
        )
    }
}
