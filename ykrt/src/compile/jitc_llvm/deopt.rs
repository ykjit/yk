//! Run-time deoptimisation support: when a guard fails, this module restores the state necessary
//! to resume interpreter execution.

use super::{
    frame::{BitcodeSection, FrameReconstructor, Value, __yktracec_get_aot_module},
    LLVMCompiledTrace,
};
use crate::{
    compile::{CompiledTrace, GuardId},
    log::stats::TimingState,
    mt::{MTThread, SideTraceInfo},
};
use llvm_sys::orc2::LLVMOrcThreadSafeModuleWithModuleDo;
use llvm_sys::{
    error::{LLVMCreateStringError, LLVMErrorRef},
    prelude::{LLVMModuleRef, LLVMValueRef},
};
use std::mem;
use std::{
    arch::asm,
    ffi::{c_void, CString},
    ptr, slice,
    sync::Arc,
};
use yksmp::Location as SMLocation;

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
/// variable in the AOT module. Mirrors `AOTInfo` in yktracec/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
struct AOTVar {
    val: LLVMValueRef,
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
pub(crate) struct NewFramesInfo {
    // Address of the new stackframes in memory.
    src: *const c_void,
    // Address into the current stack we want to write `src` to.
    dst: *const c_void,
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("__llvm_deoptimize() not yet implemented for this platform");

/// The `__llvm__deoptimize()` function required by `llvm.experimental.deoptimize` intrinsic, that
/// is called during a guard failure. Forwards all its arguments to `__ykrt_deopt` which does the
/// actual deoptimisation. See that function's description for more details. Note, that `ctr` will
/// be cast into an `Arc` inside `__ykrt_deopt` and then dropped.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
extern "C" fn __llvm_deoptimize(
    frameaddr: *mut c_void,
    aotvals: *const c_void,
    actframes: *const c_void,
    guardid: usize,
    jitcallstack: *const c_void,
    isswitchguard: usize,
) -> ! {
    // Push all registers to the stack before they can be clobbered, so that we can find their
    // values after parsing in the stackmap. The order in which we push the registers is equivalent
    // to the Sys-V x86_64 ABI, which the stackmap format uses as well. This function has the
    // "naked" attribute to keep the optimiser from generating the function prologue which messes
    // with the RSP value of the previous stack frame (this value is often referenced by the
    // stackmap).

    unsafe {
        asm!(
            // Copy the return address.
            "mov r12, [rsp]",
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

            // Additional arguments are passed via the stack in reverse order **after** alignment.
            "mov r13, rsp",                   // Copy stack pointer.
            "sub rsp, 8",                     // Align the stack.
            "push r13",                       // Pass current stack pointer.
            "push r12",                       // Pass return address.
            "call __ykrt_deopt",              // Returns NewFramesInfo
            "mov rdi, rax",                   // Pass NewFramesInfo.src as 1st argument.
            "mov rsi, rdx",                   // Pass NewFramesInfo.dst as 2nd argument.
            "call __ykrt_reconstruct_frames", // This doesn't return.
            "ret",
            options(noreturn)
        )
    }
}

/// Struct storing information we need to pass via the `LLVMOrcThreadSafeModuleWithModuleDo`
/// function.
struct ReconstructInfo<'a> {
    ctr: Arc<dyn CompiledTrace>,
    frameaddr: *mut c_void,
    aotvals: &'a LiveAOTVals,
    actframes: &'a CVec,
    retaddr: usize,
    rsp: *const c_void,
    nfi: Option<NewFramesInfo>,
}

/// Collects the relevant information needed for deoptimisation and then reconstructs the stack.
/// Returns a pointer to the new stack and a pointer to the current stack which needs to be
/// overwritten.
extern "C" fn ts_reconstruct(ctx: *mut c_void, _module: LLVMModuleRef) -> LLVMErrorRef {
    let info = unsafe { Box::<&mut ReconstructInfo>::from_raw(ctx as *mut &mut ReconstructInfo) };
    let ctr = Arc::clone(&info.ctr)
        .as_any()
        .downcast::<LLVMCompiledTrace>()
        .unwrap();
    let frameaddr = info.frameaddr;
    let aotvals = info.aotvals;
    let actframes = info.actframes;
    let retaddr = info.retaddr;
    let rsp = info.rsp;

    // Parse the live AOT values.
    let aotvalsptr =
        unsafe { (ctr.aotvals() as *const u8).offset(isize::try_from(aotvals.offset).unwrap()) };

    // Parse the live AOT values.
    let aotvals = unsafe { slice::from_raw_parts(aotvalsptr as *const AOTVar, aotvals.length) };

    // Parse active frames vector.
    // Note that the memory behind this slice is allocated on the stack (of the compiled trace) and
    // thus doesn't need to be freed.
    let activeframes =
        unsafe { slice::from_raw_parts(actframes.addr as *mut LLVMValueRef, actframes.length) };

    // Restore saved registers from the stack.
    let registers = Registers::from_ptr(rsp);

    let mut framerec = unsafe { FrameReconstructor::new(activeframes) };

    // Retrieve the live variables for this guard from this trace's stackmap.
    let live_vars = ctr.smap().get(&retaddr.try_into().unwrap()).unwrap();

    // Extract live values from the stackmap.
    // Skip first live variable that contains 3 unrelated locations (CC, Flags, Num Deopts).
    for (i, locs) in live_vars.iter().skip(1).enumerate() {
        // We currently assume that live variables have at most one location. Extra locations are
        // encoded inside a single location via the offset field. However, encoding further
        // locations may be difficult as we are running out of space, so we may need to extend
        // stackmaps later to allow for multiple locations per variable.
        // Live variables must have a location, so this unwrap should never fail. If it does,
        // something has gone wrong with stackmaps.
        let l = locs.get(0).unwrap();
        match l {
            SMLocation::Register(reg, _size, _off, _extra) => {
                let _val = unsafe { registers.get(*reg) };
                todo!();
            }
            SMLocation::Direct(..) => {
                // We use a (heap allocated) shadow stack for both the JIT and AOT stack, so
                // `Direct` locations can't (currently) arise for JITted code.
                todo!();
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
                framerec.var_init(unsafe { Value::new(aot.val) }, aot.sfidx, v);
            }
            SMLocation::Constant(v) => {
                let aot = &aotvals[i];
                framerec.var_init(unsafe { Value::new(aot.val) }, aot.sfidx, *v as u64);
            }
            SMLocation::LargeConstant(_v) => {
                todo!();
            }
        }
    }

    let (src, btmframesize) = unsafe { framerec.reconstruct_frames(frameaddr) };
    // Calculate the offset on the stack we want to write the new frames to: immediately after the
    // frame containing the control point.
    let newframesdst = unsafe { frameaddr.sub(btmframesize) };
    let nfi = NewFramesInfo {
        src,
        dst: newframesdst,
    };

    info.nfi = Some(nfi);
    Box::into_raw(info);

    let cs = CString::new("").unwrap();
    unsafe { LLVMCreateStringError(cs.into_raw()) }
}

/// Called when a guard failure occurs. After getting access to the global AOT module, passes all
/// the relevant information to `ts_reconstruct` via `ThreadSafeModuleWithModuleDo` to reconstruct
/// the stack.
///
/// The arguments are as follows:
///
///   * `frameaddr`: Address of the control point's frame.
///   * `aotvals`: Struct describing the location of the AOT live variables.
///   * `actframes`: Address and size of vector holding active AOT frame information needed to
///      recreate them.
///   * `guardid`: ID of the failing guard.
///   * `retaddr`: Return address of deoptimize call. Used to find the correct stackmap record.
///   * `jitcallstack`: The parent trace's call stack at the time of the guard failure. Required to
///      assemble a side trace.
///   * `rsp`: Current stack pointer. Needed to read spilled register values from the stack.
///   * `isswitchguard`: 1 if this guards a switch statement, 0 otherwise.
#[cfg(target_arch = "x86_64")]
#[no_mangle]
unsafe extern "C" fn __ykrt_deopt(
    frameaddr: *mut c_void,
    aotvals: &LiveAOTVals,
    actframes: &CVec,
    guardid: usize,
    jitcallstack: *const c_void,
    isswitchguard: usize,
    retaddr: usize,
    rsp: *const c_void,
) -> NewFramesInfo {
    let ctr = MTThread::with(|mtt| mtt.running_trace().unwrap())
        .as_any()
        .downcast::<LLVMCompiledTrace>()
        .unwrap();
    ctr.mt().stats.timing_state(TimingState::Deopting);

    let guardid = GuardId(guardid);

    // Check if we have a side trace and execute it.
    if !ctr.is_last_guard(guardid) {
        let guard = ctr.guard(guardid);
        if let Some(st) = guard.getct() {
            let registers = Registers::from_ptr(rsp);
            let live_vars = ctr.smap().get(&retaddr.try_into().unwrap()).unwrap();
            debug_assert!(!live_vars.is_empty());
            let mut ykctrlpvars = vec![0; live_vars.len() - 1];
            for (i, locs) in live_vars.iter().skip(1).enumerate() {
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
                        ykctrlpvars[i] = addr as u64;
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
                        ykctrlpvars[i] = v;
                    }
                    SMLocation::Constant(v) => {
                        ykctrlpvars[i] = *v as u64;
                    }
                    SMLocation::LargeConstant(_v) => {
                        todo!();
                    }
                }
            }
            // Execute the side trace.
            let f = mem::transmute::<
                *const c_void,
                unsafe extern "C" fn(*mut c_void, *const c_void) -> !,
            >(st.entry());
            ctr.mt().stats.timing_state(TimingState::JitExecuting);
            drop(ctr);

            MTThread::with(|mtt| {
                mtt.set_running_trace(Some(st));
            });

            // FIXME: Calling this function overwrites the current (Rust) function frame,
            // rather than unwinding it. https://github.com/ykjit/yk/issues/778
            f(ykctrlpvars.as_ptr() as *mut c_void, frameaddr);
        }
    }

    MTThread::with(|mtt| {
        mtt.set_running_trace(None);
    });

    let ctrn: Arc<dyn CompiledTrace> = ctr.clone();
    // Copy arguments into a struct we can pass into the ThreadSafeModuleWithModuleDo function.
    let mut info = ReconstructInfo {
        ctr: Arc::clone(&ctrn),
        frameaddr,
        aotvals,
        actframes,
        retaddr,
        rsp,
        nfi: None,
    };

    let infoptr = Box::into_raw(Box::new(&mut info));

    let bc = crate::compile::jitc_llvm::llvmbc_section();
    let moduleref = __yktracec_get_aot_module(&BitcodeSection {
        data: bc.as_ptr(),
        len: u64::try_from(bc.len()).unwrap(),
    });

    // The LLVM CAPI doesn't allow us to manually lock/unlock a ThreadSafeModule, and uses a
    // call-back function instead which it runs after locking the module. This means we need to
    // pass in variables from this scope via a struct which is passed into the function.
    LLVMOrcThreadSafeModuleWithModuleDo(moduleref, ts_reconstruct, infoptr as *mut c_void);

    ctr.mt().stats.timing_state(TimingState::OutsideYk);
    // We want to start side tracing only after we deoptimised. Otherwise we'd trace the whole
    // deopt routine which will later be costly to disassemble.
    debug_assert!(isswitchguard == 0 || isswitchguard == 1);
    if isswitchguard == 0 && !ctr.is_last_guard(guardid) {
        let guard = ctr.guard(guardid);
        if guard.inc_failed(ctr.mt()) {
            // This guard is hot, so compile a new side-trace.
            let aotvalsptr = unsafe {
                (ctr.aotvals() as *const u8).offset(isize::try_from(aotvals.offset).unwrap())
            } as *const c_void;
            let sti = SideTraceInfo {
                callstack: jitcallstack,
                aotvalsptr,
                aotvalslen: aotvals.length,
                guardid,
            };
            panic!()
        }
    }

    info.nfi.unwrap()
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
            // Stash the original value of the `newframesptr` malloc'd block.
            "mov r12, rdi",
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
            // Free `newframesptr`.
            "mov rdi, r12",
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
