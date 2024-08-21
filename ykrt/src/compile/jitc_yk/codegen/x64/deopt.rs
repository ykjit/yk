use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        jitc_yk::codegen::reg_alloc::{Register, VarLocation},
        GuardIdx,
    },
    log::{stats::TimingState, Verbosity},
    mt::MTThread,
};
use dynasmrt::Register as _;
use libc::c_void;
use std::{mem, ptr, sync::Arc};
use yksmp::Location as SMLocation;

use super::{X64CompiledTrace, RBP_DWARF_NUM, REG64_SIZE};

/// Registers (in DWARF notation) that we want to restore during deopt. Excludes `rsp` (7) and
/// `return register` (16), which we do not care about.
const RECOVER_REG: [usize; 31] = [
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32,
];

/// The number of DWARF registers required to cover the general purpose and float registers,
/// including those omitted from `RECOVER_REG` above (register 32 is XMM15 and numbering starts
/// from zero). This is used to allocate arrays whose indices need to be the DWARF register number.
const REGISTER_NUM: usize = RECOVER_REG.len() + 2;

/// Deoptimise back to the interpreter. This function is called from a failing guard (see
/// `x86_64/mod.rs`). The arguments are: `frameaddr` is the RBP value for the caller of the JIT
/// function frame; `gidx` the ID of the failing guard; `jitrbp` is the JIT function frame's RBP;
/// and `gp_regs` is a pointer to the saved values of the 16 general purpose registers in the same
/// order as [lsregalloc::GP_REGS].
#[no_mangle]
pub(crate) extern "C" fn __yk_deopt(
    frameaddr: *const c_void,
    gidx: u64,
    jitrbp: *const c_void,
    gp_regs: &[u64; 16],
    fp_regs: &[u64; 16],
) -> ! {
    let gidx = GuardIdx::from(usize::try_from(gidx).unwrap());
    let ctr = MTThread::with(|mtt| mtt.running_trace().unwrap())
        .as_any()
        .downcast::<X64CompiledTrace>()
        .unwrap();
    debug_assert!(usize::from(gidx) < ctr.deoptinfo.len());
    let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
    let info = &ctr.deoptinfo[usize::from(gidx)];
    let mt = Arc::clone(&ctr.mt);

    if let Some(st) = info.guard.ctr() {
        // Prepare the traceinputs "struct" (for now this is just a vector) and pass it into the
        // side-trace.
        let mut ykctrlpvars = Vec::new();
        for (_, jitval) in &info.live_vars {
            let val = match jitval {
                VarLocation::Stack { frame_off, size } => {
                    let p = unsafe { jitrbp.byte_sub(usize::try_from(*frame_off).unwrap()) };
                    match *size {
                        1 => unsafe { u64::from(std::ptr::read::<u8>(p as *const u8)) },
                        2 => unsafe { u64::from(std::ptr::read::<u16>(p as *const u16)) },
                        4 => unsafe { u64::from(std::ptr::read::<u32>(p as *const u32)) },
                        8 => unsafe { std::ptr::read::<u64>(p as *const u64) },
                        _ => todo!(),
                    }
                }
                VarLocation::Register(x) => match x {
                    Register::GP(x) => gp_regs[usize::from(x.code())],
                    Register::FP(_) => todo!(),
                },
                VarLocation::ConstFloat(_) => todo!(),
                VarLocation::ConstInt { bits: _, v } => *v,
                VarLocation::Direct { .. } => panic!(),
                VarLocation::Indirect { .. } => panic!(),
            };
            ykctrlpvars.push(val);
        }

        let f = unsafe {
            mem::transmute::<*const c_void, unsafe extern "C" fn(*mut c_void, *const c_void) -> !>(
                st.entry(),
            )
        };
        let mt = Arc::clone(&ctr.mt);
        drop(ctr);
        mt.stats.timing_state(TimingState::JitExecuting);
        mt.log.log(Verbosity::JITEvent, "execute-side-trace");

        MTThread::with(|mtt| {
            mtt.set_running_trace(Some(st));
        });

        // FIXME: Calling this function overwrites the current (Rust) function frame,
        // rather than unwinding it. https://github.com/ykjit/yk/issues/778
        unsafe { f(ykctrlpvars.as_ptr() as *mut c_void, frameaddr) };
    }
    mt.log.log(Verbosity::JITEvent, "deoptimise");

    // Calculate space required for the new stack.
    // Add space for live register values which we'll be adding at the end.
    let mut memsize = RECOVER_REG.len() * REG64_SIZE;
    // Calculate amount of space we need to allocate for each stack frame.
    for (i, iframe) in info.inlined_frames.iter().enumerate() {
        let (rec, _) = aot_smaps.get(usize::try_from(iframe.safepoint.id).unwrap());
        debug_assert!(rec.size != u64::MAX);
        // The controlpoint frame (i == 0) doesn't need to be recreated.
        if i > 0 {
            // We are on x86_64 so this unwrap is safe.
            memsize += usize::try_from(rec.size).unwrap();
        }
        // Reserve return address space for each frame.
        memsize += REG64_SIZE;
    }

    // Allocate space on the heap for the new stack. We will later memcpy this new stack over the
    // old stack just after the frame containing the control point. Since the stack grows downwards
    // we need to assemble it in the same way. For convenience we will be keeping pointers into
    // the newstack which we aptly call `rsp` and `rbp`.
    let newstack = unsafe { libc::malloc(memsize) };
    let mut rsp = unsafe { newstack.byte_add(memsize) };
    let mut rbp = rsp;
    // Keep track of the real address of the current frame so we can write pushed RBP values.
    let mut lastframeaddr = frameaddr;
    let mut lastframesize = 0;

    // Live register values that we need to write back into AOT registers.
    let mut registers = [0; REGISTER_NUM];
    let mut varidx = 0;
    for (i, iframe) in info.inlined_frames.iter().enumerate() {
        let (rec, pinfo) = aot_smaps.get(usize::try_from(iframe.safepoint.id).unwrap());

        // WRITE RBP
        // If the current frame has pushed RBP we need to do the same (unless we are processing
        // the bottom-most frame).
        if pinfo.hasfp && i > 0 {
            rsp = unsafe { rsp.sub(REG64_SIZE) };
            rbp = rsp;
            unsafe { ptr::write(rsp as *mut u64, lastframeaddr as u64) };
        }

        // Calculate the this frame's address by substracting the last frame's size (plus return
        // address) from the last frame's address.
        if i > 0 {
            lastframeaddr = unsafe { lastframeaddr.byte_sub(lastframesize + REG64_SIZE) };
        }
        lastframesize = usize::try_from(rec.size).unwrap();

        // Update RBP to represent this frame's address.
        if pinfo.hasfp {
            registers[usize::from(RBP_DWARF_NUM)] = lastframeaddr as u64;
        }

        // Now we write any callee-saved registers onto the new stack. Note, that if we have
        // pushed RBP above (which includes adjusting RBP) we need to temporarily re-adjust our
        // pointer. This is because the CSR index calculates from the bottom of the frame, not
        // from RBP. For example, a typical prologue looks like this:
        //   push rbp
        //   mov rbp, rsp
        //   push rbx     # this has index -2
        //   push r14     # this has index -3
        if i > 0 {
            for (reg, idx) in &pinfo.csrs {
                let mut tmp =
                    unsafe { rbp.byte_sub(usize::try_from(idx.abs()).unwrap() * REG64_SIZE) };
                if pinfo.hasfp {
                    tmp = unsafe { tmp.byte_add(REG64_SIZE) };
                }
                let val = registers[usize::from(*reg)];
                unsafe { ptr::write(tmp as *mut u64, val) };
            }
        }

        // Now write all live variables to the new stack in the order they are listed in the AOT
        // stackmap.
        for aotvar in rec.live_vars.iter() {
            // Read live JIT values from the trace's stack frame.
            let jitval = match info.live_vars[varidx].1 {
                VarLocation::Stack { frame_off, size } => {
                    let p = unsafe { jitrbp.byte_sub(usize::try_from(frame_off).unwrap()) };
                    match size {
                        1 => unsafe { u64::from(std::ptr::read::<u8>(p as *const u8)) },
                        2 => unsafe { u64::from(std::ptr::read::<u16>(p as *const u16)) },
                        4 => unsafe { u64::from(std::ptr::read::<u32>(p as *const u32)) },
                        8 => unsafe { std::ptr::read::<u64>(p as *const u64) },
                        _ => todo!(),
                    }
                }
                VarLocation::Register(x) => match x {
                    Register::GP(x) => gp_regs[usize::from(x.code())],
                    Register::FP(x) => fp_regs[usize::from(x.code())],
                },
                VarLocation::ConstInt { bits: _, v } => v,
                VarLocation::ConstFloat(f) => f.to_bits(),
                VarLocation::Direct { .. } => {
                    // See comment below: this case never needs to do anything.
                    varidx += 1;
                    continue;
                }
                VarLocation::Indirect { frame_off, size } => match size {
                    8 => unsafe {
                        (jitrbp as *const *const u64)
                            .read()
                            .byte_offset(isize::try_from(frame_off).unwrap())
                            .read()
                    },
                    4 => unsafe {
                        (jitrbp as *const *const u32)
                            .read()
                            .byte_offset(isize::try_from(frame_off).unwrap())
                            .read() as u64
                    },
                    1 => unsafe {
                        (jitrbp as *const *const u8)
                            .read()
                            .byte_offset(isize::try_from(frame_off).unwrap())
                            .read() as u64
                    },
                    _ => todo!("size={}", size),
                },
            };
            varidx += 1;

            let aotloc = if aotvar.len() == 1 {
                aotvar.get(0).unwrap()
            } else {
                todo!("Deal with multi register locations");
            };
            match aotloc {
                SMLocation::Register(reg, size, off, extra) => {
                    registers[usize::from(*reg)] = jitval;
                    if *extra != 0 {
                        // The stackmap has recorded an additional register we need to write
                        // this value to.
                        registers[usize::from(*extra - 1)] = jitval;
                    }
                    // Check if there's an additional spill location for this value. Negative
                    // values indicate stack offsets, positive values are registers. Lastly, 0
                    // indicates that there's no additional location. Note, that this means
                    // that in order to encode register locations (where RAX = 0), all register
                    // values have been offset by 1.
                    if *off < 0 {
                        let temp = if i == 0 {
                            unsafe { frameaddr.offset(isize::try_from(*off).unwrap()) }
                        } else {
                            unsafe { rbp.offset(isize::try_from(*off).unwrap()) }
                        };
                        debug_assert!(*off < i32::try_from(rec.size).unwrap());
                        match size {
                            // FIXME: Check that 16-byte writes are for float registers only.
                            16 | 8 => unsafe { ptr::write::<u64>(temp as *mut u64, jitval) },
                            4 => unsafe { ptr::write::<u32>(temp as *mut u32, jitval as u32) },
                            _ => todo!("{}", size),
                        }
                    } else if *off > 0 {
                        registers[usize::try_from(*off - 1).unwrap()] = jitval;
                    }
                }
                SMLocation::Direct(..) => {
                    // Direct locations are pointers to the stack, stored on the stack (e.g.
                    // `alloca` or GEP). Our shadow stack unifies the JIT and AOT stacks, replacing
                    // them with a heap allocation. For this reason, no `Direct` stackmap entries
                    // can exist apart from those special-cased in the shadow stack pass (e.g. the
                    // control point struct and the result of `yk_mt_location_new()`). The
                    // exceptions only appear (for now) at frame index 0 (where the control point
                    // is), and since this frame will not be re-written by deopt, there's no need
                    // to restore those direct locations anyway.
                    debug_assert_eq!(i, 0);
                    continue;
                }
                SMLocation::Indirect(reg, off, size) => {
                    debug_assert_eq!(*reg, RBP_DWARF_NUM);
                    let temp = if i == 0 {
                        // While the bottom frame is already on the stack and doesn't need to
                        // be recreated, we still need to copy over new values from the JIT.
                        // Luckily, we know the address of the bottom frame, so we can write
                        // any changes directly to it from here.
                        unsafe { frameaddr.offset(isize::try_from(*off).unwrap()) }
                    } else {
                        unsafe { rbp.offset(isize::try_from(*off).unwrap()) }
                    };
                    debug_assert!(*off < i32::try_from(rec.size).unwrap());
                    match size {
                        1 => unsafe { ptr::write::<u8>(temp as *mut u8, jitval as u8) },
                        4 => unsafe { ptr::write::<u32>(temp as *mut u32, jitval as u32) },
                        8 => unsafe { ptr::write::<u64>(temp as *mut u64, jitval) },
                        _ => todo!(),
                    }
                }
                SMLocation::Constant(_v) => todo!(),
                SMLocation::LargeConstant(_v) => todo!(),
            }
        }

        if i > 0 {
            // Advance the "virtual RSP" to the next frame.
            rsp = unsafe { rbp.byte_sub(usize::try_from(rec.size).unwrap()) };
            if pinfo.hasfp {
                // The stack size recorded by the stackmap includes a pushed RBP. However, we will
                // have already adjusted the "virtual RSP" earlier (when writing RBP) if `hasfp` is
                // true. If that's the case, re-adjust the "virtual RSP" again to account for this.
                rsp = unsafe { rsp.byte_add(REG64_SIZE) };
            }
        }

        // Write the return address for the previous frame into the current frame.
        unsafe {
            rsp = rsp.sub(REG64_SIZE);
            ptr::write(rsp as *mut u64, rec.offset);
        }
    }

    // Write the live registers into the new stack. We put these at the very end of the new stack
    // so that they can be immediately popped after we memcpy'd the new stack over.
    for reg in RECOVER_REG {
        unsafe {
            rsp = rsp.byte_sub(REG64_SIZE);
            ptr::write(rsp as *mut u64, registers[reg]);
        }
    }

    // Compute the address to which we want to write the new stack. This is immediately after the
    // frame containing the control point.
    let (rec, pinfo) = aot_smaps.get(usize::try_from(info.inlined_frames[0].safepoint.id).unwrap());
    let mut newframedst = unsafe { frameaddr.byte_sub(usize::try_from(rec.size).unwrap()) };
    if pinfo.hasfp {
        newframedst = unsafe { newframedst.byte_add(REG64_SIZE) };
    }

    // The `clone` should really be `Arc::clone(&ctr)` but that doesn't play well with type
    // inference in this (unusual) case.
    ctr.mt.guard_failure(ctr.clone(), gidx);

    // Since we won't return from this function, drop `ctr` manually.
    drop(ctr);

    // Now overwrite the existing stack with our newly recreated one.
    unsafe { __replace_stack(newframedst as *mut c_void, newstack, memsize) };
}

#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
unsafe extern "C" fn __replace_stack(dst: *mut c_void, src: *const c_void, size: usize) -> ! {
    std::arch::asm!(
        // Reset RSP to the end of the control point frame (this doesn't include the
        // return address which will thus be overwritten in the process)
        "mov rsp, rdi",
        // Move rsp to the end of the new stack.
        "sub rsp, rdx",
        // Save src ptr into a callee-save reg so we can free it later.
        "mov r12, rsi",
        // Copy the new stack over the old stack.
        "mov rdi, rsp",
        "call memcpy",
        // Restore src ptr.
        "mov rdi, r12",
        // Free the source which is no longer needed.
        "call free",
        // Recover live registers.
        "movsd xmm15, [rsp]",
        "add rsp, 8",
        "movsd xmm14, [rsp]",
        "add rsp, 8",
        "movsd xmm13, [rsp]",
        "add rsp, 8",
        "movsd xmm12, [rsp]",
        "add rsp, 8",
        "movsd xmm11, [rsp]",
        "add rsp, 8",
        "movsd xmm10, [rsp]",
        "add rsp, 8",
        "movsd xmm9, [rsp]",
        "add rsp, 8",
        "movsd xmm8, [rsp]",
        "add rsp, 8",
        "movsd xmm7, [rsp]",
        "add rsp, 8",
        "movsd xmm6, [rsp]",
        "add rsp, 8",
        "movsd xmm5, [rsp]",
        "add rsp, 8",
        "movsd xmm4, [rsp]",
        "add rsp, 8",
        "movsd xmm3, [rsp]",
        "add rsp, 8",
        "movsd xmm2, [rsp]",
        "add rsp, 8",
        "movsd xmm1, [rsp]",
        "add rsp, 8",
        "movsd xmm0, [rsp]",
        "add rsp, 8",
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
        "ret",
        options(noreturn)
    )
}
