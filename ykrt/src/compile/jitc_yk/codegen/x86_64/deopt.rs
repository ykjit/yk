use crate::{
    aotsmp::AOT_STACKMAPS, compile::jitc_yk::codegen::reg_alloc::LocalAlloc, log::log_jit_state,
    mt::MTThread,
};
use libc::c_void;
use std::ptr;
use yksmp::Location as SMLocation;

use super::{X64CompiledTrace, RBP_DWARF_NUM, REG64_SIZE};

#[no_mangle]
pub(crate) extern "C" fn __yk_deopt(
    frameaddr: *const c_void,
    deoptid: usize,
    jitrbp: *const c_void,
) -> ! {
    log_jit_state("deoptimise");

    let ctr = MTThread::with(|mtt| mtt.running_trace().unwrap())
        .as_any()
        .downcast::<X64CompiledTrace>()
        .unwrap();
    debug_assert!(deoptid < ctr.deoptinfo.len());
    let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
    let info = &ctr.deoptinfo[deoptid];

    // Calculate space required for the new stack.
    // Add space for live register values which we'll be adding at the end.
    let mut memsize = 15 * REG64_SIZE;
    // Calculate amount of space we need to allocate for each stack frame.
    for (frameid, smid) in info.frames.iter().enumerate() {
        let (rec, _) = aot_smaps.get(usize::try_from(*smid).unwrap());
        debug_assert!(rec.size != u64::MAX);
        if frameid > 0 {
            // The controlpoint frame doesn't need to be recreated.
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
    let mut registers = [0; 16];
    let mut varidx = 0;
    for (frameid, smid) in info.frames.iter().enumerate() {
        let (rec, pinfo) = aot_smaps.get(usize::try_from(*smid).unwrap());

        // WRITE RBP
        // If the current frame has pushed RBP we need to do the same (unless we are processing
        // the bottom-most frame).
        if pinfo.hasfp && frameid > 0 {
            rsp = unsafe { rsp.sub(REG64_SIZE) };
            rbp = rsp;
            unsafe { ptr::write(rsp as *mut u64, lastframeaddr as u64) };
        }

        // Calculate the this frame's address by substracting the last frame's size (plus return
        // address) from the last frame's address.
        if frameid > 0 {
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
        if frameid > 0 {
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
            let jitval = match info.lives[varidx] {
                LocalAlloc::Stack { frame_off, size } => {
                    let p = unsafe { jitrbp.byte_sub(frame_off) };
                    match size {
                        1 => unsafe { u64::from(std::ptr::read::<u8>(p as *const u8)) },
                        2 => unsafe { u64::from(std::ptr::read::<u16>(p as *const u16)) },
                        4 => unsafe { u64::from(std::ptr::read::<u32>(p as *const u32)) },
                        8 => unsafe { std::ptr::read::<u64>(p as *const u64) },
                        _ => todo!(),
                    }
                }
                LocalAlloc::Register => todo!(),
                LocalAlloc::ConstInt(c) => c,
                LocalAlloc::ConstFloat(f) => f.to_bits(),
            };
            varidx += 1;

            let aotloc = if aotvar.len() == 1 {
                aotvar.get(0).unwrap()
            } else {
                todo!("Deal with multi register locations");
            };
            match aotloc {
                SMLocation::Register(reg, _size, off, extra) => {
                    registers[usize::from(*reg)] = jitval;
                    if *extra != 0 {
                        // The stackmap has recorded an additional register we need to write
                        // this value to.
                        registers[usize::from(*extra - 1)] = jitval;
                    }
                    if frameid == 0 {
                        // skip first frame
                        continue;
                    }
                    // Check if there's an additional spill location for this value. Negative
                    // values indicate stack offsets, positive values are registers. Lastly, 0
                    // indicates that there's no additional location. Note, that this means
                    // that in order to encode register locations (where RAX = 0), all register
                    // values have been offset by 1.
                    if *off < 0 {
                        let temp = unsafe { rbp.offset(isize::try_from(*off).unwrap()) };
                        debug_assert!(*off < i32::try_from(rec.size).unwrap());
                        unsafe { ptr::write::<u64>(temp as *mut u64, jitval) };
                    } else if *off > 0 {
                        registers[usize::try_from(*off - 1).unwrap()] = jitval;
                    }
                }
                SMLocation::Direct(..) => {
                    // Due to the shadow stack we only expect direct locations for the control
                    // point frame.
                    debug_assert_eq!(frameid, 0);
                    continue;
                }
                SMLocation::Indirect(reg, off, size) => {
                    debug_assert_eq!(*reg, RBP_DWARF_NUM);
                    let temp = if frameid == 0 {
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

        if frameid > 0 {
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
    for reg in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] {
        unsafe {
            rsp = rsp.byte_sub(REG64_SIZE);
            ptr::write(rsp as *mut u64, registers[reg]);
        }
    }

    // Compute the address to which we want to write the new stack. This is immediately after the
    // frame containing the control point.
    let (rec, pinfo) = aot_smaps.get(usize::try_from(info.frames[0]).unwrap());
    let mut newframedst = unsafe { frameaddr.byte_sub(usize::try_from(rec.size).unwrap()) };
    if pinfo.hasfp {
        newframedst = unsafe { newframedst.byte_add(REG64_SIZE) };
    }

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
        // Copy the new stack over the old stack.
        "mov rdi, rsp",
        "call memcpy",
        // Free the source which is no longer needed.
        "mov rdi, rsi",
        "call free",
        // Recover live registers.
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
