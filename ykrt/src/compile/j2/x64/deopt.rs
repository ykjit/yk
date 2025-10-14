//! x64 deoptimisation.

use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        CompiledTrace, GuardId,
        j2::{
            compiled_trace::J2CompiledTrace,
            hir::{ConstKind, GuardRestoreIdx},
            regalloc::{VarLoc, VarLocs},
            x64::x64regalloc::Reg,
        },
        jitc_yk::aot_ir::InstId,
    },
    log::Verbosity,
    mt::{MTThread, TraceId},
};
use std::{
    alloc::{Layout, alloc},
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};
use strum::{EnumCount, FromRepr};

/// The order of these is relied upon by the code which pops registers from the stack.
#[derive(Clone, Copy, Debug, EnumCount, FromRepr, PartialEq)]
#[repr(u8)]
enum DeoptGpReg {
    RAX = 0,
    RCX,
    RDX,
    RBX,
    RBP,
    RSI,
    RDI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl DeoptGpReg {
    fn idx(&self) -> usize {
        usize::from(*self as u8)
    }
}

impl TryFrom<Reg> for DeoptGpReg {
    type Error = ();

    fn try_from(reg: Reg) -> Result<Self, Self::Error> {
        match reg {
            Reg::RAX => Ok(DeoptGpReg::RAX),
            Reg::RCX => Ok(DeoptGpReg::RCX),
            Reg::RDX => Ok(DeoptGpReg::RDX),
            Reg::RBX => Ok(DeoptGpReg::RBX),
            Reg::RSI => Ok(DeoptGpReg::RSI),
            Reg::RDI => Ok(DeoptGpReg::RDI),
            Reg::R8 => Ok(DeoptGpReg::R8),
            Reg::R9 => Ok(DeoptGpReg::R9),
            Reg::R10 => Ok(DeoptGpReg::R10),
            Reg::R11 => Ok(DeoptGpReg::R11),
            Reg::R12 => Ok(DeoptGpReg::R12),
            Reg::R13 => Ok(DeoptGpReg::R13),
            Reg::R14 => Ok(DeoptGpReg::R14),
            Reg::R15 => Ok(DeoptGpReg::R15),
            _ => Err(()),
        }
    }
}

/// The order of these is relied upon by the code which pops registers from the stack.
#[derive(Clone, Copy, Debug, EnumCount, FromRepr, PartialEq)]
#[repr(u8)]
#[allow(dead_code)]
enum DeoptFpReg {
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
}

thread_local! {
    // This caches the memory we use to generate the "new stack" that deopt has to create.
    static BUF: (AtomicPtr<u8>, AtomicUsize) = (
        AtomicPtr::new(unsafe { alloc(Layout::from_size_align(page_size::get(), page_size::get()).unwrap()) }),
        AtomicUsize::new(page_size::get())
    );
}

#[unsafe(no_mangle)]
pub(super) extern "C" fn __yk_j2_deopt(faddr: *mut u8, trid: u64, gid: u32) -> ! {
    let gid = GuardId::from(usize::try_from(gid).unwrap());
    let gridx = GuardRestoreIdx::from(usize::from(gid));
    let ctr = MTThread::with_borrow(|mtt| mtt.compiled_trace(TraceId::from_u64(trid)))
        .as_any()
        .downcast::<J2CompiledTrace<Reg>>()
        .unwrap();
    let mt = Arc::clone(&ctr.mt);
    mt.stats
        .timing_state(crate::log::stats::TimingState::Deopting);

    mt.log.log(
        Verbosity::Execution,
        &format!("deoptimise {:?} {gid:?}", ctr.ctrid()),
    );

    mt.deopt();

    let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
    let deopt_frames = &ctr.deopt_frames(gridx);

    // We write to the buffer backwards, starting at `stkptr + stklen`. Eventually the contents of
    // this buffer (viewed from low to high address) will be:
    //
    //   ..., , FP regs, GP regs, deopt_frames[n], ..., deopt_frames[1]
    //
    // where:
    //   * The initial "..." represents the bytes we won't write to in this deopt function.
    //   * `deopt_frames[0]` is special: it's the control point frame and we both read and write to
    //     it directly. Thus our buffer doesn't contain `deopt_frames[0]`: it ends at
    //     `deopt_frames[1]`.
    //   * "FP regs" and "GP regs" are the values we eventually want stuffed back into the CPU's
    //     registers: they'll be read by the `replace_stack` function.
    let (stkptr, stklen) =
        BUF.with(|(ptr, sz)| (ptr.load(Ordering::Relaxed), sz.load(Ordering::Relaxed)));
    let mut rbp = unsafe { stkptr.byte_add(stklen) };
    let mut rsp = rbp;

    // The address of the previous frame on the stack.
    let mut prev_faddr = faddr;
    // The length of the previous frame on the stack.
    let mut prev_flen;
    // The values we will (eventually...) put into general purpose registers.
    let mut gp_regs = [0; DeoptGpReg::COUNT];

    // The size of the buffer we'll need. To make the "is there enough space left?" calculation
    // below easier, we account for the size of the FP & GP registers here.
    let mut deoptlen = DeoptGpReg::COUNT * 8;

    // Deal with the control point frame: this is special because we are adjusting a frame that
    // already exists, rather than creating one from scratch.
    {
        let frame = &deopt_frames[0];
        let (smap, prologue) = aot_smaps.get(usize::try_from(frame.safepoint.id).unwrap());
        if prologue.hasfp {
            // Update RBP to represent this frame's address.
            gp_regs[DeoptGpReg::RBP.idx()] = prev_faddr as u64;
        }
        reconstruct(&frame.vars, &mut gp_regs, faddr, faddr);
        deoptlen += 8;
        rsp = unsafe { rsp.sub(8) };
        unsafe { (rsp as *mut u64).write(smap.offset) }
        prev_flen = usize::try_from(smap.size).unwrap();
    }

    // Deal with all the remaining frames: we create each of these from scratch.
    for frame in deopt_frames.iter().skip(1) {
        let (smap, prologue) = aot_smaps.get(usize::try_from(frame.safepoint.id).unwrap());

        // How much room does this frame need?
        let flen = 8 + usize::try_from(smap.size).unwrap();

        // Do we have enough space left to write this frame to the buffer? The more obvious
        // comparison here might seem to be `if rbp - flen < stkptr` but in the case where our
        // buffer is too small `rbp - flen` would create an out-of-bounds pointer and land us in
        // undefined behaviour.
        assert!(rbp > stkptr);
        if flen > unsafe { rbp.byte_offset_from_unsigned(stkptr) } {
            todo!();
        }
        deoptlen += flen;

        if prologue.hasfp {
            rsp = unsafe { rsp.byte_sub(8) };
            rbp = rsp;
            unsafe {
                (rsp as *mut *mut u8).write(prev_faddr);
            }
        }
        // Calculate the this frame's address by subtracting the last frame's size (plus
        // return address) from the last frame's address.
        prev_faddr = unsafe { prev_faddr.byte_sub(prev_flen + 8) };

        if prologue.hasfp {
            // Update RBP to represent this frame's address.
            gp_regs[DeoptGpReg::RBP.idx()] = prev_faddr as u64;
        }

        // Now we write any callee-saved registers onto the new stack. Note, that if we have
        // pushed RBP above (which includes adjusting RBP) we need to temporarily re-adjust our
        // pointer. This is because the CSR index calculates from the bottom of the frame, not
        // from RBP. For example, a typical prologue looks like this:
        //   push rbp
        //   mov rbp, rsp
        //   push rbx     # this has index -2
        //   push r14     # this has index -3
        for (dwarf_reg, idx) in &prologue.csrs {
            assert!(*idx < 0);
            let mut off = usize::try_from(idx.abs()).unwrap() * 8;
            if prologue.hasfp {
                off -= 8;
            }
            let reg = Reg::from_dwarf_reg(*dwarf_reg);
            if reg.is_gp() {
                let deoptreg = DeoptGpReg::try_from(reg).unwrap();
                let v = gp_regs[deoptreg.idx()];
                unsafe {
                    (rbp as *mut u64).byte_sub(off).write(v);
                }
            } else {
                todo!();
            }
        }

        reconstruct(&frame.vars, &mut gp_regs, faddr, rbp);

        // Advance RSP
        rsp = unsafe { rbp.byte_sub(usize::try_from(smap.size).unwrap()) };
        if prologue.hasfp {
            rsp = unsafe { rsp.byte_add(8) };
        }

        rsp = unsafe { rsp.sub(8) };
        unsafe { (rsp as *mut u64).write(smap.offset) }
        prev_flen = usize::try_from(smap.size).unwrap();
    }

    // Push all the registers we'll need to unpack in `replace_stack`.
    for v in gp_regs.iter() {
        rsp = unsafe { rsp.sub(8) };
        unsafe { (rsp as *mut u64).write(*v) }
    }

    let fdst = {
        let (smap, prologue) =
            aot_smaps.get(usize::try_from(deopt_frames[0].safepoint.id).unwrap());
        if prologue.hasfp {
            unsafe { faddr.byte_sub(usize::try_from(smap.size - 8).unwrap()) }
        } else {
            todo!()
        }
    };

    mt.guard_failure(ctr, gid, faddr as *mut c_void);

    unsafe { __yk_j2_replace_stack(fdst, stkptr.byte_add(stklen).byte_sub(deoptlen), deoptlen) };
}

/// Reconstruct the stack for a frame, reading from the control point frame `cpfaddr` and writing
/// to `toaddr`. Note: these two addresses can be the same.
fn reconstruct(
    varlocs: &[(InstId, u32, VarLocs<Reg>, VarLocs<Reg>)],
    gp_regs: &mut [u64; DeoptGpReg::COUNT],
    srcaddr: *const u8,
    tgtaddr: *mut u8,
) {
    for (_, bitw, fromvlocs, tovlocs) in varlocs
        .iter()
        .filter(|(_, _, _, tovlocs)| !tovlocs.is_empty())
    {
        // FIXME: For now, we only deal with 1 fromvloc.
        assert_eq!(fromvlocs.len(), 1);
        let fromvloc = fromvlocs.iter().next().unwrap();
        match bitw {
            33..=64 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        (srcaddr.byte_sub(usize::try_from(*off).unwrap()) as *const u64).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_) => todo!(),
                    VarLoc::Const(kind) => match kind {
                        ConstKind::Int(x) => x.to_zero_ext_u64().unwrap(),
                        ConstKind::Ptr(x) => u64::try_from(*x).unwrap(),
                    },
                };

                for vloc in tovlocs.iter() {
                    match vloc {
                        VarLoc::Stack(off) => unsafe {
                            (tgtaddr.byte_sub(usize::try_from(*off).unwrap()) as *mut u64).write(v);
                        },
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg) => gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = v,
                        VarLoc::Const(_) => (),
                    }
                }
            }
            17..=32 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        (srcaddr.byte_sub(usize::try_from(*off).unwrap()) as *const u32).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_) => todo!(),
                    VarLoc::Const(kind) => match kind {
                        ConstKind::Int(x) => x.to_zero_ext_u32().unwrap(),
                        ConstKind::Ptr(_) => todo!(),
                    },
                };

                for vloc in tovlocs.iter() {
                    match vloc {
                        VarLoc::Stack(off) => unsafe {
                            (tgtaddr.byte_sub(usize::try_from(*off).unwrap()) as *mut u32).write(v);
                        },
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg) => {
                            gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = u64::from(v)
                        }
                        VarLoc::Const(_) => (),
                    }
                }
            }
            1..=8 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        srcaddr.byte_sub(usize::try_from(*off).unwrap()).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_) => todo!(),
                    VarLoc::Const(ConstKind::Int(x)) => x.to_zero_ext_u8().unwrap(),
                    VarLoc::Const(ConstKind::Ptr(_)) => unreachable!(),
                };

                for vloc in tovlocs.iter() {
                    match vloc {
                        VarLoc::Stack(off) => unsafe {
                            // FIXME: We don't know if we're overwriting a value deopt
                            // needs to later read!
                            tgtaddr.byte_sub(usize::try_from(*off).unwrap()).write(v);
                        },
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg) => {
                            gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = u64::from(v)
                        }
                        VarLoc::Const(_const_kind) => todo!(),
                    }
                }
            }
            x => todo!("{x}"),
        }
    }
}

/// Writes the stack frames that we recreated in [__yk_deopt] onto the current stack, overwriting
/// the stack frames of any running traces in the process. This deoptimises trace execution after
/// which we can safely return to the normal execution of the interpreter.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn __yk_j2_replace_stack(dst: *mut u8, src: *const u8, len: usize) -> ! {
    std::arch::naked_asm!(
        // Reset RSP to the end of the control point frame (this doesn't include the
        // return address which will thus be overwritten in the process)
        "mov rsp, rdi",
        // Move rsp to the end of the new stack.
        "sub rsp, rdx",
        // Copy the new stack over the old stack.
        "mov rdi, rsp",
        "call memcpy",
        // Recover live registers.
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop r11",
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rbp",
        "pop rbx",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "ret",
    )
}
