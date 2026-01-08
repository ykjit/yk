//! x64 deoptimisation.

use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        CompiledTrace, GuardId,
        j2::{
            compiled_trace::J2CompiledTrace,
            hir::{ConstKind, GuardExtraIdx},
            regalloc::{RegFill, VarLoc, VarLocs},
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
enum DeoptFpReg {
    XMM0 = 0,
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

impl DeoptFpReg {
    fn idx(&self) -> usize {
        usize::from(*self as u8)
    }
}

impl TryFrom<Reg> for DeoptFpReg {
    type Error = ();

    fn try_from(reg: Reg) -> Result<Self, Self::Error> {
        match reg {
            Reg::XMM0 => Ok(Self::XMM0),
            Reg::XMM1 => Ok(Self::XMM1),
            Reg::XMM2 => Ok(Self::XMM2),
            Reg::XMM3 => Ok(Self::XMM3),
            Reg::XMM4 => Ok(Self::XMM4),
            Reg::XMM5 => Ok(Self::XMM5),
            Reg::XMM6 => Ok(Self::XMM6),
            Reg::XMM7 => Ok(Self::XMM7),
            Reg::XMM8 => Ok(Self::XMM8),
            Reg::XMM9 => Ok(Self::XMM9),
            Reg::XMM10 => Ok(Self::XMM10),
            Reg::XMM11 => Ok(Self::XMM11),
            Reg::XMM12 => Ok(Self::XMM12),
            Reg::XMM13 => Ok(Self::XMM13),
            Reg::XMM14 => Ok(Self::XMM14),
            Reg::XMM15 => Ok(Self::XMM15),
            _ => Err(()),
        }
    }
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
    let gridx = GuardExtraIdx::from(usize::from(gid));
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
    // The values we will (eventually...) put into registers.
    let mut gp_regs = [0; DeoptGpReg::COUNT];
    let mut fp_regs = [0; DeoptFpReg::COUNT];

    // The size of the buffer we'll need. To make the "is there enough space left?" calculation
    // below easier, we account for the size of the FP & GP registers here.
    let mut deoptlen = DeoptGpReg::COUNT * 8 + DeoptFpReg::COUNT * 8;

    // Deal with the control point frame: this is special because we are adjusting a frame that
    // already exists, rather than creating one from scratch.
    {
        let frame = &deopt_frames[0];
        let (smap, prologue) = aot_smaps.get(usize::try_from(frame.pc_safepoint.id).unwrap());
        if prologue.hasfp {
            // Update RBP to represent this frame's address.
            gp_regs[DeoptGpReg::RBP.idx()] = prev_faddr as u64;
        }
        reconstruct(&frame.vars, &mut gp_regs, &mut fp_regs, faddr, faddr);
        deoptlen += 8;
        rsp = unsafe { rsp.sub(8) };
        unsafe { (rsp as *mut u64).write(smap.offset) }
        prev_flen = usize::try_from(smap.size).unwrap();
    }

    // Deal with all the remaining frames: we create each of these from scratch.
    for frame in deopt_frames.iter().skip(1) {
        let (smap, prologue) = aot_smaps.get(usize::try_from(frame.pc_safepoint.id).unwrap());

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

        reconstruct(&frame.vars, &mut gp_regs, &mut fp_regs, faddr, rbp);

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
    for v in fp_regs.iter() {
        rsp = unsafe { rsp.sub(8) };
        unsafe { (rsp as *mut u64).write(*v) }
    }

    let fdst = {
        let (smap, prologue) =
            aot_smaps.get(usize::try_from(deopt_frames[0].pc_safepoint.id).unwrap());
        if prologue.hasfp {
            unsafe { faddr.byte_sub(usize::try_from(smap.size - 8).unwrap()) }
        } else {
            todo!()
        }
    };

    mt.guard_failure(ctr, gid, faddr as *mut c_void);

    // In an ideal world, the following assertion would go inside `replace_stack`, but as a naked
    // function, it can only contain a single `asm` statement.
    //
    // Ensure that we're pushing a 16-byte aligned value to the stack so that the `memcpy` call
    // works as expected. Note: we will pop an 8 -- but not 16! -- byte aligned amount of data from
    // the stack, and `ret` will pop another 8 bytes, ensuring that we continue execution with a
    // 16-byte aligned stack.
    assert_eq!(deoptlen % 16, 0);
    unsafe { __yk_j2_replace_stack(fdst, stkptr.byte_add(stklen).byte_sub(deoptlen), deoptlen) };
}

/// Reconstruct the stack for a frame, reading from the control point frame `cpfaddr` and writing
/// to `toaddr`. Note: these two addresses can be the same.
fn reconstruct(
    varlocs: &[(InstId, u32, VarLocs<Reg>, VarLocs<Reg>)],
    gp_regs: &mut [u64; DeoptGpReg::COUNT],
    fp_regs: &mut [u64; DeoptFpReg::COUNT],
    srcaddr: *const u8,
    tgtaddr: *mut u8,
) {
    for (_, bitw, fromvlocs, tovlocs) in varlocs
        .iter()
        .filter(|(_, _, _, tovlocs)| !tovlocs.is_empty())
    {
        // FIXME: For now, we only deal with 1 fromvloc.
        assert_eq!(fromvlocs.len(), 1, "{fromvlocs:?}");
        let fromvloc = fromvlocs.iter().next().unwrap();
        match bitw {
            33..=64 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        (srcaddr.byte_sub(usize::try_from(*off).unwrap()) as *const u64).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_, _) => todo!(),
                    VarLoc::Const(kind) => match kind {
                        ConstKind::Double(_) => todo!(),
                        ConstKind::Float(_) => todo!(),
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
                        VarLoc::Reg(reg, fill) => {
                            assert_eq!(*fill, RegFill::Zeroed);
                            if reg.is_gp() {
                                gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = v;
                            } else {
                                assert!(reg.is_fp());
                                fp_regs[DeoptFpReg::try_from(*reg).unwrap().idx()] = v;
                            }
                        }
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
                    VarLoc::Reg(_, _) => todo!(),
                    VarLoc::Const(kind) => match kind {
                        ConstKind::Double(_) => unreachable!(),
                        ConstKind::Float(_) => unreachable!(),
                        ConstKind::Int(x) => x.to_zero_ext_u32().unwrap(),
                        ConstKind::Ptr(_) => unreachable!(),
                    },
                };

                for vloc in tovlocs.iter() {
                    match vloc {
                        VarLoc::Stack(off) => unsafe {
                            (tgtaddr.byte_sub(usize::try_from(*off).unwrap()) as *mut u32).write(v);
                        },
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg, fill) => {
                            assert_eq!(*fill, RegFill::Zeroed);
                            if reg.is_gp() {
                                gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = u64::from(v);
                            } else {
                                assert!(reg.is_fp());
                                fp_regs[DeoptFpReg::try_from(*reg).unwrap().idx()] = u64::from(v);
                            }
                        }
                        VarLoc::Const(_) => (),
                    }
                }
            }
            9..=16 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        (srcaddr.byte_sub(usize::try_from(*off).unwrap()) as *const u16).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_, _) => todo!(),
                    VarLoc::Const(ConstKind::Double(_)) => unreachable!(),
                    VarLoc::Const(ConstKind::Float(_)) => unreachable!(),
                    VarLoc::Const(ConstKind::Int(x)) => x.to_zero_ext_u16().unwrap(),
                    VarLoc::Const(ConstKind::Ptr(_)) => unreachable!(),
                };

                for vloc in tovlocs.iter() {
                    match vloc {
                        VarLoc::Stack(off) => unsafe {
                            // FIXME: We don't know if we're overwriting a value deopt
                            // needs to later read!
                            (tgtaddr.byte_sub(usize::try_from(*off).unwrap()) as *mut u16).write(v);
                        },
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg, fill) => {
                            assert!(reg.is_gp());
                            assert_eq!(*fill, RegFill::Zeroed);
                            gp_regs[DeoptGpReg::try_from(*reg).unwrap().idx()] = u64::from(v)
                        }
                        VarLoc::Const(_const_kind) => todo!(),
                    }
                }
            }
            1..=8 => {
                let v = match fromvloc {
                    VarLoc::Stack(off) => unsafe {
                        srcaddr.byte_sub(usize::try_from(*off).unwrap()).read()
                    },
                    VarLoc::StackOff(_) => todo!(),
                    VarLoc::Reg(_, _) => todo!(),
                    VarLoc::Const(ConstKind::Double(_)) => unreachable!(),
                    VarLoc::Const(ConstKind::Float(_)) => unreachable!(),
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
                        VarLoc::Reg(reg, fill) => {
                            assert!(reg.is_gp());
                            assert_eq!(*fill, RegFill::Zeroed);
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
        "mov rax, [rsp+240]",
        "mov rcx, [rsp+232]",
        "mov rdx, [rsp+224]",
        "mov rbx, [rsp+216]",
        "mov rbp, [rsp+208]",
        "mov rsi, [rsp+200]",
        "mov rdi, [rsp+192]",
        "mov r8, [rsp+184]",
        "mov r9, [rsp+176]",
        "mov r10, [rsp+168]",
        "mov r11, [rsp+160]",
        "mov r12, [rsp+152]",
        "mov r13, [rsp+144]",
        "mov r14, [rsp+136]",
        "mov r15, [rsp+128]",
        "movsd xmm0, [rsp+120]",
        "movsd xmm1, [rsp+112]",
        "movsd xmm2, [rsp+104]",
        "movsd xmm3, [rsp+96]",
        "movsd xmm4, [rsp+88]",
        "movsd xmm5, [rsp+80]",
        "movsd xmm6, [rsp+72]",
        "movsd xmm7, [rsp+64]",
        "movsd xmm8, [rsp+56]",
        "movsd xmm9, [rsp+48]",
        "movsd xmm10, [rsp+40]",
        "movsd xmm11, [rsp+32]",
        "movsd xmm12, [rsp+24]",
        "movsd xmm13, [rsp+16]",
        "movsd xmm14, [rsp+8]",
        "movsd xmm15, [rsp]",
        "add rsp, 248",
        "ret",
    )
}
