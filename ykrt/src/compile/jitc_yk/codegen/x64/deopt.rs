use super::{Register, VarLocation};
use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{CompiledTrace, GuardId},
    log::Verbosity,
    mt::{MTThread, TraceId},
};
use dynasmrt::Register as _;
use libc::c_void;
use page_size;
#[cfg(debug_assertions)]
use std::collections::HashMap;
#[cfg(debug_assertions)]
use std::ops::Range;
use std::{
    alloc::{Layout, alloc, realloc},
    ptr,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};
use yksmp::Location as SMLocation;

use super::{RBP_DWARF_NUM, REG64_BYTESIZE, X64CompiledTrace};

thread_local! {
    // This caches the memory we use to generate the "new stack" that deopt has to create.
    static BUF: (AtomicPtr<u8>, AtomicUsize) = (
        AtomicPtr::new(unsafe { alloc(Layout::from_size_align(page_size::get(), page_size::get()).unwrap()) }),
        AtomicUsize::new(page_size::get())
    );
}

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

#[unsafe(no_mangle)]
pub(crate) extern "C" fn __yk_ret_from_trace(ctrid: u64) {
    let ctr = MTThread::with_borrow(|mtt| mtt.compiled_trace(TraceId::from_u64(ctrid)))
        .as_any()
        .downcast::<X64CompiledTrace>()
        .unwrap();
    let mt = &ctr.mt;
    mt.deopt();
    mt.stats
        .timing_state(crate::log::stats::TimingState::OutsideYk);
    mt.log
        .log(Verbosity::Execution, &format!("return {:?}", ctr.ctrid()));
}

/// Determine if two ranges overlap.
///
/// (At the time of writing, Rust's `is_overlapping()` is only implemented on `Range<usize>`
/// and we use `Range<isize>` in deopt).
#[cfg(debug_assertions)]
fn ranges_overlap(r1: &Range<isize>, r2: &Range<isize>) -> bool {
    r1.start.max(r2.start) < r1.end.min(r2.end)
}

/// Check for obvious deopt issues.
///
/// For example:
///  - Deopting the same register twice with different values.
///  - Deopt overlapping memory regions.
#[cfg(debug_assertions)]
#[derive(Default)]
struct DeoptChecker {
    /// Register locations we've seen deopted before.
    ///
    /// The key is a DWARF register number. The value is a tuple of the size of the value (in
    /// bytes) and the actual value we deopted.
    seen_reg_locs: HashMap<usize, (usize, u64)>,
    /// Base pointer-relative stack locations we've seen deopted before.
    ///
    /// Each element is a tuple capturing the memory range affected and a value that the range was
    /// deopted to.
    seen_mem_locs: Vec<(Range<isize>, u64)>,
}

#[cfg(debug_assertions)]
impl DeoptChecker {
    /// Records (and checks) a deopted location of a location `loc` with the value `val`.
    ///
    /// The encoding of `loc`:
    ///  - loc >= 0: DWARF register number.
    ///  - loc < 0: stack offset relative to rbp
    ///
    /// (note that a memory deopt of `basepointer-0` is never deopted and thus isn't necessary for
    /// us to express, thus it's safe to assume that `loc==0` means register zero)
    ///
    /// # Panics
    ///
    /// If anything looks fishy.
    fn record_and_check(&mut self, loc: isize, size: usize, val: u64) {
        // Note: It's OK to deopt the same value to the same location.
        if loc >= 0 {
            // register location.
            let loc = usize::try_from(loc).unwrap();
            if let std::collections::hash_map::Entry::Vacant(e) = self.seen_reg_locs.entry(loc) {
                e.insert((size, val));
            } else {
                let saw = self.seen_reg_locs[&loc];
                if saw.1 != val {
                    panic!(
                        "Overlapping register deopt for DWARF reg {}! \
                        {} bytes 0x{:016x} vs {} bytes 0x{:016x}",
                        loc, saw.0, saw.1, size, val
                    );
                };
            }
        } else {
            // mem location relative to the base pointer.
            let new_rng = loc..(loc + isize::try_from(size).unwrap());
            for (seen_rng, seen_val) in &self.seen_mem_locs {
                if ranges_overlap(seen_rng, &new_rng) {
                    // If the lower bound of the deopted memory region is the same *and* the value
                    // written is identical, then this isn't necessarily a problem on a big-endian
                    // system. This is because the least significant bytes will be written first in
                    // memory.
                    //
                    // Note that this isn't sufficient to catch *all* safe overlapping memory
                    // deopts, but it's a start, and I'd like to know what other cases can arise.
                    if seen_rng.start != new_rng.start || *seen_val != val {
                        panic!(
                            "Suspicious memory deopt! \
                        range: {seen_rng:?}, val: 0x{seen_val:016x} vs. range {new_rng:?}, val: 0x{val:016x}"
                        );
                    }
                }
            }
            self.seen_mem_locs.push((new_rng, val));
        }
    }
}

/// Deoptimise back to the interpreter. This function is called from a failing guard (see
/// [super::Assemble::codegen]).
///
/// # Arguments
///
/// * `frameaddr` - the RBP value for main interpreter loop (and also the JIT since the trace
///   executes on the same frame)
/// * `gid` - the [GuardId] of the current failing guard
/// * `gp_regs` - a pointer to the saved values of the 16 general purpose registers in the same
///   order as [crate::compile::jitc_yk::codegen::x64::lsregalloc::GP_REGS]
/// * `fp_regs` - a pointer to the saved values of the 16 floating point registers
/// * `ctrid` - the ID of the compiled trace that is being deoptimized
#[unsafe(no_mangle)]
pub(crate) extern "C" fn __yk_deopt(
    frameaddr: *mut c_void,
    gid: u64,
    gp_regs: &[u64; 16],
    fp_regs: &[u64; 16],
    ctrid: u64,
) -> *const libc::c_void {
    let ctr = MTThread::with_borrow(|mtt| mtt.compiled_trace(TraceId::from_u64(ctrid)))
        .as_any()
        .downcast::<X64CompiledTrace>()
        .unwrap();
    let mt = Arc::clone(&ctr.mt);
    mt.stats
        .timing_state(crate::log::stats::TimingState::Deopting);
    let gid = GuardId::from(usize::try_from(gid).unwrap());
    let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
    let cgd = &ctr.compiled_guard(gid);

    mt.deopt();
    mt.log.log(
        Verbosity::Execution,
        &format!("deoptimise {:?} {gid:?}", ctr.ctrid()),
    );

    // Calculate space required for the new stack.
    // Add space for live register values which we'll be adding at the end.
    let mut memsize = RECOVER_REG.len() * REG64_BYTESIZE;
    // Calculate amount of space we need to allocate for each stack frame.
    for (i, iframe) in cgd.inlined_frames.iter().enumerate() {
        let (rec, _) = aot_smaps.get(usize::try_from(iframe.safepoint.id).unwrap());
        debug_assert!(rec.size != u64::MAX);
        // The controlpoint frame (i == 0) doesn't need to be recreated.
        if i > 0 {
            // We are on x86_64 so this unwrap is safe.
            memsize += usize::try_from(rec.size).unwrap();
        }
        // Reserve return address space for each frame.
        memsize += REG64_BYTESIZE;
    }

    // Ensure we've got enough space to copy the new stack over. For convenience we will be keeping
    // pointers into the newstack which we aptly call `rsp` and `rbp`.
    let newstack = BUF.with(|(ptr, sz)| {
        if memsize < sz.load(Ordering::Relaxed) {
            ptr.load(Ordering::Relaxed)
        } else {
            let ol = Layout::from_size_align(sz.load(Ordering::Relaxed), page_size::get()).unwrap();
            let newsize = memsize.next_multiple_of(page_size::get());
            let n = unsafe { realloc(ptr.load(Ordering::Relaxed), ol, newsize) };
            assert!(!n.is_null());
            ptr.store(n, Ordering::Relaxed);
            sz.store(newsize, Ordering::Relaxed);
            n
        }
    }) as *mut c_void;
    let mut rsp = unsafe { newstack.byte_add(memsize) };
    let mut rbp = rsp;
    // Keep track of the real address of the current frame so we can write pushed RBP values.
    let mut lastframeaddr = frameaddr;
    let mut lastframesize = 0;

    // Live register values that we need to write back into AOT registers.
    let mut registers = [0; REGISTER_NUM];
    let mut varidx = 0;
    for (i, iframe) in cgd.inlined_frames.iter().enumerate() {
        let (rec, pinfo) = aot_smaps.get(usize::try_from(iframe.safepoint.id).unwrap());

        // WRITE RBP
        // If the current frame has pushed RBP we need to do the same (unless we are processing
        // the bottom-most frame).
        if pinfo.hasfp && i > 0 {
            rsp = unsafe { rsp.sub(REG64_BYTESIZE) };
            rbp = rsp;
            unsafe { ptr::write(rsp as *mut u64, lastframeaddr as u64) };
        }

        // Calculate the this frame's address by substracting the last frame's size (plus return
        // address) from the last frame's address.
        if i > 0 {
            lastframeaddr = unsafe { lastframeaddr.byte_sub(lastframesize + REG64_BYTESIZE) };
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
                    unsafe { rbp.byte_sub(usize::try_from(idx.abs()).unwrap() * REG64_BYTESIZE) };
                if pinfo.hasfp {
                    tmp = unsafe { tmp.byte_add(REG64_BYTESIZE) };
                }
                let val = registers[usize::from(*reg)];
                unsafe { ptr::write(tmp as *mut u64, val) };
            }
        }

        // Expensive sanity checks in debug builds.
        #[cfg(debug_assertions)]
        let mut checker = DeoptChecker::default();

        // Now write all live variables to the new stack in the order they are listed in the AOT
        // stackmap.
        for aotvar in rec.live_vals.iter() {
            // Read live JIT values from the trace's stack frame.
            let jitval = match cgd.live_vars[varidx].1 {
                VarLocation::Stack { frame_off, size } => {
                    // rbp-0 can't contain a variable.
                    // [rbp-0] points to either the return address or the previous frame's rbp
                    // (when using --no-omit-framepointer) and thus can't contain live variables.
                    debug_assert!(frame_off > 0);
                    let p = unsafe { frameaddr.byte_sub(usize::try_from(frame_off).unwrap()) };
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
                VarLocation::ConstPtr(v) => u64::try_from(v).unwrap(),
                VarLocation::Direct { frame_off, size } => {
                    // See comment below: this case never needs to do anything.
                    debug_assert_eq!(
                        aotvar[0],
                        SMLocation::Direct(6, frame_off, u16::try_from(size).unwrap())
                    );
                    varidx += 1;
                    continue;
                }
            };
            varidx += 1;

            let aotloc = if aotvar.len() == 1 {
                &aotvar[0]
            } else {
                todo!("Deal with multi register locations");
            };
            match aotloc {
                SMLocation::Register(reg, size, extras) => {
                    #[cfg(debug_assertions)]
                    checker.record_and_check(
                        isize::try_from(*reg).unwrap(),
                        usize::from(*size),
                        jitval,
                    );
                    registers[usize::from(*reg)] = jitval;
                    for extra in extras {
                        #[cfg(debug_assertions)]
                        checker.record_and_check(isize::from(*extra), usize::from(*size), jitval);
                        // Write any additional locations that were tracked for this variable.
                        // Numbers greater or equal to zero are registers in Dwarf notation.
                        // Negative numbers are offsets relative to RBP.
                        if *extra >= 0 {
                            registers[usize::try_from(*extra).unwrap()] = jitval;
                        } else if *extra < 0 {
                            let temp = if i == 0 {
                                // Write values to the (still intact) bottom frame.
                                unsafe { frameaddr.offset(isize::from(*extra)) }
                            } else {
                                // Write values to a reconstructed frame.
                                unsafe { rbp.offset(isize::from(*extra)) }
                            };
                            match size {
                                1 => unsafe { ptr::write::<u16>(temp as *mut u16, jitval as u16) },
                                2 => unsafe { ptr::write::<u16>(temp as *mut u16, jitval as u16) },
                                4 => unsafe { ptr::write::<u32>(temp as *mut u32, jitval as u32) },
                                8 => unsafe { ptr::write::<u64>(temp as *mut u64, jitval) },
                                16 => {
                                    // FIXME: This case is clearly not safe in general: it just so
                                    // happens to work because it the moment the biggest value we
                                    // handle is 64 bits (be that a pointer, u64, or double). If
                                    // and when we support values bigger than 64 bits, this line
                                    // will lead to weird problems.
                                    unsafe { ptr::write::<u64>(temp as *mut u64, jitval) };
                                }
                                _ => todo!("{}", size),
                            }
                        }
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
                    #[cfg(debug_assertions)]
                    checker.record_and_check(
                        isize::try_from(*off).unwrap(),
                        usize::from(*size),
                        jitval,
                    );
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
                rsp = unsafe { rsp.byte_add(REG64_BYTESIZE) };
            }
        }

        // Write the return address for the previous frame into the current frame.
        unsafe {
            rsp = rsp.sub(REG64_BYTESIZE);
            ptr::write(rsp as *mut u64, rec.offset);
        }
    }

    // Write the live registers into the new stack. We put these at the very end of the new stack
    // so that they can be immediately popped after we memcpy'd the new stack over.
    for reg in RECOVER_REG {
        unsafe {
            rsp = rsp.byte_sub(REG64_BYTESIZE);
            ptr::write(rsp as *mut u64, registers[reg]);
        }
    }

    // Compute the address to which we want to write the new stack. This is immediately after the
    // frame containing the control point.
    let (rec, pinfo) = aot_smaps.get(usize::try_from(cgd.inlined_frames[0].safepoint.id).unwrap());
    let mut newframedst = unsafe { frameaddr.byte_sub(usize::try_from(rec.size).unwrap()) };
    if pinfo.hasfp {
        // `frameaddr` is the RBP value of the bottom frame after pushing the previous frame's RBP.
        // However, `rec.size` includes the pushed RBP, so we need to subtract it here again.
        newframedst = unsafe { newframedst.byte_add(REG64_BYTESIZE) };
    }

    mt.guard_failure(ctr, gid, frameaddr);

    // Now overwrite the existing stack with our newly recreated one.
    unsafe { replace_stack(newframedst, newstack, memsize) };
}

/// Writes the stack frames that we recreated in [__yk_deopt] onto the current stack, overwriting
/// the stack frames of any running traces in the process. This deoptimises trace execution after
/// which we can safely return to the normal execution of the interpreter.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn replace_stack(dst: *mut c_void, src: *const c_void, size: usize) -> ! {
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
    )
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    use super::{DeoptChecker, ranges_overlap};

    #[test]
    #[cfg(debug_assertions)]
    fn overlapping_ranges() {
        assert!(ranges_overlap(&(1..10), &(5..6)));
        assert!(ranges_overlap(&(5..6), &(1..10)));
        assert!(ranges_overlap(&(1..2), &(1..2)));
        assert!(ranges_overlap(&(-100..100), &(-1..0)));
        assert!(ranges_overlap(&(-1..0), &(-100..100)));

        assert!(!ranges_overlap(&(1..1), &(1..1)));
        assert!(!ranges_overlap(&(0..5), &(5..10)));
        assert!(!ranges_overlap(&(5..10), &(0..5)));
        assert!(!ranges_overlap(&(1..2), &(9..10)));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn deopt_checker() {
        let mut dc = DeoptChecker::default();
        // Registers.
        dc.record_and_check(0, 8, 0x11223344556677);
        dc.record_and_check(0, 8, 0x11223344556677);
        dc.record_and_check(1, 8, 0xaabbccddeeff00);
        dc.record_and_check(8, 1, 0xaabbccddeeff00);
        dc.record_and_check(8, 1, 0xaabbccddeeff00);
        // Memory.
        dc.record_and_check(-8, 8, 0x11223344556677);
        dc.record_and_check(-12, 4, 0xffffffff);
        dc.record_and_check(-16, 4, 0x11111111);
        dc.record_and_check(-17, 1, 0xaa);
        dc.record_and_check(-17, 1, 0xaa);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Overlapping register deopt for DWARF reg 0! 8 bytes 0x1122334455667788 vs 8 bytes 0xaabbccddeeff0011"
    )]
    fn deopt_checker_diff_regval() {
        let mut dc = DeoptChecker::default();
        dc.record_and_check(0, 8, 0x1122334455667788);
        dc.record_and_check(0, 8, 0xaabbccddeeff0011);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Overlapping register deopt for DWARF reg 0! 8 bytes 0xffffffffffffffff vs 1 bytes 0x00000000000000ff"
    )]
    fn deopt_checker_ok_overlap() {
        let mut dc = DeoptChecker::default();
        dc.record_and_check(0, 8, 0xffffffffffffffff);
        // In theory this could be OK, but for now we bail out. We haven't seen this in the wild.
        dc.record_and_check(0, 1, 0xff);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Suspicious memory deopt! range: -8..0, val: 0xffffffffffffffff vs. range -8..0, val: 0x0000000000000000"
    )]
    fn deopt_checker_diff_memval() {
        let mut dc = DeoptChecker::default();
        dc.record_and_check(-8, 8, 0xffffffffffffffff);
        dc.record_and_check(-8, 8, 0x0000000000000000);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Suspicious memory deopt! range: -8..0, val: 0xffffffffffffffff vs. range -8..-4, val: 0x0000000000000000"
    )]
    fn deopt_checker_overlap_mem() {
        let mut dc = DeoptChecker::default();
        dc.record_and_check(-8, 8, 0xffffffffffffffff);
        dc.record_and_check(-8, 4, 0x000000);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn deopt_checker_ok_overlap_mem() {
        let mut dc = DeoptChecker::default();
        dc.record_and_check(-56, 8, 0x3);
        dc.record_and_check(-56, 4, 0x3);

        let mut dc = DeoptChecker::default();
        dc.record_and_check(-100, 4, 0xff);
        dc.record_and_check(-100, 1, 0xff);
    }
}
