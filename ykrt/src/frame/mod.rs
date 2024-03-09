#![allow(clippy::comparison_chain)]
#![allow(clippy::missing_safety_doc)]

use llvm_sys::{core::*, prelude::LLVMValueRef};
use object::{Object, ObjectSection};
#[cfg(not(test))]
use std::thread;
use std::{ffi::c_void, ptr, sync::LazyLock};
use ykaddr::obj::SELF_BIN_MMAP;
use yksmp::{Location as SMLocation, PrologueInfo, Record, StackMapParser};

mod llvmbridge;
pub(crate) use llvmbridge::{BitcodeSection, __yktracec_get_aot_module};
use llvmbridge::{Type, Value};

struct AOTStackmapInfo {
    pinfos: Vec<PrologueInfo>,
    records: Vec<(Record, usize)>,
}

impl AOTStackmapInfo {
    fn get(&self, stackmapid: usize) -> (&Record, &PrologueInfo) {
        let (rec, pid) = &self.records[stackmapid];
        let pinfo = &self.pinfos[*pid];
        (rec, pinfo)
    }
}

static AOT_STACKMAPS: LazyLock<Result<AOTStackmapInfo, String>> = LazyLock::new(|| {
    fn errstr(msg: &str) -> String {
        format!("failed to load stackmaps: {}", msg)
    }

    // We use an inner function so that we can use the `?` operator for errors.
    fn load_stackmaps() -> Result<AOTStackmapInfo, String> {
        // Load the stackmap from the binary to parse in tthe stackmaps.
        let object = object::File::parse(&**SELF_BIN_MMAP).map_err(|e| errstr(&e.to_string()))?;
        let sec = object
            .section_by_name(".llvm_stackmaps")
            .ok_or_else(|| errstr("can't find section"))?;

        // Parse the stackmap.
        let data = sec.data().map_err(|e| errstr(&e.to_string()))?;
        let (entries, numrecs) = StackMapParser::get_entries(data);
        let mut pinfos = Vec::new();
        let mut records = Vec::new();
        let numrecs_usize = usize::try_from(numrecs).map_err(|e| errstr(&e.to_string()))?;
        records.resize_with(numrecs_usize, || (Record::empty(), 0));
        for entry in entries {
            pinfos.push(entry.pinfo);
            for r in entry.records {
                let idx = usize::try_from(r.id).map_err(|e| errstr(&e.to_string()))?;
                records[idx] = (r, pinfos.len() - 1);
            }
        }
        Ok(AOTStackmapInfo { pinfos, records })
    }

    load_stackmaps()
});

pub(crate) fn load_aot_stackmaps() {
    // Rust unit test binaries will not contain stackmaps, so don't try to load them.
    #[cfg(not(test))]
    thread::spawn(|| LazyLock::force(&AOT_STACKMAPS));
}

static USIZEOF_POINTER: usize = std::mem::size_of::<*const ()>();
static ISIZEOF_POINTER: isize = std::mem::size_of::<*const ()>() as isize;
static RBP_DWARF_NUM: u16 = 6;

/// Live value.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct SGValue {
    pub val: u64,
    pub ty: Type,
}

impl SGValue {
    pub fn new(val: u64, ty: Type) -> Self {
        SGValue { val, ty }
    }
}

/// A frame holding live variables.
struct Frame {
    vars: Vec<SGValue>,
    pc: Value,
}

impl Frame {
    fn new(pc: Value) -> Frame {
        Frame {
            vars: Vec::new(),
            pc,
        }
    }

    /// Get the value of the variable `key` in this frame.
    fn get(&self, i: usize) -> Option<&SGValue> {
        self.vars.get(i)
    }

    /// Add new variable `key` with value `val`.
    fn add(&mut self, val: SGValue) {
        self.vars.push(val);
    }
}

/// The struct responsible for reconstructing the new frames after a guard failure.
pub(crate) struct FrameReconstructor {
    /// Current frames.
    frames: Vec<Frame>,
}

impl FrameReconstructor {
    /// Create a new instance and initialise the frames we need to reconstruct.
    pub unsafe fn new(activeframes: &[LLVMValueRef]) -> FrameReconstructor {
        FrameReconstructor {
            frames: activeframes
                .into_iter()
                .map(|x| Frame::new(Value::new(*x)))
                .collect::<Vec<_>>(),
        }
    }

    /// Generate frames from stackmap information after a guard failure. The new frames are stored
    /// inside some allocated memory whose pointer this function returns. The frames are then later
    /// `memcpy`ed to the actual stack by [ykcapi::__ykrt_reconstruct_frames].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn reconstruct_frames(&self, btmframeaddr: *mut c_void) -> (*const c_void, usize) {
        // Vec holding currently active register values.
        let mut registers = [0; 16];

        // The final size of the memory we need to allocate. Inititialised with space to store
        // registers for register recovery.
        let mut memsize: usize = 15 * USIZEOF_POINTER;
        // Vec to collect stackmaps for each frame.
        let mut smaps = Vec::new();
        // Size of the bottom-most frame (frame containing the control point).
        let mut btmframesize = 0;

        // Collect stackmaps for each frame and calculate the final size of memory required to
        // store the reconstructed stack.
        for (i, frame) in self.frames.iter().enumerate() {
            // Get stackmap ID for the current frame's pc.
            let smcall = frame.pc;
            debug_assert!(smcall.is_call());
            debug_assert!(smcall.is_intrinsic());
            let smid = unsafe { LLVMConstIntGetZExtValue(smcall.get_operand(0).get()) };
            // Find prologue info and stackmap record for this frame.
            let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
            let (rec, pinfo) = aot_smaps.get(usize::try_from(smid).unwrap());
            // We don't need to allocate memory for the bottom-most frame, i.e. the frame
            // containing the control point, since this frame already exists and doesn't need to be
            // reconstructed.
            if i > 0 {
                memsize += usize::try_from(rec.size).unwrap();
            } else {
                // Get the size of the frame containing the control point, which we'll later use to
                // calculate the stack offset to write the new frames to. Note that we'll be adding
                // this to a pointer retrieved via the `llvm.frameaddr` intrinsic, which doesn't
                // include the pushed RBP at the beginning of the frame, whereas the size reported
                // by the stackmap does. So if this frame uses the frame pointer, substract its
                // size again.
                btmframesize = usize::try_from(rec.size).unwrap();
                if pinfo.hasfp {
                    btmframesize -= USIZEOF_POINTER;
                }
            }
            memsize += USIZEOF_POINTER; // Reserve space for return address.
            smaps.push((i, frame, rec, pinfo));
        }

        // Add space to store the size of the stack which we'll need later to memcpy the correct
        // amount.
        memsize += USIZEOF_POINTER;

        // Now that we've calculated the stack's size, allocate memory for it.
        let newstack = unsafe { libc::malloc(memsize) };

        // Generate and write frames to the new stack. Since the stack grows downwards and we need
        // to keep track of spilled register values we write to `newstack` from back to front. To
        // make things easier to think about we create two variables `rbp` and `rsp` which simulate
        // their assembler namesakes.
        // Note that while the bottom-most frame still exists and doesn't need to be reconstructed,
        // we still need to get the register values from its stackmap in case those registers are
        // spilled by the next frame.

        let mut rbp = unsafe { newstack.offset(isize::try_from(memsize).unwrap()) };
        let mut rsp = rbp;
        // Keep track of the addresses of the current and previous frame so we can update the RBP
        // register as needed.
        let mut currframe = btmframeaddr;
        let mut nextframe = btmframeaddr;

        for (i, frame, rec, pinfo) in smaps {
            debug_assert!(rec.size != u64::MAX);

            // WRITE RBP
            // If the current frame has pushed RBP we need to do the same (unless we are processing
            // the bottom-most frame).
            if pinfo.hasfp && i > 0 {
                rsp = unsafe { rsp.sub(USIZEOF_POINTER) };
                rbp = rsp;
                unsafe { ptr::write(rsp as *mut u64, currframe as u64) };
            }

            // Now that we've (potentially) written the last frame's address, update currframe to
            // the actual current frame.
            currframe = nextframe;

            // Update RBP to represent this frame's address.
            if pinfo.hasfp {
                registers[usize::from(RBP_DWARF_NUM)] = currframe as u64;
            }

            // Calculate the next frame's address by substracting its size (plus return address)
            // from the current frame's address.
            nextframe =
                unsafe { currframe.sub(usize::try_from(rec.size).unwrap() + USIZEOF_POINTER) };

            // WRITE CALLEE-SAVED REGISTERS
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
                        unsafe { rbp.offset(isize::try_from(*idx).unwrap() * ISIZEOF_POINTER) };
                    if pinfo.hasfp {
                        tmp = unsafe { tmp.offset(ISIZEOF_POINTER) };
                    }
                    let val = registers[usize::from(*reg)];
                    unsafe { ptr::write(tmp as *mut u64, val) };
                }
            }

            // WRITE STACKMAP LOCATIONS.
            // Now write all live variables to the new stack in the order they are listed in the
            // AOT stackmap call.
            for (j, lv) in rec.live_vars.iter().enumerate() {
                // Adjust the operand index by 2 to skip stackmap ID and shadow bytes.
                let val = if let Some(sg) = frame.get(j) {
                    sg.val
                } else {
                    // Sidetraces require an unconditional guard at the end which deopts back to
                    // the control point. The stackmap at this location contains the value returned
                    // by the call to `frameaddr`. We are not interested in its value and tracking
                    // and deoptimising it leads to issues (e.g. overwriting RBP when we don't want
                    // to). So the easiest approach for now is to simply omit this live variable
                    // during deoptimisation. In the future we will patch side traces into the
                    // parent trace, so this hack will no longer be needed then.
                    debug_assert!(frame
                        .pc
                        .get_operand(u32::try_from(j + 2).unwrap())
                        .is_frameaddr_call());
                    continue;
                };
                let l = if lv.len() == 1 {
                    lv.get(0).unwrap()
                } else {
                    todo!("Deal with multi register locations");
                };

                // Iterate over all locations. Register locations just update the current value in
                // the registers vector. Direct locations make up part of the stack so need to
                // written to the allocated memory. Other locations we haven't encountered yet, so
                // will deal with them as they appear.
                match l {
                    SMLocation::Register(reg, _size, off, extra) => {
                        if *reg > 16 {
                            // FIXME: Implement deoptimisation into floating point registers.
                            //        LLVM stackmaps use DWARF register number mapping.
                            //        See page 57 in https://refspecs.linuxbase.org /elf/x86_64-abi-0.99.pdf
                            //        Anything above 16 (most notably floating registers) we currently do not support.
                            todo!()
                        }
                        registers[usize::from(*reg)] = val;
                        if *extra != 0 {
                            // The stackmap has recorded an additional register we need to write
                            // this value to.
                            registers[usize::from(*extra - 1)] = val;
                        }
                        if i == 0 {
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
                            unsafe { ptr::write::<u64>(temp as *mut u64, val) };
                        } else if *off > 0 {
                            registers[usize::try_from(*off - 1).unwrap()] = val;
                        }
                    }
                    SMLocation::Direct(..) => {
                        // Direct locations are pointers to the stack, stored on the stack (e.g.
                        // `alloca` or GEP). Our shadow stack unifies the JIT and AOT stacks,
                        // replacing them with a heap allocation. For this reason, no `Direct`
                        // stackmap entries can exist apart from those special-cased in the shadow
                        // stack pass (e.g. the control point struct and the result of
                        // `yk_mt_location_new()`). The exceptions only appear (for now) at frame
                        // index 0 (where the control point is), and since this frame will not be
                        // re-written by deopt, there's no need to restore those direct locations
                        // anyway.
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
                            unsafe { btmframeaddr.offset(isize::try_from(*off).unwrap()) }
                        } else {
                            unsafe { rbp.offset(isize::try_from(*off).unwrap()) }
                        };
                        debug_assert!(*off < i32::try_from(rec.size).unwrap());
                        // FIXME: The minimum size reported by the stackmap is 1 which represents 1
                        // byte. LLVM IR allows for smaller sizes, e.g. `i1` representing a single
                        // bit. It is currently unclear how that affects this code, so I'm leaving
                        // this comment here so we don't forget.
                        match size {
                            1 => unsafe { ptr::write::<u8>(temp as *mut u8, val as u8) },
                            4 => unsafe { ptr::write::<u32>(temp as *mut u32, val as u32) },
                            8 => unsafe { ptr::write::<u64>(temp as *mut u64, val) },
                            _ => todo!(),
                        }
                    }
                    SMLocation::Constant(_v) => {
                        todo!()
                    }
                    SMLocation::LargeConstant(_v) => {
                        todo!();
                    }
                }
            }
            if i > 0 {
                // Advance the "virtual RSP" to the next frame.
                rsp = unsafe { rbp.sub(usize::try_from(rec.size).unwrap()) };
                if pinfo.hasfp {
                    // The stack size recorded by the stackmap includes a pushed RBP. However, we
                    // will have already adjusted the "virtual RSP" at the beginning if `hasfp` is
                    // true. If that's the case, re-adjust the "virtual RSP" again to account for
                    // this.
                    rsp = unsafe { rsp.offset(ISIZEOF_POINTER) };
                }
            }
            // WRITE RETURN ADDRESS.
            // Write the return address for the previous frame into the current frame.
            rsp = unsafe { rsp.sub(USIZEOF_POINTER) };
            unsafe {
                ptr::write(rsp as *mut u64, rec.offset);
            }
        }

        // WRITE REGISTERS
        // Now we write the current values of the registers we care about (note: there might be
        // others, but this works so far). These do not belong to any stack frame and will be
        // immedialy popped after stack reconstruction to reset the registers to the correct state.
        // Note that this will inevitably set some registers to a different value than than if we
        // had not run a trace. But since those registers aren't live this will be fine.
        for reg in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] {
            rsp = unsafe { rsp.sub(USIZEOF_POINTER) };
            unsafe {
                ptr::write(rsp as *mut u64, registers[reg]);
            }
        }

        // PUSH STACK SIZE
        // Finally we push the size of the allocated memory which we use later to memcpy the
        // correct amount.
        unsafe {
            rsp = rsp.sub(USIZEOF_POINTER);
            ptr::write(rsp as *mut usize, memsize);
        }

        // Return the pointer to the new stack.
        (newstack, btmframesize)
    }

    /// Add a live variable and its value to the current frame.
    pub fn var_init(&mut self, aotval: *const c_void, sfidx: usize, mut val: u64) {
        let aotval = unsafe { Value::new(aotval as LLVMValueRef) };
        let ty = aotval.get_type();
        if aotval.get_type().is_integer() {
            // Stackmap "small constants" get their value sign-extended to fill the reserved 32-bit
            // space in the stackmap record. If the type of the constant is actually smaller than
            // 32 bits, then we have to discard the unwanted high-order bits.
            let iw = ty.get_int_width();
            val &= u64::MAX >> (64 - iw);
        }

        let liveval = SGValue::new(val, ty);
        self.frames.get_mut(sfidx).unwrap().add(liveval);
    }
}
