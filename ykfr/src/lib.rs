#![feature(once_cell)]

use llvm_sys::{core::*, target::LLVMABISizeOfType};
use object::{Object, ObjectSection};
use std::{
    collections::HashMap,
    convert::TryFrom,
    env,
    ffi::{c_void, CStr},
    fs, ptr, slice,
    sync::LazyLock,
};
use yksmp::{Location as SMLocation, SMEntry, StackMapParser};

mod llvmbridge;
use llvmbridge::{get_aot_original, Module, Type, Value};

pub static AOT_STACKMAPS: LazyLock<Vec<SMEntry>> = LazyLock::new(|| {
    // Load the stackmap from the binary to parse in the stackmaps.
    // FIXME: Don't use current_exe.
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(pathb.as_path()).unwrap();
    let exemmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let object = object::File::parse(&*exemmap).unwrap();
    let sec = object.section_by_name(".llvm_stackmaps").unwrap();

    // Parse the stackmap.
    let slice = unsafe {
        slice::from_raw_parts(
            sec.address() as *mut u8,
            usize::try_from(sec.size()).unwrap(),
        )
    };
    StackMapParser::get_entries(slice)
});

static USIZEOF_POINTER: usize = std::mem::size_of::<*const ()>();
static ISIZEOF_POINTER: isize = std::mem::size_of::<*const ()>() as isize;
static RBP_DWARF_NUM: u16 = 6;

/// Active frames (basic block index, instruction index, function name) in the AOTModule where the
/// guard failure occured. Mirrors the struct defined in ykllvmwrap/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
pub struct FrameInfo {
    pub bbidx: usize,
    pub instridx: usize,
    pub fname: *const i8,
}

/// Stopgap interpreter values.
#[derive(Clone, Copy, PartialEq)]
pub struct SGValue {
    pub val: u64,
    pub ty: Type,
}

impl SGValue {
    pub fn new(val: u64, ty: Type) -> Self {
        SGValue { val, ty }
    }

    pub fn with_type(&self, ty: Type) -> Self {
        SGValue::new(self.val, ty)
    }
}

/// A frame holding live variables.
struct Frame {
    vars: HashMap<Value, SGValue>,
    pc: Value,
}

impl Frame {
    fn new(pc: Value) -> Frame {
        Frame {
            vars: HashMap::new(),
            pc,
        }
    }

    /// Get the value of the variable `key` in this frame.
    fn get(&self, key: &Value) -> Option<&SGValue> {
        self.vars.get(key)
    }

    /// Add new variable `key` with value `val`.
    fn add(&mut self, key: Value, val: SGValue) {
        self.vars.insert(key, val);
    }
}

fn get_stackmap_call(pc: Value) -> Value {
    debug_assert!(pc.is_instruction());
    // Stackmap instructions are inserted after calls, but before branch instructions. So we need
    // slightly different logic to find them.
    let sm = if pc.is_call() {
        unsafe { Value::new(LLVMGetNextInstruction(pc.get())) }
    } else {
        unsafe { Value::new(LLVMGetPreviousInstruction(pc.get())) }
    };
    if cfg!(debug_assertions) {
        // If we are in debug mode, make sure this is indeed always the stackmap call.
        debug_assert!(sm.is_call());
        debug_assert!(sm.is_intrinsic());
        let id = unsafe { LLVMGetIntrinsicID(LLVMGetCalledValue(sm.get())) };
        let mut len = 0;
        let ptr = unsafe { LLVMIntrinsicGetName(id, &mut len) };
        let name = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        debug_assert_eq!(name, "llvm.experimental.stackmap");
    }
    sm
}

/// The struct responsible for reconstructing the new frames after a guard failure.
pub struct FrameReconstructor {
    /// AOT IR module.
    module: Module,
    /// Current frames.
    frames: Vec<Frame>,
}

impl FrameReconstructor {
    /// Create a new instance and initialise the frames we need to reconstruct.
    pub unsafe fn new(activeframes: &[FrameInfo]) -> FrameReconstructor {
        // Get AOT module IR and parse it.
        let module = Module::from_bc();

        // Initialise frames.
        let mut frames = Vec::with_capacity(activeframes.len());
        for frame in activeframes {
            let funcname = std::ffi::CStr::from_ptr(frame.fname);
            let func = module.function(funcname.as_ptr());
            let bb = func.bb(frame.bbidx);
            let instr = bb.instruction(frame.instridx);
            frames.push(Frame::new(instr));
        }
        FrameReconstructor { module, frames }
    }

    /// Generate frames from stackmap information after a guard failure. The new frames are stored
    /// inside some allocated memory whose pointer this function returns. The frames are then later
    /// `memcpy`ed to the actual stack by [ykcapi::__ykrt_reconstruct_frames].
    #[cfg(target_arch = "x86_64")]
    pub fn reconstruct_frames(&self, btmframeaddr: *mut c_void) -> *const c_void {
        // Vec holding currently active register values.
        let mut registers = vec![0; 16];

        // The final size of the memory we need to allocate. Inititialised with space to store
        // registers for register recovery.
        let mut memsize: usize = 15 * USIZEOF_POINTER;
        // Vec to collect stackmaps for each frame.
        let mut smaps = Vec::new();

        // Collect stackmaps for each frame and calculate the final size of memory required to
        // store the reconstructed stack.
        for (i, frame) in self.frames.iter().enumerate() {
            // Get stackmap ID for the current frame's pc.
            let smcall = get_stackmap_call(frame.pc);
            let smid = unsafe { LLVMConstIntGetZExtValue(smcall.get_operand(0).get()) };
            // Find prologue info and stackmap record for this frame.
            let mut pinfo = None;
            let mut rec = None;
            // Iterate over function entries to find the correct record and relevant prologue info.
            for entry in AOT_STACKMAPS.iter() {
                for r in &entry.records {
                    if r.id == smid {
                        pinfo = Some(&entry.pinfo);
                        rec = Some(r);
                        break;
                    }
                }
            }
            let rec = rec.unwrap();
            let pinfo = pinfo.unwrap();
            // We don't need to allocate memory for the bottom-most frame, i.e. the frame
            // containing the control point, since this frame already exists and doesn't need to be
            // reconstructed.
            if i > 0 {
                memsize += usize::try_from(rec.size).unwrap();
            }
            memsize += USIZEOF_POINTER; // Reserve space for return address.
            smaps.push((i, frame, rec, pinfo));
        }

        // Add space to store the size of the stack which we'll need later to memcpy the correct
        // amount.
        memsize += USIZEOF_POINTER;

        // Now that we've calculated the stack's size, allocate it using mmap.
        let mmap = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                memsize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        // Get the modules layout which we'll need to extract type sizes of LLVM IR.
        let layout = self.module.datalayout();

        // Generate and write frames to the mmap. Since the stack grows downwards and we need to
        // keep track of spilled register values we write to the mmap from back to front. To make
        // things easier to think about we create two variables rbp and rsp which simulate their
        // assembler namesakes.
        // Note that while the bottom-most frame still exists and doesn't need to be reconstructed,
        // we still need to get the register values from its stackmap in case those registers are
        // spilled by the next frame.

        let mut rbp = unsafe { mmap.offset(isize::try_from(memsize).unwrap()) };
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
            let smcall = get_stackmap_call(frame.pc);
            for (j, lv) in rec.live_vars.iter().enumerate() {
                // Adjust the operand index by 2 to skip stackmap ID and shadow bytes.
                let op = smcall.get_operand(u32::try_from(j + 2).unwrap());
                let val = frame.get(&op).unwrap().val;
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
                    SMLocation::Register(reg, _size, off) => {
                        registers[usize::from(*reg)] = val;
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
                    SMLocation::Direct(reg, off, _) => {
                        if i == 0 {
                            // skip first frame
                            continue;
                        }
                        debug_assert!(op.is_alloca());

                        // The sizes reported by the stackmap aren't always correct. But we can get
                        // the correct size from the IR.
                        let eltype = unsafe { LLVMGetAllocatedType(op.get()) };
                        let size = unsafe { LLVMABISizeOfType(layout, eltype) };
                        // Direct locations are always be in regards to RBP.
                        debug_assert_eq!(*reg, RBP_DWARF_NUM);
                        let temp = unsafe { rbp.offset(isize::try_from(*off).unwrap()) };
                        debug_assert!(*off < i32::try_from(rec.size).unwrap());
                        unsafe {
                            libc::memcpy(temp, val as *const c_void, usize::try_from(size).unwrap())
                        };
                    }
                    SMLocation::Indirect(reg, off, size) => {
                        if i == 0 {
                            // skip first frame
                            continue;
                        }
                        debug_assert_eq!(*reg, RBP_DWARF_NUM);
                        let temp = unsafe { rbp.offset(isize::try_from(*off).unwrap()) };
                        debug_assert!(*off < i32::try_from(rec.size).unwrap());
                        match size {
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
            // Write the return address for the previous frame into the current frame. Note that
            // for calls we don't want to return to the address just before the call since that
            // would mean executing the call twice. For anything else on the other hand we do need
            // to re-execute the instruction (e.g. icmp, test) in order to set the correct flags
            // for the following jump instruction.
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

        // Mark the memory read-only and return its pointer.
        unsafe { libc::mprotect(mmap, memsize, libc::PROT_READ) };
        mmap
    }

    /// Add a live variable and its value to the current frame.
    pub fn var_init(
        &mut self,
        bbidx: usize,
        instridx: usize,
        fname: &CStr,
        sfidx: usize,
        mut val: u64,
    ) {
        let func = self.module.function(fname.as_ptr());
        let bb = func.bb(bbidx);
        let instr = bb.instruction(instridx);

        if instr.get_type().is_integer() {
            // Stackmap "small constants" get their value sign-extended to fill the reserved 32-bit
            // space in the stackmap record. If the type of the constant is actually smaller than
            // 32 bits, then we have to discard the unwanted high-order bits.
            let iw = instr.get_type().get_int_width();
            val &= u64::MAX >> (64 - iw);
        }

        let orgaot = if sfidx == 0 {
            unsafe { get_aot_original(&instr) }
        } else {
            // Only the root stackframe contains the control point call, so for the other frames
            // there's no need to match live variables to their corresponding variables passed into
            // the control point. See `get_aot_original` for more details.
            None
        };
        let ty = instr.get_type();
        let value = SGValue::new(val, ty);
        self.frames.get_mut(sfidx).unwrap().add(instr, value);
        if let Some(v) = orgaot {
            self.frames.get_mut(sfidx).unwrap().add(v, value);
        }
    }
}
