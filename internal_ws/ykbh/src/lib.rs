use std::alloc::{alloc, dealloc, Layout};
use std::convert::TryFrom;
use ykpack::{
    self, CallOperand, Constant, ConstantInt, IPlace, Local, Statement, Terminator, TyKind,
    UnsignedInt, UnsignedIntTy,
};
use yktrace::sir::SIR;

/// Stores information needed to recreate stack frames in the SIRInterpreter.
pub struct FrameInfo {
    /// The symbol name of this frame.
    pub sym: String,
    /// Index of the current basic block we are in. When returning from a function call, the
    /// terminator of this block is were we continue.
    pub bbidx: usize,
    /// Pointer to memory containing the live variables.
    pub mem: *mut u8,
}

/// A stack frame for writing and reading locals. Note that the allocated memory this frame points
/// to needs to be freed manually before the stack frame is destoyed.
pub struct StackFrame {
    /// Pointer to allocated memory containing a frame's locals.
    locals: *mut u8,
    /// The offset of each Local into locals.
    offsets: Vec<usize>,
    /// The layout of locals. Needed for deallocating locals upon drop.
    layout: Layout,
}

impl Drop for StackFrame {
    fn drop(&mut self) {
        unsafe { dealloc(self.locals, self.layout) }
    }
}

impl StackFrame {
    /// Given a pointer `src` and a size, write its value to the pointer `dst`.
    pub fn write_val(&mut self, dst: *mut u8, src: *const u8, size: usize) {
        unsafe {
            std::ptr::copy(src, dst, size);
        }
    }

    /// Write a constant to the pointer `dst`.
    pub fn write_const(&mut self, dest: *mut u8, constant: &Constant) {
        match constant {
            Constant::Int(ci) => match ci {
                ConstantInt::UnsignedInt(ui) => match ui {
                    UnsignedInt::U8(v) => self.write_val(dest, [*v].as_ptr(), 1),
                    _ => todo!(),
                },
                ConstantInt::SignedInt(_si) => todo!(),
            },
            Constant::Bool(_b) => todo!(),
            Constant::Tuple(t) => {
                if SIR.ty(t).size() == 0 {
                    // ZST: do nothing.
                } else {
                    todo!()
                }
            }
            _ => todo!(),
        }
    }

    /// Stores one IPlace into another.
    fn store(&mut self, dest: &IPlace, src: &IPlace) {
        match src {
            IPlace::Val { .. } | IPlace::Indirect { .. } => {
                let src_ptr = self.iplace_to_ptr(src);
                let dst_ptr = self.iplace_to_ptr(dest);
                let size = usize::try_from(SIR.ty(&src.ty()).size()).unwrap();
                self.write_val(dst_ptr, src_ptr, size);
            }
            IPlace::Const { val, ty: _ty } => {
                let dst_ptr = self.iplace_to_ptr(dest);
                self.write_const(dst_ptr, val);
            }
            _ => todo!(),
        }
    }

    /// Copy over the call arguments from another frame.
    pub fn copy_args(&mut self, args: &Vec<IPlace>, frame: &StackFrame) {
        for (i, arg) in args.iter().enumerate() {
            let dst = self.local_ptr(&Local(u32::try_from(i + 1).unwrap()));
            match arg {
                IPlace::Val { .. } | IPlace::Indirect { .. } => {
                    let src = frame.iplace_to_ptr(arg);
                    let size = usize::try_from(SIR.ty(&arg.ty()).size()).unwrap();
                    self.write_val(dst, src, size);
                }
                IPlace::Const { val, .. } => {
                    self.write_const(dst, val);
                }
                _ => unreachable!(),
            }
        }
    }

    /// Get the pointer to a Local.
    fn local_ptr(&self, local: &Local) -> *mut u8 {
        let offset = self.offsets[usize::try_from(local.0).unwrap()];
        unsafe { self.locals.add(offset) }
    }

    /// Get the pointer for an IPlace, while applying all offsets.
    fn iplace_to_ptr(&self, place: &IPlace) -> *mut u8 {
        match place {
            IPlace::Val {
                local,
                off,
                ty: _ty,
            } => {
                // Get a pointer to the Val.
                let dest_ptr = self.local_ptr(&local);
                unsafe { dest_ptr.add(usize::try_from(*off).unwrap()) }
            }
            IPlace::Indirect { ptr, off, ty: _ty } => {
                // Get a pointer to the Indirect, which itself points to another pointer.
                let dest_ptr = self.local_ptr(&ptr.local) as *mut *mut u8;
                let ptr = unsafe {
                    // Dereference the pointer, by reading its value.
                    let mut p = std::ptr::read::<*mut u8>(dest_ptr);
                    // Add the offsets of the Indirect.
                    p = p.offset(isize::try_from(ptr.off).unwrap());
                    p = p.offset(isize::try_from(*off).unwrap());
                    p
                };
                // Now return the value as a pointer.
                ptr
            }
            _ => unreachable!(),
        }
    }
}

/// The SIR interpreter, also known as blackholing interpreter, is invoked when a guard fails in a
/// trace. It is initalised with information from the trace, e.g. live variables, stack frames, and
/// then run to get us back to a control point from where the normal interpreter then takes over.
pub struct SIRInterpreter {
    /// Keeps track of the current stack frames, created by function calls.
    frames: Vec<StackFrame>,
    /// Index of the basic block we were in when a function was called. Upon returning from a
    /// function, we go back to this basic block and look up its terminator to decide where to
    /// continue.
    returns: Vec<ykpack::BasicBlockIndex>,
    /// Function names relating to each stack frame. These are needed to retrieve the SIR body of
    /// the function which contains the statements we want to interpret.
    funcs: Vec<String>,
    /// Index of the basic block we are currently interpreting.
    bbidx: ykpack::BasicBlockIndex,
}

impl SIRInterpreter {
    pub fn new(sym: String) -> Self {
        let frame = SIRInterpreter::create_frame(&sym);
        SIRInterpreter {
            frames: vec![frame],
            returns: Vec::new(),
            funcs: vec![sym],
            bbidx: 0,
        }
    }

    /// Initialises the interpreter with information about live variables and stack frames,
    /// received from the failing guard.
    pub fn init_frames(v: Vec<FrameInfo>) -> Self {
        let mut frames = Vec::new();
        let mut funcs = Vec::new();
        let mut returns = Vec::new();
        for fi in v {
            let body = SIR.body(&fi.sym).unwrap();
            let frame = StackFrame {
                locals: fi.mem,
                offsets: body.offsets.clone(),
                layout: Layout::from_size_align(body.layout.0, body.layout.1).unwrap(),
            };
            frames.push(frame);
            funcs.push(fi.sym);
            returns.push(u32::try_from(fi.bbidx).unwrap());
        }
        SIRInterpreter {
            frames,
            returns,
            funcs,
            bbidx: 0,
        }
    }

    /// Run the SIR interpreter after it has been initialised by a guard failure. Since we start in
    /// the block where the guard failed, we immediately skip to the terminator and interpret it to
    /// see which block we need to start interpretation in.
    pub unsafe fn interpret(&mut self) {
        // Jump to the correct basic block.
        let lastfunc = self.funcs.last().unwrap();
        let lastret = *self.returns.last().unwrap();
        let body = SIR.body(lastfunc).unwrap();
        self.terminator(&body.blocks[lastret as usize].term);
        // Start interpretation.
        self._interpret();
    }

    /// Given a vector of local declarations, create a new StackFrame, which allocates just enough
    /// space to hold all of them.
    fn create_frame(sym: &String) -> StackFrame {
        let body = SIR.body(&sym).unwrap();
        let (size, align) = body.layout;
        let offsets = body.offsets.clone();
        let layout = Layout::from_size_align(size, align).unwrap();
        // Allocate memory for the locals
        let locals = unsafe { alloc(layout) };
        StackFrame {
            locals,
            offsets,
            layout,
        }
    }

    /// Returns a reference to the currently active frame.
    fn frame(&self) -> &StackFrame {
        self.frames.last().unwrap()
    }

    /// Returns a mutable reference to the currently active frame.
    fn frame_mut(&mut self) -> &mut StackFrame {
        self.frames.last_mut().unwrap()
    }

    /// Inserts a pointer to the trace inputs into `locals`.
    pub fn set_trace_inputs(&mut self, tio: *mut u8) {
        // FIXME Later this also sets other already initialised variables as well as the program
        // counter of the interpreter.
        let ptr = self.frame().local_ptr(&Local(1)); // The trace inputs live in $1
        unsafe {
            // Write the pointer value of `tio` into locals.
            std::ptr::write::<*mut u8>(ptr as *mut *mut u8, tio);
        }
    }

    pub unsafe fn _interpret(&mut self) {
        while let Some(func) = self.funcs.last() {
            let body = SIR.body(&func).unwrap();
            let bbidx = usize::try_from(self.bbidx).unwrap();
            let block = &body.blocks[bbidx];
            for stmt in block.stmts.iter() {
                match stmt {
                    Statement::MkRef(dest, src) => self.mkref(&dest, &src),
                    Statement::DynOffs { .. } => todo!(),
                    Statement::Store(dest, src) => self.store(&dest, &src),
                    Statement::BinaryOp { .. } => todo!(),
                    Statement::Nop => {}
                    Statement::Unimplemented(_) | Statement::Debug(_) => todo!(),
                    Statement::Cast(..) => todo!(),
                    Statement::StorageLive(_) | Statement::StorageDead(_) => {}
                    Statement::Call(..) => unreachable!(),
                }
            }

            self.terminator(&block.term);
        }
    }

    fn terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Call {
                operand: op,
                args,
                destination: _dest,
            } => {
                let fname = if let CallOperand::Fn(sym) = op {
                    sym
                } else {
                    todo!("unknown call target");
                };

                // Initialise the new stack frame.
                let mut frame = SIRInterpreter::create_frame(&fname);
                frame.copy_args(args, self.frame());
                self.frames.push(frame);
                self.returns.push(self.bbidx);
                self.funcs.push(fname.to_string());
                self.bbidx = 0;
            }
            Terminator::Return => {
                self.funcs.pop();
                // Are we still inside a nested call? Otherwise we are returning from the first
                // body, so we are done interpreting.
                if self.funcs.len() > 0 {
                    let returnbb = self.returns.pop().unwrap();
                    let func = self.funcs.last().unwrap();
                    let body = SIR.body(&func).unwrap();
                    // Check the previous call terminator to find out where we have to go next.
                    let (dest, bbidx) = match &body.blocks[usize::try_from(returnbb).unwrap()].term
                    {
                        Terminator::Call {
                            operand: _,
                            args: _,
                            destination: dest,
                        } => dest.as_ref().map(|(p, b)| (p.clone(), *b)).unwrap(),
                        _ => unreachable!(),
                    };

                    // Restore the previous stack frame, but keep the other frame around so we
                    // can copy over the return value to the destination.
                    let oldframe = self.frames.pop().unwrap();
                    // Get a pointer to the return value of the called frame.
                    let ret_ptr = oldframe.local_ptr(&Local(0));
                    // Write the return value to the destination in the previous frame.
                    let dst_ptr = self.frame().iplace_to_ptr(&dest);
                    let size = usize::try_from(SIR.ty(&dest.ty()).size()).unwrap();
                    self.frame_mut().write_val(dst_ptr, ret_ptr, size);
                    self.bbidx = bbidx;
                }
            }
            Terminator::SwitchInt {
                discr,
                values,
                target_bbs,
                otherwise_bb,
            } => {
                let val = self.read_int(discr);
                self.bbidx = *otherwise_bb;
                for (i, v) in values.iter().enumerate() {
                    if val == *v {
                        self.bbidx = target_bbs[i];
                        break;
                    }
                }
            }
            Terminator::Goto(bb) => {
                self.bbidx = *bb;
            }
            t => todo!("{}", t),
        }
    }

    fn read_int(&self, src: &IPlace) -> u128 {
        let ptr = self.frame().iplace_to_ptr(src);
        match &SIR.ty(&src.ty()).kind {
            TyKind::UnsignedInt(ui) => match ui {
                UnsignedIntTy::Usize => todo!(),
                UnsignedIntTy::U8 => unsafe { u128::from(std::ptr::read::<u8>(ptr)) },
                UnsignedIntTy::U16 => todo!(),
                UnsignedIntTy::U32 => todo!(),
                UnsignedIntTy::U64 => todo!(),
                UnsignedIntTy::U128 => todo!(),
            },
            TyKind::SignedInt(_si) => unreachable!(),
            _ => unreachable!(),
        }
    }

    /// Implements the Store statement.
    fn store(&mut self, dest: &IPlace, src: &IPlace) {
        self.frames.last_mut().unwrap().store(dest, src);
    }

    /// Creates a reference to an IPlace.
    fn mkref(&mut self, dest: &IPlace, src: &IPlace) {
        match dest {
            IPlace::Val { .. } | IPlace::Indirect { .. } => {
                // Get pointer to src.
                let frame = self.frames.last_mut().unwrap();
                let src_ptr = frame.iplace_to_ptr(src);
                let dst_ptr = frame.iplace_to_ptr(dest);
                unsafe {
                    std::ptr::write::<*mut u8>(dst_ptr as *mut *mut u8, src_ptr);
                }
            }
            _ => unreachable!(),
        }
    }
}
