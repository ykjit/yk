use std::alloc::{alloc, dealloc, Layout};
use std::convert::TryFrom;
use std::sync::Arc;
use ykpack::{
    self, Body, BodyFlags, CallOperand, Constant, ConstantInt, IPlace, Local, Statement,
    Terminator, UnsignedInt,
};
use yktrace::sir::SIR;

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

pub struct SIRInterpreter {
    frames: Vec<StackFrame>,
    bbidx: ykpack::BasicBlockIndex,
}

impl SIRInterpreter {
    pub fn new(body: &Body) -> Self {
        let frame = SIRInterpreter::create_frame(body);
        SIRInterpreter {
            frames: vec![frame],
            bbidx: 0,
        }
    }

    /// Given a vector of local declarations, create a new StackFrame, which allocates just enough
    /// space to hold all of them.
    fn create_frame(body: &Body) -> StackFrame {
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

    pub unsafe fn interpret(&mut self, body: Arc<ykpack::Body>) {
        // Ignore yktrace::trace_debug.
        if body.flags.contains(BodyFlags::TRACE_DEBUG) {
            return;
        }

        let mut bodies = vec![body];
        let mut returns = Vec::new();
        while let Some(body) = bodies.last() {
            let bbidx = usize::try_from(self.bbidx).unwrap();
            let block = &body.blocks[bbidx];
            for stmt in block.stmts.iter() {
                match stmt {
                    Statement::MkRef(dest, src) => self.mkref(dest, src),
                    Statement::DynOffs { .. } => todo!(),
                    Statement::Store(dest, src) => self.store(dest, src),
                    Statement::BinaryOp { .. } => todo!(),
                    Statement::Nop => {}
                    Statement::Unimplemented(_) | Statement::Debug(_) => todo!(),
                    Statement::Cast(..) => todo!(),
                    Statement::StorageLive(_) | Statement::StorageDead(_) => {}
                    Statement::Call(..) => unreachable!(),
                }
            }
            match &block.term {
                Terminator::Call {
                    operand: op,
                    args,
                    destination: dest,
                } => {
                    let fname = if let CallOperand::Fn(sym) = op {
                        sym
                    } else {
                        todo!("unknown call target");
                    };

                    // Initialise the new stack frame.
                    let body = SIR.body(fname).unwrap();
                    let mut frame = SIRInterpreter::create_frame(&*body);
                    frame.copy_args(args, self.frame());
                    self.frames.push(frame);
                    self.bbidx = 0;
                    returns.push(dest.as_ref().map(|(p, b)| (p.clone(), *b)));
                    bodies.push(body);
                }
                Terminator::Return => {
                    // Are we returning from a call?
                    if let Some(v) = returns.pop() {
                        // Restore the previous stack frame, but keep the other frame around so we
                        // can copy over the return value to the destination.
                        let oldframe = self.frames.pop().unwrap();
                        if let Some((dest, bbidx)) = v {
                            // Get a pointer to the return value of the called frame.
                            let ret_ptr = oldframe.local_ptr(&Local(0));
                            // Write the return value to the destination in the previous frame.
                            let dst_ptr = self.frame().iplace_to_ptr(&dest);
                            let size = usize::try_from(SIR.ty(&dest.ty()).size()).unwrap();
                            self.frame_mut().write_val(dst_ptr, ret_ptr, size);
                            self.bbidx = bbidx;
                        }
                        // Restore previous body.
                        bodies.pop();
                    } else {
                        // We are returning from the first body, so we are done interpreting.
                        break;
                    }
                }
                t => todo!("{}", t),
            }
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
