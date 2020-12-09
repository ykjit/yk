use std::alloc::{alloc, dealloc, Layout};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;
use ykpack::{
    self, BodyFlags, CallOperand, Constant, ConstantInt, IPlace, Local, LocalDecl, Statement,
    Terminator, TypeId, UnsignedInt,
};
use yktrace::sir::SIR;

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

pub struct SIRInterpreter {
    frames: Vec<StackFrame>,
    bbidx: ykpack::BasicBlockIndex,
}

impl SIRInterpreter {
    pub fn new(local_decls: &Vec<LocalDecl>) -> Self {
        let frame = SIRInterpreter::allocate_locals(local_decls);
        SIRInterpreter {
            frames: vec![frame],
            bbidx: 0,
        }
    }

    fn allocate_locals(local_decls: &Vec<LocalDecl>) -> StackFrame {
        // FIXME Soon this will be pre-computed and handed to us by SIR.
        let mut offsets = Vec::new();
        let mut layout = Layout::from_size_align(0, 1).unwrap();
        for d in local_decls {
            let align = SIR.ty(&d.ty).align();
            let size = SIR.ty(&d.ty).size();
            let l = Layout::from_size_align(size.try_into().unwrap(), align.try_into().unwrap())
                .unwrap();
            let (nl, s) = layout.extend(l).unwrap();
            offsets.push(s);
            layout = nl;
        }
        layout = layout.pad_to_align();

        // Allocate memory for the locals
        let locals = unsafe { alloc(layout) };
        StackFrame {
            locals,
            offsets,
            layout,
        }
    }

    fn frame(&self) -> &StackFrame {
        self.frames.last().unwrap()
    }

    /// Inserts a pointer to the trace inputs into `locals`.
    pub fn set_trace_inputs(&mut self, tio: *mut u8) {
        // FIXME Later this also sets other already initialised variables as well as the program
        // counter of the interpreter.
        let ptr = self.local_ptr(&Local(1)); // The trace inputs live in $1
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

        loop {
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
                    Statement::Call(..) | Statement::StorageDead(_) => unreachable!(),
                }
            }

            match &block.term {
                Terminator::Call {
                    operand: op,
                    args: _args,
                    destination: dest,
                } => {
                    let fname = if let CallOperand::Fn(sym) = op {
                        sym
                    } else {
                        todo!("unknown call target");
                    };

                    // Initialise the new stack frame.
                    let body = SIR.body(fname).unwrap();
                    let frame = SIRInterpreter::allocate_locals(&body.local_decls);
                    self.frames.push(frame);
                    self.bbidx = 0;

                    self.interpret(body);
                    // Get pointer to result from current frame.
                    let ptr = self.local_ptr(&Local(0));
                    // Restore previous stack frame, but keep the other frame around so the pointer to
                    // the return value stays valid until we've copied it.
                    let _oldframe = self.frames.pop().unwrap();
                    // Write results to destination.
                    if let Some((dest, bbidx)) = dest {
                        self.write(dest, ptr);
                        self.bbidx = *bbidx;
                    }
                }
                Terminator::Return => break,
                t => todo!("{}", t),
            }
        }
    }

    /// Get the pointer to a Local.
    fn local_ptr(&self, local: &Local) -> *mut u8 {
        let offset = self.frame().offsets[usize::try_from(local.0).unwrap()];
        unsafe { self.frame().locals.add(offset) }
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

    /// Write some bytes to an IPlace. The amount of bytes is determined by the type of the
    /// destination.
    fn write(&mut self, dest: &IPlace, src: *const u8) {
        match dest {
            IPlace::Val {
                local: _,
                off: _,
                ty,
            }
            | IPlace::Indirect { ptr: _, off: _, ty } => {
                let size = usize::try_from(SIR.ty(ty).size()).unwrap();
                let ptr = self.iplace_to_ptr(dest);
                unsafe {
                    std::ptr::copy(src, ptr, size);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Implements the Store statement.
    fn store(&mut self, dest: &IPlace, src: &IPlace) {
        match src {
            IPlace::Val { .. } | IPlace::Indirect { .. } => {
                let src_ptr = self.iplace_to_ptr(src);
                self.write(dest, src_ptr);
            }
            IPlace::Const { val, ty } => {
                self.store_const(dest, val, ty);
            }
            _ => todo!(),
        }
    }

    /// Writes a constant to an IPlace.
    fn store_const(&mut self, dest: &IPlace, val: &Constant, _ty: &TypeId) {
        match val {
            Constant::Int(ci) => match ci {
                ConstantInt::UnsignedInt(ui) => match ui {
                    UnsignedInt::U8(v) => self.write(dest, [*v].as_ptr()),
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

    /// Creates a reference to an IPlace.
    fn mkref(&mut self, dest: &IPlace, src: &IPlace) {
        match dest {
            IPlace::Val { .. } | IPlace::Indirect { .. } => {
                // Get pointer to src.
                let src_ptr = self.iplace_to_ptr(src);
                let dst_ptr = self.iplace_to_ptr(dest);
                unsafe {
                    std::ptr::write::<*mut u8>(dst_ptr as *mut *mut u8, src_ptr);
                }
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SIRInterpreter;
    use yktrace::sir::SIR;

    fn interp(fname: &str, tio: *mut u8) {
        let body = SIR.body(fname).unwrap();
        let mut si = SIRInterpreter::new(&body.local_decls);
        // The raw pointer `tio` and the reference it was created from do not alias since we won't
        // be using the reference until the function `interpret` returns.
        si.set_trace_inputs(tio);
        unsafe {
            si.interpret(body);
        }
    }

    #[test]
    fn test_simple() {
        struct IO(u8, u8);
        #[no_mangle]
        fn simple(io: &mut IO) {
            let a = 3;
            io.1 = a;
        }
        let mut tio = IO(0, 0);
        interp("simple", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.1, 3);
    }

    #[test]
    fn test_tuple() {
        struct IO((u8, u8, u8));
        #[no_mangle]
        fn func_tuple(io: &mut IO) {
            let a = io.0;
            let b = a.2;
            (io.0).1 = b;
        }

        let mut tio = IO((1, 2, 3));
        interp("func_tuple", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.0, (1, 3, 3));
    }

    #[test]
    fn test_ref() {
        struct IO(u8, u8);
        #[no_mangle]
        fn func_ref(io: &mut IO) {
            let a = 5u8;
            let b = &a;
            io.1 = *b;
        }

        let mut tio = IO(5, 0);
        interp("func_ref", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.1, 5);
    }

    #[test]
    fn test_tupleref() {
        struct IO((u8, u8));
        #[no_mangle]
        fn func_tupleref(io: &mut IO) {
            let a = io.0;
            (io.0).1 = 5; // Make sure the line above copies.
            let b = &a;
            (io.0).0 = b.1;
        }

        let mut tio = IO((0, 3));
        interp("func_tupleref", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.0, (3, 5));
    }

    #[test]
    fn test_doubleref() {
        struct IO((u8, u8));
        #[no_mangle]
        fn func_doubleref(io: &mut IO) {
            let a = &io.0;
            (io.0).0 = a.1;
        }

        let mut tio = IO((0, 3));
        interp("func_doubleref", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.0, (3, 3));
    }

    #[test]
    fn test_call() {
        struct IO(u8, u8);

        fn foo() -> u8 {
            5
        }

        #[no_mangle]
        fn func_call(io: &mut IO) {
            let a = foo();
            io.0 = a;
        }

        let mut tio = IO(0, 0);
        interp("func_call", &mut tio as *mut _ as *mut u8);
        assert_eq!(tio.0, 5);
    }
}
