//! The stopgap interpreter.
//!
//! After a guard failure, the StopgapInterpreter takes over. It interprets SIR to execute the
//! program from the guard failure until execution arrives back to the control point, at which
//! point the normal interpreter can continue.
//!
//! In other systems, this process is sometimes called "blackholing".
//!
//! Tests for this module are in ../tests/src/stopgap/.

use std::alloc::{alloc, dealloc, Layout};
use std::convert::TryFrom;
use std::sync::Arc;
use ykpack::{
    self, BinOp, Body, CallOperand, Constant, ConstantInt, IRPlace, Local, Statement, Terminator,
    TyKind, UnsignedInt, UnsignedIntTy,
};
use yktrace::sir::{RETURN_LOCAL, SIR};

/// When the StopgapInterpreter is called after a guard failure, it is passed a `Vec` of
/// `IncomingFrame`s from the machine code. These are then converted into other structures more
/// suitable for interpretation by the StopgapInterpreter.
pub struct IncomingFrame {
    /// The body of this frame.
    body: Arc<Body>,
    /// Index of the current basic block we are in. When returning from a function call, the
    /// terminator of this block is were we continue.
    bbidx: usize,
    /// Pointer to memory containing the live variables.
    locals: *mut u8,
}

impl IncomingFrame {
    pub fn new(body: Arc<Body>, bbidx: usize, locals: *mut u8) -> Self {
        Self {
            body,
            bbidx,
            locals,
        }
    }
}

/// An interpreter function frame representing the current state of an executing function.
#[derive(Debug)]
struct Frame {
    /// This frame's local variables.
    locals: *mut u8,
    /// This frame's program counter (which always increments in terms of basic blocks).
    pc: ykpack::BasicBlockIndex,
    /// Body of this stack frame.
    body: Arc<Body>,
}

impl Frame {
    /// Write a constant to the pointer `dst`.
    unsafe fn write_const(&mut self, dst: *mut u8, constant: &Constant) {
        match constant {
            Constant::Int(ci) => match ci {
                ConstantInt::UnsignedInt(ui) => match ui {
                    UnsignedInt::U8(v) => std::ptr::copy(v, dst, 1),
                    UnsignedInt::Usize(v) => std::ptr::copy(v, dst as *mut usize, 1),
                    _ => todo!(),
                },
                ConstantInt::SignedInt(_) => todo!(),
            },
            Constant::Bool(b) => std::ptr::copy(b as *const bool, dst as *mut bool, 1),
            Constant::Tuple(t) if SIR.ty(t).size() == 0 => (), // ZST: do nothing
            _ => todo!(),
        }
    }

    /// Stores one IRPlace into another.
    unsafe fn store(&mut self, dst: &IRPlace, src: &IRPlace) {
        match src {
            IRPlace::Val { .. } | IRPlace::Indirect { .. } => {
                let src_ptr = self.irplace_to_ptr(src);
                let dst_ptr = self.irplace_to_ptr(dst);
                let size = usize::try_from(SIR.ty(&src.ty()).size()).unwrap();
                std::ptr::copy(src_ptr, dst_ptr, size);
            }
            IRPlace::Const { val, ty: _ty } => {
                let dst_ptr = self.irplace_to_ptr(dst);
                self.write_const(dst_ptr, val);
            }
            _ => todo!(),
        }
    }

    /// Copy over the call arguments from another frame.
    unsafe fn copy_args(&mut self, args: &[IRPlace], oframe: &Frame) {
        for (i, arg) in args.iter().enumerate() {
            let dst = self.local_ptr(&Local(u32::try_from(i + 1).unwrap()));
            match arg {
                IRPlace::Val { .. } | IRPlace::Indirect { .. } => {
                    let src = oframe.irplace_to_ptr(arg);
                    let size = usize::try_from(SIR.ty(&arg.ty()).size()).unwrap();
                    std::ptr::copy(src, dst, size);
                }
                IRPlace::Const { val, .. } => {
                    self.write_const(dst, val);
                }
                _ => unreachable!(),
            }
        }
    }

    /// Get the pointer to a Local.
    fn local_ptr(&self, local: &Local) -> *mut u8 {
        let offset = self.body.offsets()[usize::try_from(local.0).unwrap()];
        unsafe { self.locals.add(offset) }
    }

    /// Get the pointer for an IRPlace, while applying all offsets.
    fn irplace_to_ptr(&self, place: &IRPlace) -> *mut u8 {
        match place {
            IRPlace::Val {
                local,
                off,
                ty: _ty,
            } => {
                // Get a pointer to the Val.
                let dst_ptr = self.local_ptr(&local);
                unsafe { dst_ptr.add(usize::try_from(*off).unwrap()) }
            }
            IRPlace::Indirect { ptr, off, ty: _ty } => {
                // Get a pointer to the Indirect, which itself points to another pointer.
                let dst_ptr = self.local_ptr(&ptr.local()) as *mut *mut u8;
                unsafe {
                    // Dereference the pointer, by reading its value.
                    let mut p = std::ptr::read::<*mut u8>(dst_ptr);
                    // Add the offsets of the Indirect.
                    p = p.offset(isize::try_from(ptr.off()).unwrap());
                    p.offset(isize::try_from(*off).unwrap())
                }
            }
            _ => unreachable!(),
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let layout = self.body.layout();
        unsafe {
            dealloc(
                self.locals,
                Layout::from_size_align_unchecked(layout.0, layout.1),
            );
        }
    }
}

/// The SIR interpreter, also known as stopgap interpreter, is invoked when a guard fails in a
/// trace. It is initalised with information from the trace, e.g. live variables, stack frames, and
/// then run to get us back to a control point from which point the normal interpreter can take
/// over.
pub struct StopgapInterpreter {
    /// Active stack frames (most recent last).
    frames: Vec<Frame>,
    /// Value to be returned by the interpreter.
    rv: bool,
}

impl StopgapInterpreter {
    /// Initialise the interpreter from a symbol name.
    #[cfg(feature = "testing")]
    pub fn from_symbol(sym: String) -> Self {
        let frame = StopgapInterpreter::create_frame(&sym);
        StopgapInterpreter {
            frames: vec![frame],
            rv: true,
        }
    }

    /// Initialise the interpreter from a vector of `IncomingFrame`s. Each contains information about
    /// live variables and stack frames, received from a failing guard.
    pub fn from_frames(v: Vec<IncomingFrame>) -> Self {
        let mut frames = Vec::new();
        for fi in v.into_iter() {
            let frame = Frame {
                locals: fi.locals,
                pc: u32::try_from(fi.bbidx).unwrap(),
                body: Arc::clone(&fi.body),
            };
            frames.push(frame);
        }
        let mut sg = StopgapInterpreter { frames, rv: true };
        let frame = sg.peek_frame();
        // Since we start in the block where the guard failed, we immediately skip to the
        // terminator and interpret it to initialise the block where actual interpretation needs to
        // start.
        let body = frame.body.clone();
        let pc = usize::try_from(frame.pc).unwrap();
        unsafe {
            sg.terminator(&body.blocks()[pc].term());
        }
        sg
    }

    // Return a reference to the most recent frame on the stack.
    fn peek_frame(&self) -> &Frame {
        &self.frames.last().unwrap()
    }

    // Return a mutable reference to the most recent frame on the stack.
    fn peek_frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().unwrap()
    }

    /// Given the symbol name of a function, generate a `Frame` which allocates the precise
    /// amount of memory required by the locals used in that function.
    fn create_frame(sym: &str) -> Frame {
        let body = SIR.body(&sym).unwrap();
        let (size, align) = body.layout();
        let layout = Layout::from_size_align(size, align).unwrap();
        let locals = unsafe { alloc(layout) };
        Frame {
            locals,
            pc: 0,
            body,
        }
    }

    /// Inserts a pointer to the interpreter context into the `interp_step` frame.
    #[cfg(feature = "testing")]
    pub unsafe fn set_interp_ctx(&mut self, ctx: *mut u8) {
        // The interpreter context lives in $1
        let ptr = self
            .frames
            .first()
            .unwrap()
            .local_ptr(&yktrace::sir::INTERP_STEP_ARG);
        std::ptr::write::<*mut u8>(ptr as *mut *mut u8, ctx);
    }

    pub unsafe fn interpret(&mut self) -> bool {
        while let Some(frame) = self.frames.last() {
            let body = frame.body.clone();
            let block = &body.blocks()[usize::try_from(frame.pc).unwrap()];
            for stmt in block.stmts().iter() {
                match stmt {
                    Statement::MkRef(dst, src) => self.mkref(&dst, &src),
                    Statement::DynOffs { .. } => todo!(),
                    Statement::Store(dst, src) => self.store(&dst, &src),
                    Statement::BinaryOp {
                        dst,
                        op,
                        opnd1,
                        opnd2,
                        checked,
                    } => self.binop(dst, op, opnd1, opnd2, *checked),
                    Statement::Nop => {}
                    Statement::Unimplemented(_) | Statement::Debug(_) => todo!(),
                    Statement::Cast(..) => todo!(),
                    Statement::StorageLive(_) | Statement::StorageDead(_) => {}
                    Statement::Call(..) | Statement::LoopStart | Statement::LoopEnd => {
                        unreachable!()
                    }
                }
            }
            self.terminator(block.term());
        }
        self.rv
    }

    unsafe fn terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Call {
                operand: op,
                args,
                destination: _dst,
            } => {
                let fname = if let CallOperand::Fn(sym) = op {
                    sym
                } else {
                    todo!("unknown call target");
                };

                // Initialise the new stack frame.
                let oldframe = self.peek_frame();
                let mut newframe = StopgapInterpreter::create_frame(&fname);
                newframe.copy_args(args, oldframe);
                self.frames.push(newframe);
            }
            Terminator::Return => {
                let oldframe = self.frames.pop().unwrap();
                // If there are no more frames left, we are returning from the `interp_step`
                // function, which means we have reached the control point and are done here.
                if let Some(curframe) = self.frames.last_mut() {
                    let pc = usize::try_from(curframe.pc).unwrap();
                    let body = &curframe.body;
                    // Check the previous frame's call terminator to find out where we have to go
                    // next.
                    let (dst, pc) = match body.blocks()[pc].term() {
                        Terminator::Call {
                            operand: _,
                            args: _,
                            destination: dst,
                        } => dst.as_ref().map(|(p, b)| (p.clone(), *b)).unwrap(),
                        _ => unreachable!(),
                    };
                    // Get a pointer to the return value of the called frame.
                    let ret_ptr = oldframe.local_ptr(&RETURN_LOCAL);
                    // Write the return value to the destination in the previous frame.
                    let dst_ptr = curframe.irplace_to_ptr(&dst);
                    let size = usize::try_from(SIR.ty(&dst.ty()).size()).unwrap();
                    std::ptr::copy(ret_ptr, dst_ptr, size);
                    curframe.pc = pc;
                } else {
                    // The return value of `interp_step` tells the meta-tracer whether to stop or
                    // continue running the interpreter.
                    let ret_ptr = oldframe.local_ptr(&RETURN_LOCAL);
                    self.rv = std::ptr::read::<bool>(ret_ptr as *const bool);
                }
            }
            Terminator::SwitchInt {
                discr,
                values,
                target_bbs,
                otherwise_bb,
            } => {
                let val = self.read_int(discr);
                let frame = self.peek_frame_mut();
                frame.pc = *otherwise_bb;
                for (i, v) in values.iter().enumerate() {
                    if val == *v {
                        frame.pc = target_bbs[i];
                        break;
                    }
                }
            }
            Terminator::Goto(bb) => {
                self.peek_frame_mut().pc = *bb;
            }
            Terminator::Assert {
                cond,
                expected,
                target_bb,
            } => {
                let b = self.read_int(cond) == 1;
                if b != *expected {
                    todo!() // FIXME raise error
                }
                self.peek_frame_mut().pc = *target_bb;
            }
            t => todo!("{}", t),
        }
    }

    fn read_int(&self, src: &IRPlace) -> u128 {
        if let IRPlace::Const { val, ty: _ty } = src {
            let val = match val {
                Constant::Int(ci) => match ci {
                    ConstantInt::UnsignedInt(ui) => match ui {
                        UnsignedInt::U8(v) => u128::try_from(*v).unwrap(),
                        _ => todo!(),
                    },
                    ConstantInt::SignedInt(_si) => todo!(),
                },
                _ => todo!(),
            };
            return val;
        }
        let ptr = self.peek_frame().irplace_to_ptr(src);
        match &SIR.ty(&src.ty()).kind() {
            TyKind::UnsignedInt(ui) => match ui {
                UnsignedIntTy::Usize => todo!(),
                UnsignedIntTy::U8 => unsafe { u128::from(std::ptr::read::<u8>(ptr)) },
                UnsignedIntTy::U16 => todo!(),
                UnsignedIntTy::U32 => todo!(),
                UnsignedIntTy::U64 => todo!(),
                UnsignedIntTy::U128 => todo!(),
            },
            TyKind::SignedInt(_) => unreachable!(),
            TyKind::Bool => unsafe { u128::from(std::ptr::read::<u8>(ptr)) },
            _ => unreachable!(),
        }
    }

    /// Store the IRPlace src in the IRPlace dst in the current frame.
    unsafe fn store(&mut self, dst: &IRPlace, src: &IRPlace) {
        self.peek_frame_mut().store(dst, src);
    }

    /// Creates a reference to an IRPlace, e.g. `dst = &src`.
    fn mkref(&mut self, dst: &IRPlace, src: &IRPlace) {
        match dst {
            IRPlace::Val { .. } | IRPlace::Indirect { .. } => {
                // Get pointer to src.
                let frame = &self.peek_frame_mut();
                let src_ptr = frame.irplace_to_ptr(src);
                let dst_ptr = frame.irplace_to_ptr(dst);
                unsafe {
                    std::ptr::write::<*mut u8>(dst_ptr as *mut *mut u8, src_ptr);
                }
            }
            _ => unreachable!(),
        }
    }

    fn binop(
        &mut self,
        dst: &IRPlace,
        op: &ykpack::BinOp,
        opnd1: &IRPlace,
        opnd2: &IRPlace,
        checked: bool,
    ) {
        let ty = SIR.ty(&opnd1.ty());
        if !ty.is_int() {
            todo!("binops for non-integers");
        }

        match &ty.kind() {
            TyKind::UnsignedInt(ui) => match ui {
                UnsignedIntTy::U8 => self.binop_u8(dst, op, opnd1, opnd2, checked),
                UnsignedIntTy::U16 => todo!(),
                UnsignedIntTy::U32 => todo!(),
                UnsignedIntTy::U64 => todo!(),
                UnsignedIntTy::U128 => todo!(),
                UnsignedIntTy::Usize => todo!(),
            },
            TyKind::SignedInt(_si) => todo!(),
            TyKind::Bool => unreachable!(),
            e => unreachable!("{:?}", e),
        }
    }
}

/// Binary operations macros.
macro_rules! make_binop {
    ($name: ident, $type: ident) => {
        fn $name(
            &mut self,
            dst: &IRPlace,
            op: &BinOp,
            opnd1: &IRPlace,
            opnd2: &IRPlace,
            checked: bool,
        ) {
            let a = $type::try_from(self.read_int(opnd1)).unwrap();
            let b = $type::try_from(self.read_int(opnd2)).unwrap();
            let frame = self.peek_frame_mut();
            let ptr = frame.irplace_to_ptr(dst);
            let (v, of) = match op {
                BinOp::Add => a.overflowing_add(b),
                BinOp::Lt => ($type::from(a < b), false),
                BinOp::Gt => ($type::from(a > b), false),
                _ => todo!(),
            };
            if checked {
                // Write overflow result into result tuple.
                let ty = SIR.ty(&dst.ty());
                let tty = ty.unwrap_tuple();
                let flag_off = isize::try_from(tty.fields().offset(1)).unwrap();
                unsafe {
                    std::ptr::write::<u8>(ptr.offset(flag_off), u8::from(of));
                }
            } else if of {
                todo!("Raise error.")
            }
            let bytes = v.to_ne_bytes();
            unsafe {
                std::ptr::copy(bytes.as_ptr(), ptr, bytes.len());
            }
        }
    };
}

impl StopgapInterpreter {
    make_binop!(binop_u8, u8);
}
