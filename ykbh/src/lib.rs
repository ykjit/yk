use ykpack::{self, Local, Terminator, Statement, IPlace, Constant, ConstantInt, UnsignedInt};
use yktrace::sir::SIR;
use std::convert::TryInto;

const ZST: Object = Object { data: Vec::new() };

#[derive(Clone, Debug)]
/// A data object, storing bytes.
pub struct Object {
    data: Vec<u8>
}

impl Object {

    /// Creates a new Object from a `usize`.
    fn from_usize(ptr: usize) -> Object {
        let data = ptr.to_ne_bytes();
        Object {
            data: Vec::from(data)
        }
    }

    /// Creates a new Object from a `u8`.
    fn from_u8(val: u8) -> Object {
        Object {
            data: Vec::from([val])
        }
    }

    /// Converts the objects data back to a `usize`.
    fn to_usize(&self) -> usize {
        usize::from_ne_bytes(self.data.as_slice().try_into().expect("Unexpected data length."))
    }

    /// Given a pointer and size, creates a new object by copying `size` data from `ptr`.
    fn from_ptr(ptr: usize, size: usize) -> Object {
        let mut data = Vec::with_capacity(size);
        unsafe {
            std::ptr::copy(ptr as *const u8, data.as_mut_ptr(), size);
            data.set_len(size);
        }
        Object {
            data
        }
    }
}

pub struct SIRInterpreter {
    vars: Vec<Option<Object>>,
}

impl SIRInterpreter {
    pub fn new(vars: Vec<Option<Object>>) -> Self {
        SIRInterpreter {
            vars,
        }
    }

    pub fn interpret(&mut self, body: &ykpack::Body) {
        // Ignore yktrace::trace_debug.
        if body.flags & ykpack::bodyflags::TRACE_DEBUG != 0 {
            return;
        }

        for stmt in body.blocks[0].stmts.iter() {
            match stmt {
                Statement::MkRef(dest, src) => self.mkref(dest, src),
                Statement::DynOffs { .. } => todo!("dynoffs"),
                Statement::Store(dest, src) => self.store(dest, src),
                Statement::BinaryOp { .. } => todo!("binop"),
                Statement::Nop => {},
                Statement::Unimplemented(_) | Statement::Debug(_) => { todo!("Unimplemented") },
                Statement::Cast(..) => todo!("cast"),
                Statement::Call(..) | Statement::StorageDead(_) => unreachable!()
            }
        }

        match &body.blocks[0].term {
            Terminator::Call {
                operand: _op,
                args: _args,
                destination: _dest
            } => {
                todo!()
            }
            Terminator::Return => {
            }
            t => todo!("{}", t)
        }
    }

    /// Get the Object referenced by a Local.
    fn get_var(&self, local: &Local) -> &Option<Object> {
        &self.vars[local.0 as usize]
    }

    /// Implements the Store statement.
    fn store(&mut self, dest: &IPlace, src: &IPlace) {
        self.store_obj(dest, self.to_obj(src));
    }

    /// Write an Object to an IPlace. This either creates a new Object if the IPlace is a Val, or
    /// writes the data to some pointer, if the IPlace is an Indirect.
    fn store_obj(&mut self, dest: &IPlace, src: Object) {
        match dest {
            IPlace::Val { local, off, ty: _ty } => {
                // Store into a local.
                if *off == 0 {
                    self.vars[local.0 as usize] = Some(src);
                } else {
                    todo!()
                }
            }
            IPlace::Indirect { ptr, off, ty: _ty } => {
                // Write to a pointer.
                if let Some(obj) = self.get_var(&ptr.local) {
                    let dstptr = obj.to_usize() + ptr.off as usize + *off as usize;
                    self.store_raw(dstptr, src);
                }
            }
            _ => unreachable!()
        };
    }

    fn store_raw(&self, dest: usize, src: Object) {
        unsafe {
            std::ptr::copy(src.data.as_ptr(), dest as *mut u8, src.data.len());
        }
    }

    fn to_obj(&self, src: &IPlace) -> Object {
        match src {
            IPlace::Val { local, off, ty } => {
                if *off == 0 {
                    // If the offset is 0 we can just copy the object, e.g. $1 = $2.
                    if let Some(obj) = self.get_var(local) {
                        obj.clone()
                    } else {
                        unreachable!()
                    }
                } else {
                    // Copy data from an offset, e.g. $1 = $2+2.
                    if let Some(obj) = self.get_var(local) {
                        let ptr = obj.data.as_ptr() as usize + *off as usize;
                        let size = SIR.ty(ty).size();
                        Object::from_ptr(ptr, size as usize)
                    } else {
                        unreachable!()
                    }
                }
            }
            IPlace::Indirect { ptr, off, ty } => {
                // Dereference a pointer and copy the data it points to.
                if let Some(obj) = self.get_var(&ptr.local) {
                    let dstptr = obj.to_usize() + ptr.off as usize + *off as usize;
                    let size = SIR.ty(ty).size();
                    Object::from_ptr(dstptr, size as usize)
                } else {
                    unreachable!()
                }
            }
            IPlace::Const { val, ty: _ty } => {
                // The source is a constant: create a new object.
                match val {
                    Constant::Int(ci) => match ci {
                        ConstantInt::UnsignedInt(ui) => {
                            match ui {
                                UnsignedInt::U8(v) => Object::from_u8(*v),
                                _ => todo!()
                            }
                        }
                        ConstantInt::SignedInt(_si) => todo!(),
                    },
                    Constant::Bool(_b) => todo!(),
                    Constant::Tuple(t) => {
                        if SIR.ty(t).size() == 0 {
                            ZST
                        } else {
                            todo!()
                        }
                    },
                    _ => todo!(),
                }
            }
            _ => todo!()
        }
    }


    /// Creates a reference to an IPlace.
    fn mkref(&mut self, dest: &IPlace, src: &IPlace) {
        let obj = match src {
            IPlace::Val { local, off, ty: _ty } => {
                if let Some(obj) = self.get_var(local) {
                    // The pointer to the vector should remain valid, since no changes to the
                    // vector later on will change its size, and so it won't be reallocated.
                    // FIXME: Check this is true!
                    let ptr = obj.data.as_ptr() as usize + *off as usize;
                    Object::from_usize(ptr)
                } else {
                    unreachable!()
                }
            }
            IPlace::Indirect { ptr, off, ty: _ty } => {
                if let Some(obj) = self.get_var(&ptr.local) {
                    let ptr = obj.to_usize() + ptr.off as usize + *off as usize;
                    Object::from_usize(ptr)
                } else {
                    unreachable!()
                }
            }
            _ => todo!()
        };
        self.store_obj(dest, obj);
    }
}

#[cfg(test)]
mod tests {
    use super::{SIRInterpreter, Object};
    use yktrace::sir::SIR;

    fn interp(fname: &str, tio: usize) {
        let body = match SIR.bodies.get(fname) {
            Some(b) => b,
            None => panic!("No SIR")
        };
        let mut vars: Vec<Option<Object>> = vec![None; body.local_decls.len()];
        vars[1] = Some(Object::from_usize(tio));
        let mut si = SIRInterpreter::new(vars);
        si.interpret(body);
    }

    #[test]
    fn test_simple() {
        struct IO(u8, u8);
        #[no_mangle]
        fn simple(io: &mut IO) {
            let a = 3;
            io.1 = a;
        }
        let tio = IO(0, 0);
        interp("simple", &tio as *const _ as usize);
        assert_eq!(tio.1, 3);
    }

    #[test]
    fn test_tuple() {
        struct IO((u8, u8, u8));
        #[no_mangle]
        fn func_tuple(io: &mut IO) {
            let a = io.0;
            let b = a.2;
            io.0.1 = b;
        }

        let tio = IO((1,2,3));
        interp("func_tuple", &tio as *const _ as usize);
        assert_eq!(tio.0, (1,3,3));
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

        let tio = IO(5, 0);
        interp("func_ref", &tio as *const _ as usize);
        assert_eq!(tio.1, 5);
    }

    #[test]
    fn test_tupleref() {
        struct IO((u8, u8));
        #[no_mangle]
        fn func_tupleref(io: &mut IO) {
            let a = io.0;
            io.0.1 = 5; // Make sure the line above copies.
            let b = &a;
            io.0.0 = b.1;
        }

        let tio = IO((0, 3));
        interp("func_tupleref", &tio as *const _ as usize);
        assert_eq!(tio.0, (3, 5));
    }

    #[test]
    fn test_doubleref() {
        struct IO((u8, u8));
        #[no_mangle]
        fn func_doubleref(io: &mut IO) {
            let a = &io.0;
            io.0.0 = a.1;
        }

        let tio = IO((0, 3));
        interp("func_doubleref", &tio as *const _ as usize);
        assert_eq!(tio.0, (3, 3));
    }
}
