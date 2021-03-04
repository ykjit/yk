//! Code generation for stores.

use super::{IndirectLoc, Location, RegAndOffset, TraceCompiler, TEMP_REG};
use dynasmrt::{dynasm, DynasmApi};
use ykpack::IRPlace;
use yktrace::sir::SIR;

impl TraceCompiler {
    /// Store the value in `src_loc` into `dst_loc`.
    pub(crate) fn store(&mut self, dst_ip: &IRPlace, src_ip: &IRPlace) {
        let dst_loc = self.iplace_to_location(dst_ip);
        let src_loc = self.iplace_to_location(src_ip);
        debug_assert!(SIR.ty(&dst_ip.ty()).size() == SIR.ty(&src_ip.ty()).size());
        self.store_raw(&dst_loc, &src_loc, SIR.ty(&dst_ip.ty()).size());
    }

    /// Stores src_loc into dst_loc.
    pub(crate) fn store_raw(&mut self, dst_loc: &Location, src_loc: &Location, size: u64) {
        // This is the one place in the compiler where we allow an explosion of cases over the
        // variants of `Location`. If elsewhere you find yourself matching over a pair of locations
        // you should try and re-work you code so it calls this.
        //
        // FIXME avoid partial register stalls.
        // FIXME this is massive. Move this (and store() to a new file).
        // FIXME constants are assumed to fit in a 64-bit register.

        /// Break a 64-bit value down into two 32-bit values. Used in scenarios where the X86_64
        /// ISA doesn't allow 64-bit constant encodings.
        fn split_i64(v: i64) -> (i32, i32) {
            ((v >> 32) as i32, (v & 0xffffffff) as i32)
        }

        // This can happen due to ZSTs.
        if size == 0 {
            return;
        }

        match (&dst_loc, &src_loc) {
            (Location::Reg(dst_reg), Location::Reg(src_reg)) => {
                dynasm!(self.asm
                    ; mov Rq(dst_reg), Rq(src_reg)
                );
            }
            (Location::Mem(dst_ro), Location::Reg(src_reg)) => match size {
                1 => dynasm!(self.asm
                    ; mov BYTE [Rq(dst_ro.reg) + dst_ro.off], Rb(src_reg)
                ),
                2 => dynasm!(self.asm
                    ; mov WORD [Rq(dst_ro.reg) + dst_ro.off], Rw(src_reg)
                ),
                4 => dynasm!(self.asm
                    ; mov DWORD [Rq(dst_ro.reg) + dst_ro.off], Rd(src_reg)
                ),
                8 => dynasm!(self.asm
                    ; mov QWORD [Rq(dst_ro.reg) + dst_ro.off], Rq(src_reg)
                ),
                _ => unreachable!(),
            },
            (Location::Mem(dst_ro), Location::Mem(src_ro)) => {
                if size <= 8 {
                    debug_assert!(dst_ro.reg != *TEMP_REG);
                    debug_assert!(src_ro.reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rb(*TEMP_REG), BYTE [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dst_ro.reg) + dst_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rw(*TEMP_REG), WORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dst_ro.reg) + dst_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rd(*TEMP_REG), DWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dst_ro.reg) + dst_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dst_ro.reg) + dst_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    self.copy_memory(dst_ro, src_ro, size);
                }
            }
            (Location::Reg(dst_reg), Location::Mem(src_ro)) => match size {
                1 => dynasm!(self.asm
                    ; mov Rb(dst_reg), BYTE [Rq(src_ro.reg) + src_ro.off]
                ),
                2 => dynasm!(self.asm
                    ; mov Rw(dst_reg), WORD [Rq(src_ro.reg) + src_ro.off]
                ),
                4 => dynasm!(self.asm
                    ; mov Rd(dst_reg), DWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                8 => dynasm!(self.asm
                    ; mov Rq(dst_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                _ => unreachable!(),
            },
            (Location::Reg(dst_reg), Location::Const { val: c_val, .. }) => {
                let i64_c = c_val.i64_cast();
                dynasm!(self.asm
                    ; mov Rq(dst_reg), QWORD i64_c
                );
            }
            (Location::Mem(ro), Location::Const { val: c_val, ty }) => {
                let c_i64 = c_val.i64_cast();
                match SIR.ty(&ty).size() {
                    1 => dynasm!(self.asm
                        ; mov BYTE [Rq(ro.reg) + ro.off], c_i64 as i8
                    ),
                    2 => dynasm!(self.asm
                        ; mov WORD [Rq(ro.reg) + ro.off], c_i64 as i16
                    ),
                    4 => dynasm!(self.asm
                        ; mov DWORD [Rq(ro.reg) + ro.off], c_i64 as i32
                    ),
                    8 => {
                        let (hi, lo) = split_i64(c_i64);
                        dynasm!(self.asm
                            ; mov DWORD [Rq(ro.reg) + ro.off], lo as i32
                            ; mov DWORD [Rq(ro.reg) + ro.off + 4], hi as i32
                        );
                    }
                    _ => todo!(),
                }
            }
            (
                Location::Reg(dst_reg),
                Location::Indirect {
                    ptr: src_indloc,
                    off: src_off,
                },
            ) => match src_indloc {
                IndirectLoc::Reg(src_reg) => match size {
                    1 => dynasm!(self.asm
                        ; mov Rb(dst_reg), BYTE [Rq(src_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                        ; mov Rw(dst_reg), WORD [Rq(src_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                        ; mov Rd(dst_reg), DWORD [Rq(src_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                        ; mov Rq(dst_reg), QWORD [Rq(src_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(src_ro) => match size {
                    1 => dynasm!(self.asm
                        ; mov Rq(dst_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rb(dst_reg), BYTE [Rq(dst_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                        ; mov Rq(dst_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rw(dst_reg), WORD [Rq(dst_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                        ; mov Rq(dst_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rd(dst_reg), DWORD [Rq(dst_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                        ; mov Rq(dst_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rq(dst_reg), QWORD [Rq(dst_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
            },
            (
                Location::Indirect {
                    ptr: dst_indloc,
                    off: dst_off,
                },
                Location::Const { val: src_cval, .. },
            ) => {
                let src_i64 = src_cval.i64_cast();
                match dst_indloc {
                    IndirectLoc::Reg(dst_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov BYTE [Rq(dst_reg) + *dst_off], src_i64 as i8
                        ),
                        2 => dynasm!(self.asm
                            ; mov WORD [Rq(dst_reg) + *dst_off], src_i64 as i16
                        ),
                        4 => dynasm!(self.asm
                            ; mov DWORD [Rq(dst_reg) + *dst_off], src_i64 as i32
                        ),
                        8 => {
                            let (hi, lo) = split_i64(src_i64);
                            dynasm!(self.asm
                                ; mov DWORD [Rq(dst_reg) + *dst_off], lo as i32
                                ; mov DWORD [Rq(dst_reg) + *dst_off + 4], hi as i32
                            );
                        }
                        _ => todo!(),
                    },
                    IndirectLoc::Mem(dst_ro) => {
                        debug_assert!(dst_ro.reg != *TEMP_REG);
                        match size {
                            1 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                                    ; mov BYTE [Rq(*TEMP_REG) + *dst_off], src_i64 as i8
                                );
                            }
                            2 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                                    ; mov WORD [Rq(*TEMP_REG) + *dst_off], src_i64 as i16
                                );
                            }
                            4 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dst_off], src_i64 as i32
                                );
                            }
                            8 => {
                                let (hi, lo) = split_i64(src_i64);
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dst_off], lo as i32
                                    ; mov DWORD [Rq(*TEMP_REG) + *dst_off + 4], hi as i32
                                );
                            }
                            _ => todo!(),
                        }
                    }
                }
            }
            (
                Location::Indirect {
                    ptr: dst_indloc,
                    off: dst_off,
                },
                Location::Reg(src_reg),
            ) => match dst_indloc {
                IndirectLoc::Reg(dst_reg) => match size {
                    1 => dynasm!(self.asm
                        ; mov BYTE [Rq(dst_reg) + *dst_off], Rb(src_reg)
                    ),
                    2 => dynasm!(self.asm
                        ; mov WORD [Rq(dst_reg) + *dst_off], Rw(src_reg)
                    ),
                    4 => dynasm!(self.asm
                        ; mov DWORD [Rq(dst_reg) + *dst_off], Rd(src_reg)
                    ),
                    8 => dynasm!(self.asm
                        ; mov QWORD [Rq(dst_reg) + *dst_off], Rq(src_reg)
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(dst_ro) => {
                    debug_assert!(*src_reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov BYTE [Rq(*TEMP_REG) + *dst_off], Rb(src_reg)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov WORD [Rq(*TEMP_REG) + *dst_off], Rw(src_reg)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov DWORD [Rq(*TEMP_REG) + *dst_off], Rd(src_reg)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov QWORD [Rq(*TEMP_REG) + *dst_off], Rq(src_reg)
                        ),
                        _ => todo!(),
                    }
                }
            },
            (
                Location::Mem(dst_ro),
                Location::Indirect {
                    ptr: src_ind,
                    off: src_off,
                },
            ) => match src_ind {
                IndirectLoc::Mem(src_ro) => {
                    debug_assert!(src_ro.reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov BYTE [Rq(dst_ro.reg) + dst_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov WORD [Rq(dst_ro.reg) + dst_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov DWORD [Rq(dst_ro.reg) + dst_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov QWORD [Rq(dst_ro.reg) + dst_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => todo!(),
                    }
                }
                IndirectLoc::Reg(src_reg) => {
                    debug_assert!(*src_reg != *TEMP_REG);
                    match size {
                        1 | 2 | 4 => todo!(),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_reg) + *src_off]
                            ; mov QWORD [Rq(dst_ro.reg) + dst_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => {
                            let src_ro = RegAndOffset {
                                reg: *src_reg,
                                off: *src_off,
                            };
                            self.copy_memory(&dst_ro, &src_ro, size);
                        }
                    }
                }
            },
            (
                Location::Indirect {
                    ptr: dst_ind,
                    off: dst_off,
                },
                Location::Mem(src_ro),
            ) => {
                debug_assert!(src_ro.reg != *TEMP_REG);
                match dst_ind {
                    IndirectLoc::Reg(dst_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dst_reg) + *dst_off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dst_reg) + *dst_off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dst_reg) + *dst_off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dst_reg) + *dst_off], Rq(*TEMP_REG)
                        ),
                        _ => {
                            let dst_ro = RegAndOffset {
                                reg: *dst_reg,
                                off: 0,
                            };
                            self.copy_memory(&dst_ro, src_ro, size);
                        }
                    },
                    IndirectLoc::Mem(dst_ro) => match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov BYTE [Rq(src_ro.reg) + *dst_off], Rb(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov WORD [Rq(src_ro.reg) + *dst_off], Rw(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov DWORD [Rq(src_ro.reg) + *dst_off], Rd(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dst_ro.reg) + dst_ro.off]
                            ; mov QWORD [Rq(src_ro.reg) + *dst_off], Rq(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        _ => todo!(),
                    },
                }
            }
            _ => todo!(),
        }
    }
}
