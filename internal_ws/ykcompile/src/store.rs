//! Code generation for stores.

use crate::{IndirectLoc, Location, RegAndOffset, TraceCompiler, TEMP_REG};
use dynasmrt::DynasmApi;
use ykpack::IPlace;
use yktrace::sir::SIR;

impl TraceCompiler {
    /// Store the value in `src_loc` into `dest_loc`.
    pub(crate) fn store(&mut self, dest_ip: &IPlace, src_ip: &IPlace) {
        let dest_loc = self.iplace_to_location(dest_ip);
        let src_loc = self.iplace_to_location(src_ip);
        debug_assert!(SIR.ty(&dest_ip.ty()).size() == SIR.ty(&src_ip.ty()).size());
        self.store_raw(&dest_loc, &src_loc, SIR.ty(&dest_ip.ty()).size());
    }

    /// Stores src_loc into dest_loc.
    pub(crate) fn store_raw(&mut self, dest_loc: &Location, src_loc: &Location, size: u64) {
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

        match (&dest_loc, &src_loc) {
            (Location::Reg(dest_reg), Location::Reg(src_reg)) => {
                dynasm!(self.asm
                    ; mov Rq(dest_reg), Rq(src_reg)
                );
            }
            (Location::Mem(dest_ro), Location::Reg(src_reg)) => match size {
                1 => dynasm!(self.asm
                    ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(src_reg)
                ),
                2 => dynasm!(self.asm
                    ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(src_reg)
                ),
                4 => dynasm!(self.asm
                    ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(src_reg)
                ),
                8 => dynasm!(self.asm
                    ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(src_reg)
                ),
                _ => unreachable!(),
            },
            (Location::Mem(dest_ro), Location::Mem(src_ro)) => {
                if size <= 8 {
                    debug_assert!(dest_ro.reg != *TEMP_REG);
                    debug_assert!(src_ro.reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rb(*TEMP_REG), BYTE [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rw(*TEMP_REG), WORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rd(*TEMP_REG), DWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    self.copy_memory(dest_ro, src_ro, size);
                }
            }
            (Location::Reg(dest_reg), Location::Mem(src_ro)) => match size {
                1 => dynasm!(self.asm
                    ; mov Rb(dest_reg), BYTE [Rq(src_ro.reg) + src_ro.off]
                ),
                2 => dynasm!(self.asm
                    ; mov Rw(dest_reg), WORD [Rq(src_ro.reg) + src_ro.off]
                ),
                4 => dynasm!(self.asm
                    ; mov Rd(dest_reg), DWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                8 => dynasm!(self.asm
                    ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                _ => unreachable!(),
            },
            (Location::Reg(dest_reg), Location::Const { val: c_val, .. }) => {
                let i64_c = c_val.i64_cast();
                if i64_c <= i64::from(u32::MAX) {
                    dynasm!(self.asm
                        ; mov Rq(dest_reg), i64_c as i32
                    );
                } else {
                    // Can't move 64-bit constants in x86_64.
                    let i64_c = c_val.i64_cast();
                    let hi_word = (i64_c >> 32) as i32;
                    let lo_word = (i64_c & 0xffffffff) as i32;
                    dynasm!(self.asm
                        ; mov Rq(dest_reg), hi_word
                        ; shl Rq(dest_reg), 32
                        ; or Rq(dest_reg), lo_word
                    );
                }
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
                Location::Reg(dest_reg),
                Location::Indirect {
                    ptr: src_indloc,
                    off: src_off,
                },
            ) => match src_indloc {
                IndirectLoc::Reg(src_reg) => match size {
                    1 => dynasm!(self.asm
                        ; mov Rb(dest_reg), BYTE [Rq(src_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                        ; mov Rw(dest_reg), WORD [Rq(src_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                        ; mov Rd(dest_reg), DWORD [Rq(src_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(src_ro) => match size {
                    1 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rb(dest_reg), BYTE [Rq(dest_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rw(dest_reg), WORD [Rq(dest_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rd(dest_reg), DWORD [Rq(dest_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rq(dest_reg), QWORD [Rq(dest_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
            },
            (
                Location::Indirect {
                    ptr: dest_indloc,
                    off: dest_off,
                },
                Location::Const { val: src_cval, .. },
            ) => {
                let src_i64 = src_cval.i64_cast();
                match dest_indloc {
                    IndirectLoc::Reg(dest_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov BYTE [Rq(dest_reg) + *dest_off], src_i64 as i8
                        ),
                        2 => dynasm!(self.asm
                            ; mov WORD [Rq(dest_reg) + *dest_off], src_i64 as i16
                        ),
                        4 => dynasm!(self.asm
                            ; mov DWORD [Rq(dest_reg) + *dest_off], src_i64 as i32
                        ),
                        8 => {
                            let (hi, lo) = split_i64(src_i64);
                            dynasm!(self.asm
                                ; mov DWORD [Rq(dest_reg) + *dest_off], lo as i32
                                ; mov DWORD [Rq(dest_reg) + *dest_off + 4], hi as i32
                            );
                        }
                        _ => todo!(),
                    },
                    IndirectLoc::Mem(dest_ro) => {
                        debug_assert!(dest_ro.reg != *TEMP_REG);
                        match size {
                            1 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov BYTE [Rq(*TEMP_REG) + *dest_off], src_i64 as i8
                                );
                            }
                            2 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov WORD [Rq(*TEMP_REG) + *dest_off], src_i64 as i16
                                );
                            }
                            4 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off], src_i64 as i32
                                );
                            }
                            8 => {
                                let (hi, lo) = split_i64(src_i64);
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off], lo as i32
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off + 4], hi as i32
                                );
                            }
                            _ => todo!(),
                        }
                    }
                }
            }
            (
                Location::Indirect {
                    ptr: dest_indloc,
                    off: dest_off,
                },
                Location::Reg(src_reg),
            ) => match dest_indloc {
                IndirectLoc::Reg(dest_reg) => match size {
                    1 => dynasm!(self.asm
                        ; mov BYTE [Rq(dest_reg) + *dest_off], Rb(src_reg)
                    ),
                    2 => dynasm!(self.asm
                        ; mov WORD [Rq(dest_reg) + *dest_off], Rw(src_reg)
                    ),
                    4 => dynasm!(self.asm
                        ; mov DWORD [Rq(dest_reg) + *dest_off], Rd(src_reg)
                    ),
                    8 => dynasm!(self.asm
                        ; mov QWORD [Rq(dest_reg) + *dest_off], Rq(src_reg)
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(dest_ro) => {
                    debug_assert!(*src_reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov BYTE [Rq(*TEMP_REG) + *dest_off], Rb(src_reg)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov WORD [Rq(*TEMP_REG) + *dest_off], Rw(src_reg)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov DWORD [Rq(*TEMP_REG) + *dest_off], Rd(src_reg)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov QWORD [Rq(*TEMP_REG) + *dest_off], Rq(src_reg)
                        ),
                        _ => todo!(),
                    }
                }
            },
            (
                Location::Mem(dest_ro),
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
                            ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
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
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => {
                            let src_ro = RegAndOffset {
                                reg: *src_reg,
                                off: *src_off,
                            };
                            self.copy_memory(&dest_ro, &src_ro, size);
                        }
                    }
                }
            },
            (
                Location::Indirect {
                    ptr: dest_ind,
                    off: dest_off,
                },
                Location::Mem(src_ro),
            ) => {
                debug_assert!(src_ro.reg != *TEMP_REG);
                match dest_ind {
                    IndirectLoc::Reg(dest_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dest_reg) + *dest_off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dest_reg) + *dest_off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dest_reg) + *dest_off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dest_reg) + *dest_off], Rq(*TEMP_REG)
                        ),
                        _ => {
                            let dest_ro = RegAndOffset {
                                reg: *dest_reg,
                                off: 0,
                            };
                            self.copy_memory(&dest_ro, src_ro, size);
                        }
                    },
                    IndirectLoc::Mem(dest_ro) => match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov BYTE [Rq(src_ro.reg) + *dest_off], Rb(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov WORD [Rq(src_ro.reg) + *dest_off], Rw(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov DWORD [Rq(src_ro.reg) + *dest_off], Rd(*TEMP_REG)
                            ; pop Rq(src_ro.reg)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; push Rq(src_ro.reg)
                            ; mov Rq(src_ro.reg), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov QWORD [Rq(src_ro.reg) + *dest_off], Rq(*TEMP_REG)
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
