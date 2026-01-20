//! A HIR parser, suitable for creating [hir::Mod]s for testing purposes.
//!
//! Broadly speaking, the parser accepts HIR output as input, with the following differences:
//!
//! 1. Calls are of the form `call <extern_name> <local> (<args>)` where `<extern_name>` must
//!    reference an `extern` declaration at the beginning of the input.
//! 2. `arg` in the parser has the format `reg [<name>] | stack [<off>]`. This allows you to
//!    specify automatic, or specific, registers / stack offsets that are then converted into HIR's
//!    `arg <n>` format. `arg reg` will auto-assign a register; `arg "rax"` will (on x64) assign
//!    `RAX` and so on.
//! 3. Integers are either (possibly negative) decimal numbers or (always positive) hex numbers.
//!    i.e. `-15` is allowed, but `-0xF` is not. Where sign extension is relevant, use decimal
//!    numbers if you want sign extension, and hex numbers if you do not want sign extension.

use crate::{
    compile::{
        j2::{
            hir::*,
            regalloc::{RegFill, RegT, TestRegIter, VarLoc, VarLocs},
        },
        jitc_yk::{
            aot_ir::{BBlockId, BBlockIdx, FuncIdx},
            arbbitint::ArbBitInt,
        },
    },
    mt::TraceId,
};
use index_vec::IndexVec;
use lrlex::{DefaultLexerTypes, LRNonStreamingLexer, lrlex_mod};
use lrpar::{NonStreamingLexer, Span, lrpar_mod};
use smallvec::SmallVec;
use std::{collections::HashMap, ffi::CString, marker::PhantomData};

lrlex_mod!("compile/j2/hir.l");
lrpar_mod!("compile/j2/hir.y");
type StorageT = u16;

struct HirParser<'lexer, 'input: 'lexer, Reg: RegT> {
    lexer: &'lexer LRNonStreamingLexer<'lexer, 'input, DefaultLexerTypes<StorageT>>,
    externs: HashMap<&'input str, TyIdx>,
    insts: IndexVec<InstIdx, Inst>,
    tys: IndexVec<TyIdx, Ty>,
    /// The [TyIdx] for [Ty::Int(1)].
    tyidx_int1: TyIdx,
    /// The [TyIdx] for [Ty::Ptr(0)].
    tyidx_ptr0: TyIdx,
    /// The [TyIdx] for [Ty::Void].
    tyidx_void: TyIdx,
    /// ty_map is used to ensure that only distinct [Ty]s lead to new [TyIdx]s.
    ty_map: HashMap<Ty, TyIdx>,
    phantom: PhantomData<Reg>,
}

impl<'lexer, 'input: 'lexer, Reg: RegT> HirParser<'lexer, 'input, Reg> {
    fn push_ty(&mut self, ty: Ty) -> TyIdx {
        if let Some(tyidx) = self.ty_map.get(&ty) {
            return *tyidx;
        }
        let tyidx = self.tys.push(ty.clone());
        self.ty_map.insert(ty, tyidx);
        tyidx
    }

    fn build(mut self, astexterns: Vec<AstExtern>, astinsts: Vec<AstInst>) -> Mod<Reg> {
        for AstExtern {
            name,
            ty:
                AstFuncTy {
                    arg_tys,
                    has_varargs,
                    rtn_ty,
                },
        } in astexterns
        {
            let name = self.lexer.span_str(name);
            let args_tyidxs = arg_tys
                .into_iter()
                .map(|x| self.p_ty(x))
                .collect::<SmallVec<_>>();
            let rtn_tyidx = self.p_ty(rtn_ty);
            let tyidx = self.push_ty(Ty::Func(Box::new(FuncTy {
                args_tyidxs,
                has_varargs,
                rtn_tyidx,
            })));
            self.externs.insert(name, tyidx);
        }

        let mut entry_vlocs = Vec::new();
        let mut guards = IndexVec::new();
        let mut gblocks = IndexVec::new();
        let mut testregiter = Reg::iter_test_regs();
        let mut autoregused = false;
        let mut manualregused = false;
        for inst in astinsts {
            match inst {
                AstInst::Blackbox(span) => {
                    let val = self.p_local(span);
                    self.insts.push(BlackBox { val }.into());
                }
                AstInst::Abs {
                    local,
                    ty,
                    val,
                    int_min_poison,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(
                        Abs {
                            tyidx,
                            val,
                            int_min_poison,
                        }
                        .into(),
                    );
                }
                AstInst::Add {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        Add {
                            tyidx,
                            lhs,
                            rhs,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                AstInst::And {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(And { tyidx, lhs, rhs }.into());
                }
                AstInst::AShr {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        AShr {
                            tyidx,
                            lhs,
                            rhs,
                            exact: false,
                        }
                        .into(),
                    );
                }
                AstInst::Arg {
                    local,
                    ty,
                    vlocs: ast_vlocs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let mut vlocs = VarLocs::with_capacity(ast_vlocs.len());
                    for ast_vloc in ast_vlocs {
                        let vloc = match ast_vloc {
                            AstVLoc::AutoReg => {
                                if manualregused {
                                    self.err_span(
                                        local,
                                        "Can't mix `auto` and manually assigned registers",
                                    );
                                }
                                autoregused = true;
                                VarLoc::Reg(
                                    testregiter.next_reg(&self.tys[tyidx]).unwrap_or_else(|| {
                                        self.err_span(local, "Exhausted automatic test registers")
                                    }),
                                    RegFill::Undefined,
                                )
                            }
                            AstVLoc::Reg(span, fill) => {
                                if autoregused {
                                    self.err_span(
                                        local,
                                        "Can't mix `auto` and manually assigned registers",
                                    );
                                }
                                manualregused = true;
                                let s = self.lexer.span_str(span).trim_prefix('"').trim_suffix('"');
                                match Reg::from_str(s) {
                                    Some(reg) => VarLoc::Reg(reg, fill),
                                    None => self.err_span(span, &format!("No such register {s}")),
                                }
                            }
                            AstVLoc::AutoStack => todo!(),
                            AstVLoc::Stack(_span) => todo!(),
                            AstVLoc::StackOff(span) => {
                                let s = self.lexer.span_str(span);
                                let stack_off = s.parse::<u32>().unwrap_or_else(|e| {
                                    self.err_span(span, &format!("Invalid StackOff: {e}"))
                                });
                                VarLoc::StackOff(stack_off)
                            }
                        };
                        vlocs.push(vloc);
                    }
                    entry_vlocs.push(vlocs);
                    self.insts.push(Arg { tyidx }.into());
                }
                AstInst::Call {
                    local,
                    ty,
                    extern_,
                    tgt,
                    args,
                } => {
                    assert!((local.is_none() && ty.is_none()) || (local.is_some() && ty.is_some()));
                    if let (Some(local), Some(_ty)) = (local, ty) {
                        self.p_def_local(local);
                    }
                    let func_tyidx = self.externs[self.lexer.span_str(extern_)];
                    let tgt = self.p_local(tgt);
                    let args = args
                        .iter()
                        .map(|x| self.p_local(*x))
                        .collect::<SmallVec<_>>();
                    self.insts.push(
                        Call {
                            tgt,
                            func_tyidx,
                            args,
                        }
                        .into(),
                    );
                }
                AstInst::Const { local, ty, kind } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    match kind {
                        AstConst::Double(span) => {
                            let s = self.lexer.span_str(span).trim_suffix("double");
                            let v = s
                                .parse::<f64>()
                                .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                            self.insts.push(Inst::Const(Const {
                                tyidx,
                                kind: ConstKind::Double(v),
                            }));
                        }
                        AstConst::Float(span) => {
                            let s = self.lexer.span_str(span).trim_suffix("float");
                            let v = s
                                .parse::<f32>()
                                .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                            self.insts.push(Inst::Const(Const {
                                tyidx,
                                kind: ConstKind::Float(v),
                            }));
                        }
                        AstConst::Int(span) => {
                            let s = self.lexer.span_str(span);
                            let bitw = match &self.tys[tyidx] {
                                Ty::Int(bitw) => {
                                    // We only handle 64-bit ints for now.
                                    assert!(*bitw <= 64);
                                    *bitw
                                }
                                Ty::Ptr(0) => {
                                    u32::try_from(std::mem::size_of::<usize>() * 8).unwrap()
                                }
                                x => todo!("{x:?}"),
                            };
                            assert!(
                                usize::try_from(bitw).unwrap() <= std::mem::size_of::<usize>() * 8
                            );
                            let val = if s.starts_with("0x") || s.starts_with("0X") {
                                let val = u64::from_str_radix(&s[2..], 16)
                                    .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                                if bitw < 64 && val > (1 << bitw) - 1 {
                                    self.err_span(span,
                          &format!("Unsigned constant {val} exceeds the bit width {bitw} of the integer type"));
                                }
                                val
                            } else if s.starts_with("-") {
                                let val = s
                                    .parse::<i64>()
                                    .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                                if bitw < 64
                                    && (val < -((1 << bitw) - 1) / 2 - 1
                                        || val >= ((1 << bitw) - 1) / 2)
                                {
                                    self.err_span(span,
                          &format!("Signed constant {val} exceeds the bit width {bitw} of the integer type"));
                                }
                                val as u64
                            } else {
                                let val = s
                                    .parse::<u64>()
                                    .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                                if bitw < 64 && val > (1 << bitw) - 1 {
                                    self.err_span(span,
                          &format!("Unsigned constant {val} exceeds the bit width {bitw} of the integer type"));
                                }
                                val
                            };
                            self.insts.push(match &self.tys[tyidx] {
                                Ty::Int(_) => Inst::Const(Const {
                                    tyidx,
                                    kind: ConstKind::Int(ArbBitInt::from_u64(bitw, val)),
                                }),
                                Ty::Ptr(0) => Inst::Const(Const {
                                    tyidx,
                                    kind: ConstKind::Ptr(usize::try_from(val).unwrap()),
                                }),
                                _ => unreachable!(),
                            });
                        }
                    }
                }
                AstInst::CtPop { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(CtPop { tyidx, val }.into());
                }
                AstInst::DynPtrAdd {
                    local,
                    ty,
                    ptr,
                    num_elems,
                    elem_size,
                } => {
                    self.p_def_local(local);
                    let _tyidx = self.p_ty(ty);
                    let ptr = self.p_local(ptr);
                    let num_elems = self.p_local(num_elems);
                    let elem_size = self.p_bitw(elem_size);
                    self.insts.push(
                        DynPtrAdd {
                            ptr,
                            num_elems,
                            elem_size,
                        }
                        .into(),
                    );
                }
                AstInst::FAdd {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(FAdd { tyidx, lhs, rhs }.into());
                }
                AstInst::FCmp {
                    local,
                    ty,
                    pred,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    assert_eq!(&self.tys[tyidx], &Ty::Int(1));
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(FCmp { pred, lhs, rhs }.into());
                }
                AstInst::FDiv {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(FDiv { tyidx, lhs, rhs }.into());
                }
                AstInst::Floor { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(Floor { tyidx, val }.into());
                }
                AstInst::FMul {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(FMul { tyidx, lhs, rhs }.into());
                }
                AstInst::FNeg { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(FNeg { tyidx, val }.into());
                }
                AstInst::FSub {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(FSub { tyidx, lhs, rhs }.into());
                }
                AstInst::FPExt { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(FPExt { tyidx, val }.into());
                }
                AstInst::FPToSI { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(FPToSI { tyidx, val }.into());
                }
                AstInst::Global { local, ty, name } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    if !matches!(&self.tys[tyidx], Ty::Ptr(_)) {
                        self.err_span(local, "Type of @global instructions must be 'ptr'");
                    }
                    let s = self.lexer.span_str(name);
                    assert_eq!(s.chars().nth(0).unwrap(), '@');
                    let n = CString::new(&s[1..]).unwrap();
                    let rtn = unsafe { libc::dlsym(std::ptr::null_mut(), n.as_c_str().as_ptr()) };
                    if rtn.is_null() {
                        self.err_span(name, &format!("Symbol '{}' not found", &s[1..]));
                    }
                    self.insts.push(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Ptr(rtn.addr()),
                    }));
                }
                AstInst::Guard {
                    expect,
                    cond,
                    exit_vars,
                } => {
                    let cond = self.p_local(cond);
                    let exit_vars = exit_vars
                        .into_iter()
                        .map(|x| self.p_local(x))
                        .collect::<Vec<_>>();
                    let mut ginsts = IndexVec::with_capacity(exit_vars.len());
                    for iidx in &exit_vars {
                        match &self.insts[*iidx] {
                            Inst::Const(x) => {
                                ginsts.push(x.clone().into());
                            }
                            Inst::Guard(_) => panic!(),
                            Inst::Term(_) => panic!(),
                            x => {
                                ginsts.push(Inst::Arg(Arg {
                                    tyidx: x.tyidx(&self),
                                }));
                            }
                        }
                    }
                    ginsts.push(Inst::Term(Term(
                        (0..exit_vars.len()).map(InstIdx::from).collect::<Vec<_>>(),
                    )));
                    let gbidx = Some(gblocks.push(Block { insts: ginsts }));
                    let bid = BBlockId::new(FuncIdx::from(0), BBlockIdx::from(0));
                    let geidx = guards.push(GuardExtra {
                        bid,
                        switch: None,
                        exit_vars,
                        deopt_frames: SmallVec::new(),
                        gbidx,
                    });
                    self.insts.push(Inst::Guard(Guard {
                        geidx,
                        expect,
                        cond,
                    }));
                }
                AstInst::ICmp {
                    local,
                    ty,
                    pred,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    assert_eq!(&self.tys[tyidx], &Ty::Int(1));
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        ICmp {
                            pred,
                            lhs,
                            rhs,
                            samesign: false,
                        }
                        .into(),
                    );
                }
                AstInst::IntToPtr { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(IntToPtr { tyidx, val }.into());
                }
                AstInst::Load { local, ty, ptr } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let ptr = self.p_local(ptr);
                    self.insts.push(
                        Load {
                            tyidx,
                            ptr,
                            is_volatile: false,
                        }
                        .into(),
                    );
                }
                AstInst::LShr {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        LShr {
                            tyidx,
                            lhs,
                            rhs,
                            exact: false,
                        }
                        .into(),
                    );
                }
                AstInst::MemCpy {
                    dst,
                    src,
                    len,
                    volatile,
                } => {
                    let dst = self.p_local(dst);
                    let src = self.p_local(src);
                    let len = self.p_local(len);
                    self.insts.push(
                        MemCpy {
                            dst,
                            src,
                            len,
                            volatile,
                        }
                        .into(),
                    );
                }
                AstInst::MemSet {
                    dst,
                    val,
                    len,
                    volatile,
                } => {
                    let dst = self.p_local(dst);
                    let val = self.p_local(val);
                    let len = self.p_local(len);
                    self.insts.push(
                        MemSet {
                            dst,
                            val,
                            len,
                            volatile,
                        }
                        .into(),
                    );
                }
                AstInst::Mul {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        Mul {
                            tyidx,
                            lhs,
                            rhs,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                AstInst::Or {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        Or {
                            tyidx,
                            lhs,
                            rhs,
                            disjoint: false,
                        }
                        .into(),
                    );
                }
                AstInst::PtrAdd {
                    local,
                    ty,
                    ptr,
                    off,
                } => {
                    self.p_def_local(local);
                    let _tyidx = self.p_ty(ty);
                    let ptr = self.p_local(ptr);
                    let off = self
                        .lexer
                        .span_str(off)
                        .parse::<i32>()
                        .unwrap_or_else(|e| self.err_span(off, &e.to_string()));
                    self.insts.push(
                        PtrAdd {
                            ptr,
                            off,
                            in_bounds: false,
                            nusw: false,
                            nuw: false,
                        }
                        .into(),
                    );
                }
                AstInst::PtrToInt { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(PtrToInt { tyidx, val }.into());
                }
                AstInst::SDiv {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        SDiv {
                            tyidx,
                            lhs,
                            rhs,
                            exact: false,
                        }
                        .into(),
                    );
                }
                AstInst::Select {
                    local,
                    ty,
                    cond,
                    truev,
                    falsev,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let cond = self.p_local(cond);
                    let truev = self.p_local(truev);
                    let falsev = self.p_local(falsev);
                    self.insts.push(
                        Select {
                            tyidx,
                            cond,
                            truev,
                            falsev,
                        }
                        .into(),
                    );
                }
                AstInst::SExt { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(SExt { tyidx, val }.into());
                }
                AstInst::Shl {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        Shl {
                            tyidx,
                            lhs,
                            rhs,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                AstInst::SIToFP { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(SIToFP { tyidx, val }.into());
                }
                AstInst::SMax {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(SMax { tyidx, lhs, rhs }.into());
                }
                AstInst::SMin {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(SMin { tyidx, lhs, rhs }.into());
                }
                AstInst::SRem {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(SRem { tyidx, lhs, rhs }.into());
                }
                AstInst::Store { val, ptr } => {
                    let val = self.p_local(val);
                    let ptr = self.p_local(ptr);
                    self.insts.push(
                        Store {
                            val,
                            ptr,
                            is_volatile: false,
                        }
                        .into(),
                    );
                }
                AstInst::Sub {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        Sub {
                            tyidx,
                            lhs,
                            rhs,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                AstInst::Term { locals } => {
                    let term_vars = locals.iter().map(|x| self.p_local(*x)).collect::<Vec<_>>();
                    self.insts.push(Inst::Term(Term(term_vars)));
                }
                AstInst::Trunc { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(
                        Trunc {
                            tyidx,
                            val,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                AstInst::UDiv {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(
                        UDiv {
                            tyidx,
                            lhs,
                            rhs,
                            exact: false,
                        }
                        .into(),
                    );
                }
                AstInst::UIToFP { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(
                        UIToFP {
                            tyidx,
                            val,
                            nneg: false,
                        }
                        .into(),
                    );
                }
                AstInst::Xor {
                    local,
                    ty,
                    lhs,
                    rhs,
                } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let lhs = self.p_local(lhs);
                    let rhs = self.p_local(rhs);
                    self.insts.push(Xor { tyidx, lhs, rhs }.into());
                }
                AstInst::ZExt { local, ty, val } => {
                    self.p_def_local(local);
                    let tyidx = self.p_ty(ty);
                    let val = self.p_local(val);
                    self.insts.push(ZExt { tyidx, val }.into());
                }
            }
        }

        let block = Block { insts: self.insts };
        let m = Mod {
            trid: TraceId::testing(),
            trace_start: TraceStart::Test,
            trace_end: TraceEnd::Test { entry_vlocs, block },
            tys: self.tys,
            tyidx_int1: self.tyidx_int1,
            tyidx_ptr0: self.tyidx_ptr0,
            tyidx_void: self.tyidx_void,
            guard_extras: guards,
            gblocks,
            addr_name_map: None,
        };
        m.assert_well_formed();
        m
    }

    fn err_span(&self, span: Span, msg: &str) -> ! {
        let ((line_off, col), _) = self.lexer.line_col(span);
        let code = self
            .lexer
            .span_lines_str(span)
            .split('\n')
            .next()
            .unwrap()
            .trim();
        panic!(
            "Line {}, column {}:\n  {}\n{}",
            line_off,
            col,
            code.trim(),
            msg
        );
    }

    fn p_bitw(&self, span: Span) -> u32 {
        let bitw = self
            .lexer
            .span_str(span)
            .parse::<u32>()
            .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
        if bitw > 1 << 23 {
            todo!();
        }
        bitw
    }

    fn p_def_local(&mut self, span: Span) {
        let n = self.p_local(span);
        if n != self.insts.len() {
            self.err_span(
                span,
                &format!("Incorrect local: should be '%{}'", self.insts.len()),
            );
        }
    }

    fn p_local(&mut self, span: Span) -> InstIdx {
        let s = self.lexer.span_str(span);
        assert_eq!(s.chars().nth(0).unwrap(), '%');
        s[1..]
            .parse::<u32>()
            .map(|x| InstIdx::from(usize::try_from(x).unwrap()))
            .unwrap_or_else(|e| self.err_span(span, &e.to_string()))
    }

    fn p_ty(&mut self, astty: AstTy) -> TyIdx {
        match astty {
            AstTy::Double => self.push_ty(Ty::Double),
            AstTy::Float => self.push_ty(Ty::Float),
            AstTy::Int(span) => {
                let s = self.lexer.span_str(span);
                assert_eq!(s.chars().nth(0).unwrap(), 'i');
                let bitw = s[1..]
                    .parse::<u32>()
                    .unwrap_or_else(|e| self.err_span(span, &e.to_string()));
                if bitw > (1 << 22) {
                    todo!();
                }
                self.push_ty(Ty::Int(bitw))
            }
            AstTy::Ptr => self.push_ty(Ty::Ptr(0)),
            AstTy::Void => self.push_ty(Ty::Void),
        }
    }
}

impl<'lexer, 'input: 'lexer, Reg: RegT> ModLikeT for HirParser<'lexer, 'input, Reg> {
    fn ty(&self, tyidx: TyIdx) -> &Ty {
        &self.tys[tyidx]
    }

    fn tyidx_int1(&self) -> TyIdx {
        self.tyidx_int1
    }

    fn tyidx_ptr0(&self) -> TyIdx {
        self.tyidx_ptr0
    }

    fn tyidx_void(&self) -> TyIdx {
        self.tyidx_void
    }

    fn gextra(&self, _geidx: GuardExtraIdx) -> &GuardExtra {
        todo!()
    }

    fn gextra_mut(&mut self, _geidx: GuardExtraIdx) -> &mut GuardExtra {
        todo!()
    }

    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        todo!()
    }
}

/// Parse the string `s` into a [Mod].
///
/// # Panics
///
/// If `s` is not parsable or otherwise does not lead to the creation of a valid [Mod].
pub(super) fn str_to_mod<Reg: RegT>(s: &str) -> Mod<Reg> {
    let lexerdef = hir_l::lexerdef();
    let lexer = lexerdef.lexer(s);
    let (res, errs) = hir_y::parse(&lexer);
    if !errs.is_empty() {
        for e in errs {
            eprintln!("{}", e.pp(&lexer, &hir_y::token_epp));
        }
        panic!("Could not parse input");
    }

    let Some(Ok((externs, insts))) = res else {
        panic!("No AST produced")
    };

    let mut tys = IndexVec::new();
    let mut ty_map = HashMap::new();
    let tyidx_void = tys.push(Ty::Void);
    ty_map.insert(Ty::Void, tyidx_void);
    let tyidx_ptr0 = tys.push(Ty::Ptr(0));
    ty_map.insert(Ty::Ptr(0), tyidx_ptr0);
    let tyidx_int1 = tys.push(Ty::Int(1));
    ty_map.insert(Ty::Int(1), tyidx_int1);

    let hp = HirParser {
        lexer: &lexer,
        externs: HashMap::new(),
        insts: IndexVec::new(),
        tys,
        tyidx_int1,
        tyidx_ptr0,
        tyidx_void,
        ty_map,
        phantom: PhantomData,
    };
    hp.build(externs, insts)
}

struct AstExtern {
    name: Span,
    ty: AstFuncTy,
}

enum AstConst {
    Double(Span),
    Float(Span),
    Int(Span),
}

struct AstFuncTy {
    arg_tys: Vec<AstTy>,
    has_varargs: bool,
    rtn_ty: AstTy,
}

enum AstInst {
    Blackbox(Span),
    Abs {
        local: Span,
        ty: AstTy,
        val: Span,
        int_min_poison: bool,
    },
    Add {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    And {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    AShr {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Arg {
        local: Span,
        ty: AstTy,
        vlocs: Vec<AstVLoc>,
    },
    Call {
        local: Option<Span>,
        ty: Option<AstTy>,
        extern_: Span,
        tgt: Span,
        args: Vec<Span>,
    },
    Const {
        local: Span,
        ty: AstTy,
        kind: AstConst,
    },
    CtPop {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    DynPtrAdd {
        local: Span,
        ty: AstTy,
        ptr: Span,
        num_elems: Span,
        elem_size: Span,
    },
    FAdd {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    FCmp {
        local: Span,
        pred: FPred,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    FDiv {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Floor {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    FMul {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    FNeg {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    FSub {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    FPExt {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    FPToSI {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    Global {
        local: Span,
        ty: AstTy,
        name: Span,
    },
    Guard {
        expect: bool,
        cond: Span,
        exit_vars: Vec<Span>,
    },
    ICmp {
        local: Span,
        pred: IPred,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    IntToPtr {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    Load {
        local: Span,
        ty: AstTy,
        ptr: Span,
    },
    LShr {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    MemCpy {
        dst: Span,
        src: Span,
        len: Span,
        volatile: bool,
    },
    MemSet {
        dst: Span,
        val: Span,
        len: Span,
        volatile: bool,
    },
    Mul {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Or {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    PtrAdd {
        local: Span,
        ty: AstTy,
        ptr: Span,
        off: Span,
    },
    PtrToInt {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    SDiv {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Select {
        local: Span,
        ty: AstTy,
        cond: Span,
        truev: Span,
        falsev: Span,
    },
    SExt {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    Shl {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    SIToFP {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    SMax {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    SMin {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    SRem {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Store {
        val: Span,
        ptr: Span,
    },
    Sub {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Term {
        locals: Vec<Span>,
    },
    Trunc {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    UIToFP {
        local: Span,
        ty: AstTy,
        val: Span,
    },
    UDiv {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    Xor {
        local: Span,
        ty: AstTy,
        lhs: Span,
        rhs: Span,
    },
    ZExt {
        local: Span,
        ty: AstTy,
        val: Span,
    },
}

#[derive(Debug)]
enum AstTy {
    Double,
    Float,
    Int(Span),
    Ptr,
    Void,
}

enum AstVLoc {
    AutoReg,
    Reg(Span, RegFill),
    AutoStack,
    Stack(Span),
    StackOff(Span),
}
