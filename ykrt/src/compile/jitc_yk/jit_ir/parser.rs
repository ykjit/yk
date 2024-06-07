//! A basic parser for JIT IR.
//!
//! This module -- which is only intended to be compiled-in in `cfg(test)` -- adds a `from_str`
//! method to [Module] which takes in JIT IR as a string, parses it, and produces a [Module]. This
//! makes it possible to write JIT IR tests using JIT IR concrete syntax.

use super::super::{
    aot_ir::{BinOp, Predicate},
    jit_ir::{
        BinOpInst, BlackBoxInst, Const, DirectCallInst, DynPtrAddInst, FuncDecl, FuncTy, GuardInfo,
        GuardInst, IcmpInst, Inst, InstIdx, LoadInst, LoadTraceInputInst, Module, Operand,
        PtrAddInst, SExtInst, StoreInst, TruncInst, Ty, TyIdx,
    },
};
use fm::FMBuilder;
use lrlex::{lrlex_mod, DefaultLexerTypes, LRNonStreamingLexer};
use lrpar::{lrpar_mod, NonStreamingLexer, Span};
use regex::Regex;
use std::{collections::HashMap, convert::TryFrom, error::Error, sync::OnceLock};

lrlex_mod!("compile/jitc_yk/jit_ir/jit_ir.l");
lrpar_mod!("compile/jitc_yk/jit_ir/jit_ir.y");

type StorageT = u8;

impl Module {
    /// Parse the string `s` into a [Module].
    ///
    /// # Panics
    ///
    /// If `s` is not parsable or otherwise does not lead to the creation of a valid `Module`.
    pub(crate) fn from_str(s: &str) -> Self {
        // Get the `LexerDef` for the `calc` language.
        let lexerdef = jit_ir_l::lexerdef();
        let lexer = lexerdef.lexer(s);
        // Pass the lexer to the parser and lex and parse the input.
        let (res, errs) = jit_ir_y::parse(&lexer);
        if !errs.is_empty() {
            for e in errs {
                eprintln!("{}", e.pp(&lexer, &jit_ir_y::token_epp));
            }
            panic!("Could not parse input");
        }
        match res {
            Some(Ok((func_decls, globals, bblocks))) => {
                let mut m = Module::new_testing();
                let mut p = JITIRParser {
                    m: &mut m,
                    lexer: &lexer,
                    inst_idx_map: HashMap::new(),
                };
                p.process(func_decls, globals, bblocks).unwrap();
                m.assert_well_formed();
                m
            }
            _ => panic!("Could not produce JIT Module."),
        }
    }

    /// Assert that for the given IR input, as a string, `ir_input`, when transformed by
    /// `ir_transform(Module)` it produces a [Module] that when printed corresponds to the [fm]
    /// pattern `transformed_ptn`.
    ///
    /// `transformed_ptn` allows the following [fm] patterns:
    ///   * `{{.+?}}` matches against the text using [fm]'s name matching. If you have two patterns
    ///     with the same name (e.g. `${{1}} xyz ${{2}}`) then both must match the same literal
    ///     text.
    ///   * `{{_}}` matches against the text using [fm]'s "ignore" name matching. If you have two
    ///     patterns with `{{_}}` then they may match against different literal text.
    pub(crate) fn assert_ir_transform_eq<F>(ir_input: &str, ir_transform: F, transformed_ptn: &str)
    where
        F: FnOnce(Module) -> Module,
    {
        // We want to share the compilation of regexes amongst threads, *but* there is some locking
        // involved, so we want to `clone` the compiled regexes before using them for matching.
        // Hence this odd looking "`static` then `let`" dance.
        static PTN_RE: OnceLock<Regex> = OnceLock::new();
        static PTN_RE_IGNORE: OnceLock<Regex> = OnceLock::new();
        static LITERAL_RE: OnceLock<Regex> = OnceLock::new();
        let ptn_re = PTN_RE
            .get_or_init(|| Regex::new(r"\{\{.+?\}\}").unwrap())
            .clone();
        let ptn_re_ignore = PTN_RE_IGNORE
            .get_or_init(|| Regex::new(r"\{\{_\}\}").unwrap())
            .clone();
        let literal_re = LITERAL_RE
            .get_or_init(|| Regex::new(r"[a-zA-Z0-9\._]+").unwrap())
            .clone();

        let m = Self::from_str(ir_input);
        let m = ir_transform(m);
        let fmm = FMBuilder::new(transformed_ptn)
            .unwrap()
            .name_matcher_ignore(ptn_re_ignore, literal_re.clone())
            .name_matcher(ptn_re, literal_re)
            .build()
            .unwrap();
        if let Err(e) = fmm.matches(&m.to_string()) {
            panic!("{e}");
        }
    }
}

struct JITIRParser<'lexer, 'input: 'lexer, 'a> {
    lexer: &'lexer LRNonStreamingLexer<'lexer, 'input, DefaultLexerTypes<StorageT>>,
    m: &'a mut Module,
    /// A mapping from local operands (e.g. `%3`) in the input text to instruction offsets. For
    /// example if the first instruction in the input is `%3 = ...` the map will be `%3 -> %0`
    /// indicating that whenever we see `%3` in the input we map that to `%0` in the IR we're
    /// generating. To populate this map, each instruction that defines a new local operand needs
    /// to call [Self::add_assign].
    inst_idx_map: HashMap<InstIdx, InstIdx>,
}

impl<'lexer, 'input: 'lexer> JITIRParser<'lexer, 'input, '_> {
    fn process(
        &mut self,
        func_decls: Vec<ASTFuncDecl>,
        _globals: Vec<()>,
        bblocks: Vec<ASTBBlock>,
    ) -> Result<(), Box<dyn Error>> {
        self.process_func_decls(func_decls)?;

        for bblock in bblocks.into_iter() {
            for inst in bblock.insts {
                match inst {
                    ASTInst::BinOp {
                        assign,
                        type_: _,
                        bin_op,
                        lhs,
                        rhs,
                    } => {
                        let inst = BinOpInst::new(
                            self.process_operand(lhs)?,
                            bin_op,
                            self.process_operand(rhs)?,
                        );
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::Call {
                        assign,
                        name: name_span,
                        args,
                    } => {
                        let name = &self.lexer.span_str(name_span)[1..];
                        let fd_idx = self.m.find_func_decl_idx_by_name(name);
                        let ops = self.process_operands(args)?;
                        let inst = DirectCallInst::new(self.m, fd_idx, ops)
                            .map_err(|e| self.error_at_span(name_span, &e.to_string()))?;
                        if let Some(x) = assign {
                            self.add_assign(self.m.len(), x)?;
                        }
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::Eq {
                        assign,
                        type_: _,
                        lhs,
                        rhs,
                    } => {
                        let inst = IcmpInst::new(
                            self.process_operand(lhs)?,
                            Predicate::Equal,
                            self.process_operand(rhs)?,
                        );
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::Guard { operand, is_true } => {
                        let gidx = self
                            .m
                            .push_guardinfo(GuardInfo::new(Vec::new(), Vec::new()))
                            .unwrap();
                        let inst = GuardInst::new(self.process_operand(operand)?, is_true, gidx);
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::Load {
                        assign,
                        type_,
                        val,
                        volatile,
                    } => {
                        let inst = LoadInst::new(
                            self.process_operand(val)?,
                            self.process_type(type_)?,
                            volatile,
                        );
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::LoadTraceInput { assign, type_, off } => {
                        let off = self
                            .lexer
                            .span_str(off)
                            .parse::<u32>()
                            .map_err(|e| self.error_at_span(off, &e.to_string()))?;
                        let inst = LoadTraceInputInst::new(off, self.process_type(type_)?);
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::PtrAdd {
                        assign,
                        type_: _,
                        ptr,
                        off,
                    } => {
                        let off = self
                            .lexer
                            .span_str(off)
                            .parse::<i32>()
                            .map_err(|e| self.error_at_span(off, &e.to_string()))?;
                        let inst = PtrAddInst::new(self.process_operand(ptr)?, off);
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::DynPtrAdd {
                        assign,
                        type_: _,
                        ptr,
                        num_elems,
                        elem_size,
                    } => {
                        let elem_size = self
                            .lexer
                            .span_str(elem_size)
                            .parse::<u16>()
                            .map_err(|e| self.error_at_span(elem_size, &e.to_string()))?;
                        let inst = DynPtrAddInst::new(
                            self.process_operand(ptr)?,
                            self.process_operand(num_elems)?,
                            elem_size,
                        );
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::SExt { assign, type_, val } => {
                        let inst =
                            SExtInst::new(&self.process_operand(val)?, self.process_type(type_)?);
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::Store { tgt, val, volatile } => {
                        let inst = StoreInst::new(
                            self.process_operand(tgt)?,
                            self.process_operand(val)?,
                            volatile,
                        );
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::BlackBox(op) => {
                        let inst = BlackBoxInst::new(self.process_operand(op)?);
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::TraceLoopStart => {
                        self.m.push(Inst::TraceLoopStart).unwrap();
                    }
                    ASTInst::Trunc {
                        assign,
                        type_,
                        operand,
                    } => {
                        let inst = TruncInst::new(
                            &self.process_operand(operand)?,
                            self.process_type(type_)?,
                        );
                        self.add_assign(self.m.len(), assign)?;
                        self.m.push(inst.into()).unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    fn process_func_decls(&mut self, func_decls: Vec<ASTFuncDecl>) -> Result<(), Box<dyn Error>> {
        for ASTFuncDecl {
            name: name_span,
            arg_tys,
            is_varargs,
            rtn_ty,
        } in func_decls
        {
            let name = self.lexer.span_str(name_span).to_owned();
            let arg_tys = {
                let mut mapped = Vec::with_capacity(arg_tys.len());
                for x in arg_tys.into_iter() {
                    mapped.push(self.process_type(x)?);
                }
                mapped
            };
            let rtn_ty = self.process_type(rtn_ty)?;
            let func_ty = self
                .m
                .insert_ty(Ty::Func(FuncTy::new(arg_tys, rtn_ty, is_varargs)))
                .map_err(|e| self.error_at_span(name_span, &e.to_string()))?;
            self.m
                .insert_func_decl(FuncDecl::new(name, func_ty))
                .map_err(|e| self.error_at_span(name_span, &e.to_string()))?;
        }
        Ok(())
    }

    fn process_operands(&mut self, ops: Vec<ASTOperand>) -> Result<Vec<Operand>, Box<dyn Error>> {
        let mut mapped = Vec::with_capacity(ops.len());
        for x in ops {
            mapped.push(self.process_operand(x)?);
        }
        Ok(mapped)
    }

    fn process_operand(&mut self, op: ASTOperand) -> Result<Operand, Box<dyn Error>> {
        match op {
            ASTOperand::ConstInt(span) => {
                let s = self.lexer.span_str(span);
                let [val, type_] = <[&str; 2]>::try_from(s.split('i').collect::<Vec<_>>()).unwrap();
                let width = type_
                    .parse::<u32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let const_ = match width {
                    64 => {
                        let val = val
                            .parse::<i64>()
                            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                        Const::I64(val)
                    }
                    32 => {
                        let val = val
                            .parse::<i32>()
                            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                        Const::I32(val)
                    }
                    16 => {
                        let val = val
                            .parse::<i16>()
                            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                        Const::I16(val)
                    }
                    8 => {
                        let val = val
                            .parse::<i8>()
                            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                        Const::I8(val)
                    }
                    x => todo!("{x:}"),
                };
                Ok(Operand::Const(
                    self.m
                        .insert_const(const_)
                        .map_err(|e| self.error_at_span(span, &e.to_string()))?,
                ))
            }
            ASTOperand::ConstPtr(span) => {
                let s = self.lexer.span_str(span);
                debug_assert!(matches!(&s[0..2], "0x" | "0X"));
                let val = usize::from_str_radix(&s[2..], 16)
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let const_ = Const::Ptr(val);
                Ok(Operand::Const(
                    self.m
                        .insert_const(const_)
                        .map_err(|e| self.error_at_span(span, &e.to_string()))?,
                ))
            }
            ASTOperand::Local(span) => {
                let idx = self.lexer.span_str(span)[1..]
                    .parse::<usize>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let idx =
                    InstIdx::new(idx).map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let mapped_idx = self.inst_idx_map.get(&idx).ok_or_else(|| {
                    self.error_at_span(
                        span,
                        &format!("Undefined local operand '%{}'", usize::from(idx)),
                    )
                })?;
                Ok(Operand::Local(*mapped_idx))
            }
        }
    }

    fn process_type(&mut self, type_: ASTType) -> Result<TyIdx, Box<dyn Error>> {
        match type_ {
            ASTType::Int(span) => {
                let width = self.lexer.span_str(span)[1..]
                    .parse::<u32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                self.m
                    .insert_ty(Ty::Integer(width))
                    .map_err(|e| self.error_at_span(span, &e.to_string()))
            }
            ASTType::Ptr => Ok(self.m.ptr_ty_idx()),
            ASTType::Void => Ok(self.m.void_ty_idx()),
        }
    }

    /// Add an assignment of a local operand (e.g. `%3 = ...`) to the instruction to be inserted at
    /// `instrs_len`. Note: this must be called *before* inserting the instruction, as it allowws
    /// this function to capture cases where users try referencing the variable itself (e.g. `%0 =
    /// %0` will be caught as an error by this function).
    fn add_assign(&mut self, instrs_len: usize, span: Span) -> Result<(), Box<dyn Error>> {
        let idx = self.lexer.span_str(span)[1..]
            .parse::<usize>()
            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
        let idx = InstIdx::new(idx).map_err(|e| self.error_at_span(span, &e.to_string()))?;
        match self
            .inst_idx_map
            .insert(idx, InstIdx::new(instrs_len).unwrap())
        {
            None => Ok(()),
            Some(_) => Err(format!("Local operand '%{}' redefined", usize::from(idx)).into()),
        }
    }

    /// Return an error message pinpointing `span` as the culprit.
    fn error_at_span(&self, span: Span, msg: &str) -> Box<dyn Error> {
        let ((line_off, col), _) = self.lexer.line_col(span);
        let code = self
            .lexer
            .span_lines_str(span)
            .split('\n')
            .next()
            .unwrap()
            .trim();
        Box::from(format!(
            "Line {}, column {}:\n  {}\n{}",
            line_off,
            col,
            code.trim(),
            msg
        ))
    }
}

struct ASTFuncDecl {
    name: Span,
    arg_tys: Vec<ASTType>,
    is_varargs: bool,
    rtn_ty: ASTType,
}

#[derive(Debug)]
struct ASTBBlock {
    #[allow(dead_code)]
    label: Span,
    insts: Vec<ASTInst>,
}

#[derive(Debug)]
enum ASTInst {
    BinOp {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        bin_op: BinOp,
        lhs: ASTOperand,
        rhs: ASTOperand,
    },
    Call {
        assign: Option<Span>,
        name: Span,
        args: Vec<ASTOperand>,
    },
    Eq {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        lhs: ASTOperand,
        rhs: ASTOperand,
    },
    Guard {
        operand: ASTOperand,
        is_true: bool,
    },
    Load {
        assign: Span,
        type_: ASTType,
        val: ASTOperand,
        volatile: bool,
    },
    LoadTraceInput {
        assign: Span,
        type_: ASTType,
        off: Span,
    },
    PtrAdd {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        ptr: ASTOperand,
        off: Span,
    },
    DynPtrAdd {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        ptr: ASTOperand,
        num_elems: ASTOperand,
        elem_size: Span,
    },
    SExt {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        val: ASTOperand,
    },
    Store {
        tgt: ASTOperand,
        val: ASTOperand,
        volatile: bool,
    },
    BlackBox(ASTOperand),
    TraceLoopStart,
    Trunc {
        assign: Span,
        type_: ASTType,
        operand: ASTOperand,
    },
}

#[derive(Debug)]
enum ASTOperand {
    Local(Span),
    ConstInt(Span),
    ConstPtr(Span),
}

#[derive(Debug)]
enum ASTType {
    Int(Span),
    Ptr,
    Void,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::jitc_yk::jit_ir::{FuncTy, Ty};

    #[test]
    fn roundtrip() {
        let mut m = Module::new_testing();
        let i16_ty_idx = m.insert_ty(Ty::Integer(16)).unwrap();
        let op1 = m
            .push_and_make_operand(LoadTraceInputInst::new(0, i16_ty_idx).into())
            .unwrap();
        let op2 = m
            .push_and_make_operand(LoadTraceInputInst::new(16, i16_ty_idx).into())
            .unwrap();
        let op3 = m
            .push_and_make_operand(BinOpInst::new(op1.clone(), BinOp::Add, op2.clone()).into())
            .unwrap();
        let op4 = m
            .push_and_make_operand(BinOpInst::new(op1.clone(), BinOp::Add, op3.clone()).into())
            .unwrap();
        m.push(BlackBoxInst::new(op3).into()).unwrap();
        m.push(BlackBoxInst::new(op4).into()).unwrap();
        let s = m.to_string();
        let parsed_m = Module::from_str(&s);
        assert_eq!(m.to_string(), parsed_m.to_string());
    }

    #[test]
    fn module_patterns() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i16 = load_ti 0
            %1: i16 = load_ti 1
            %2: i16 = add %0, %1
        ",
            |m| m,
            "
          ...
          entry:
            %{{0}}: i16 = load_ti 0
            %{{1}}: i16 = load_ti 1
            %{{_}}: i16 = add %{{0}}, %{{1}}
        ",
        );
    }

    #[test]
    fn all_jit_ir_syntax() {
        Module::from_str(
            "
            func_decl f1()
            func_decl f2(i8) -> i32
            func_decl f3(i8, i32, ...) -> i64
            func_decl f4(...)
            entry:
              %0: i16 = load_ti 0
              %1: i16 = trunc %0
              %2: i16 = add %0, %1
              %4: i16 = eq %1, %2
              tloop_start
              guard %4, true
              call @f1()
              %5: i8 = load_ti 1
              %6: i32 = call @f2(%5)
              %7: i32 = load_ti 2
              %8: i64 = call @f3(%5, %7, %0)
              call @f4(%0, %1)
              %9: ptr = load_ti 3
              *%9 = %8
              %10: i32 = load %9
              %11: i64 = sext %10
              %12: i32 = add %0, %1
              %13: i32 = sub %0, %1
              %14: i32 = mul %0, %1
              %15: i32 = or %0, %1
              %16: i32 = and %0, %1
              %17: i32 = xor %0, %1
              %18: i32 = shl %0, %1
              %19: i32 = ashr %0, %1
              %20: i32 = fadd %0, %1
              %21: i32 = fdiv %0, %1
              %22: i32 = fmul %0, %1
              %23: i32 = frem %0, %1
              %24: i32 = fsub %0, %1
              %25: i32 = lshr %0, %1
              %26: i32 = sdiv %0, %1
              %27: i32 = srem %0, %1
              %28: i32 = udiv %0, %1
              %29: i32 = urem %0, %1
              %30: i8 = load_ti 4
              %31: i16 = load_ti 5
              %32: i32 = load_ti 5
              %33: i64 = load_ti 6
              %34: i8 = add %30, 127i8
              %35: i8 = add %30, -128i8
              %36: i16 = add %31, 32767i16
              %37: i16 = add %31, -32768i16
              %38: i32 = add %32, 2147483647i32
              %39: i32 = add %32, -2147483648i32
              %40: i64 = add %33, 9223372036854775807i64
              %41: i64 = add %33, -9223372036854775808i64
              *%9 = 0x0
              *%9 = 0xFFFFFFFF
        ",
        );
    }

    #[test]
    fn func_decls() {
        let mut m = Module::from_str(
            "
              func_decl f1()
              func_decl f2(i8) -> i32
              func_decl f3(i8, i32, ...) -> i64
              func_decl f4(...)
        ",
        );
        // We don't have a "lookup a function declaration" function, but we can check that if we
        // "insert" identical IR function declarations to the module that no new function
        // declarations are actually added.
        assert_eq!(m.func_decls_len(), 4);

        let f1_ty_idx = m
            .insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_ty_idx(), false)))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f1".to_owned(), f1_ty_idx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let i32_ty_idx = m.insert_ty(Ty::Integer(32)).unwrap();
        let f2_ty_idx = m
            .insert_ty(Ty::Func(FuncTy::new(
                vec![m.int8_ty_idx()],
                i32_ty_idx,
                false,
            )))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f2".to_owned(), f2_ty_idx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let i64_ty_idx = m.insert_ty(Ty::Integer(64)).unwrap();
        let f3_ty_idx = m
            .insert_ty(Ty::Func(FuncTy::new(
                vec![m.int8_ty_idx(), i32_ty_idx],
                i64_ty_idx,
                true,
            )))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f3".to_owned(), f3_ty_idx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let f4_ty_idx = m
            .insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_ty_idx(), true)))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f4".to_owned(), f4_ty_idx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);
    }

    #[test]
    fn discontinuous_operand_ids() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %7: i16 = load_ti 0
            %3: i16 = load_ti 1
            %19: i16 = add %7, %3
        ",
            |m| m,
            "
          ...
          entry:
            %0: i16 = load_ti 0
            %1: i16 = load_ti 1
            %2: i16 = add %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "Undefined local operand '%7'")]
    fn no_such_local_operand() {
        Module::from_str(
            "
          entry:
            %19: i16 = add %7, %3
        ",
        );
    }

    #[test]
    #[should_panic(expected = "Local operand '%3' redefined")]
    fn repeated_local_operand_definition() {
        Module::from_str(
            "
          entry:
            %3: i16 = load_ti 0
            %4: i16 = load_ti 1
            %3: i16 = load_ti 2
        ",
        );
    }
}
