//! A basic parser for JIT IR.
//!
//! This module -- which is only intended to be compiled-in in `cfg(test)` -- adds a `from_str`
//! method to [Module] which takes in JIT IR as a string, parses it, and produces a [Module]. This
//! makes it possible to write JIT IR tests using JIT IR concrete syntax.

use super::super::{
    aot_ir::{BinOp, FloatPredicate, Predicate},
    jit_ir::{
        BinOpInst, BlackBoxInst, Const, DirectCallInst, DynPtrAddInst, FPExtInst, FcmpInst,
        FloatTy, FuncDecl, FuncTy, GuardInfo, GuardInst, IcmpInst, IndirectCallInst, Inst, InstIdx,
        LoadInst, LoadTraceInputInst, Module, Operand, PtrAddInst, SExtInst, SIToFPInst,
        SelectInst, StoreInst, TruncInst, Ty, TyIdx,
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
            Some(Ok((func_types, func_decls, globals, bblocks))) => {
                let mut m = Module::new_testing();
                let mut p = JITIRParser {
                    m: &mut m,
                    lexer: &lexer,
                    inst_idx_map: HashMap::new(),
                    func_types_map: HashMap::new(),
                };
                p.process(func_types, func_decls, globals, bblocks).unwrap();
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
    func_types_map: HashMap<String, TyIdx>,
}

impl<'lexer, 'input: 'lexer> JITIRParser<'lexer, 'input, '_> {
    fn process(
        &mut self,
        func_types: Vec<ASTFuncType>,
        func_decls: Vec<ASTFuncDecl>,
        _globals: Vec<()>,
        bblocks: Vec<ASTBBlock>,
    ) -> Result<(), Box<dyn Error>> {
        self.process_func_types(func_types)?;
        self.process_func_decls(func_decls)?;

        for bblock in bblocks.into_iter() {
            for inst in bblock.insts {
                match inst {
                    ASTInst::BinOp {
                        assign,
                        type_,
                        bin_op,
                        lhs,
                        rhs,
                    } => {
                        let tyidx = self.process_type(type_)?;
                        let lhs = self.process_operand(lhs)?;
                        let rhs = self.process_operand(rhs)?;
                        if lhs.tyidx(self.m) != tyidx || rhs.tyidx(self.m) != tyidx {
                            return Err(self.error_at_span(
                                assign,
                                "Binop result type incorrect for one or more operands",
                            ));
                        }
                        let inst = BinOpInst::new(lhs, bin_op, rhs);
                        self.push_assign(inst.into(), assign)?;
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
                            self.push_assign(inst.into(), x)?;
                        } else {
                            self.m.push(inst.into()).unwrap();
                        }
                    }
                    ASTInst::Guard {
                        operand,
                        is_true,
                        live_vars,
                    } => {
                        let mut mlive_vars = Vec::with_capacity(live_vars.len());
                        for span in live_vars {
                            let iidx = self.lexer.span_str(span)[1..]
                                .parse::<usize>()
                                .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                            let iidx = InstIdx::new(iidx)
                                .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                            if self.inst_idx_map.get(&iidx).is_none() {
                                return Err(self.error_at_span(
                                    span,
                                    &format!("No such local variable %{iidx}"),
                                ));
                            }
                            mlive_vars.push(iidx);
                        }
                        let gidx = self
                            .m
                            .push_guardinfo(GuardInfo::new(Vec::new(), mlive_vars))
                            .unwrap();
                        let inst = GuardInst::new(self.process_operand(operand)?, is_true, gidx);
                        self.m.push(inst.into()).unwrap();
                    }
                    ASTInst::ICall {
                        assign,
                        func_type: ft_span,
                        target,
                        args,
                    } => {
                        let ft_name = &self.lexer.span_str(ft_span);
                        let ftidx =
                            *self.func_types_map.get(ft_name.to_owned()).ok_or_else(|| {
                                self.error_at_span(
                                    ft_span,
                                    &format!("No such function type '{ft_name}'"),
                                )
                            })?;
                        let tgt = self.process_operand(target)?;
                        let ops = self.process_operands(args)?;
                        let ic = IndirectCallInst::new(self.m, ftidx, tgt, ops)
                            .map_err(|e| self.error_at_span(ft_span, &e.to_string()))?;
                        let icidx = self
                            .m
                            .push_indirect_call(ic)
                            .map_err(|e| self.error_at_span(ft_span, &e.to_string()))?;
                        let inst = Inst::IndirectCall(icidx);
                        if let Some(x) = assign {
                            self.push_assign(inst, x)?;
                        } else {
                            self.m.push(inst).unwrap();
                        }
                    }
                    ASTInst::ICmp {
                        assign,
                        type_,
                        pred,
                        lhs,
                        rhs,
                    } => {
                        let ty = self.process_type(type_)?;
                        match self.m.type_(ty) {
                            Ty::Integer(1) => (),
                            x => {
                                return Err(self.error_at_span(
                                    assign,
                                    &format!(
                                        "ICmp instructions must assign to an i1, not '{}'",
                                        x.display(self.m)
                                    ),
                                ))
                            }
                        }
                        let inst = IcmpInst::new(
                            self.process_operand(lhs)?,
                            pred,
                            self.process_operand(rhs)?,
                        );
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::FCmp {
                        assign,
                        type_,
                        pred,
                        lhs,
                        rhs,
                    } => {
                        let ty = self.process_type(type_)?;
                        match self.m.type_(ty) {
                            Ty::Integer(1) => (),
                            x => {
                                return Err(self.error_at_span(
                                    assign,
                                    &format!(
                                        "FCmp instructions must assign to an i1, not '{}'",
                                        x.display(self.m)
                                    ),
                                ))
                            }
                        }
                        let inst = FcmpInst::new(
                            self.process_operand(lhs)?,
                            pred,
                            self.process_operand(rhs)?,
                        );
                        self.push_assign(inst.into(), assign)?;
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
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::LoadTraceInput { assign, type_, off } => {
                        let off = self
                            .lexer
                            .span_str(off)
                            .parse::<u32>()
                            .map_err(|e| self.error_at_span(off, &e.to_string()))?;
                        let inst = LoadTraceInputInst::new(off, self.process_type(type_)?);
                        self.push_assign(inst.into(), assign)?;
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
                        self.push_assign(inst.into(), assign)?;
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
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::SExt { assign, type_, val } => {
                        let inst =
                            SExtInst::new(&self.process_operand(val)?, self.process_type(type_)?);
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::SIToFP { assign, type_, val } => {
                        let inst =
                            SIToFPInst::new(&self.process_operand(val)?, self.process_type(type_)?);
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::FPExt { assign, type_, val } => {
                        let inst =
                            FPExtInst::new(&self.process_operand(val)?, self.process_type(type_)?);
                        self.push_assign(inst.into(), assign)?;
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
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::Select {
                        assign,
                        cond,
                        trueval,
                        falseval,
                    } => {
                        let inst = SelectInst::new(
                            self.process_operand(cond)?,
                            self.process_operand(trueval)?,
                            self.process_operand(falseval)?,
                        );
                        self.push_assign(inst.into(), assign)?;
                    }
                    ASTInst::Proxy { assign, val } => {
                        let op = self.process_operand(val)?;
                        let inst = match op {
                            Operand::Local(_) => todo!(),
                            Operand::Const(cidx) => Inst::ProxyConst(cidx),
                        };
                        self.push_assign(inst.into(), assign)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn process_func_types(&mut self, func_types: Vec<ASTFuncType>) -> Result<(), Box<dyn Error>> {
        for ASTFuncType {
            name: name_span,
            arg_tys,
            is_varargs,
            rtn_ty,
        } in func_types
        {
            let name = self.lexer.span_str(name_span).to_owned();
            if self.func_types_map.get(&name).is_some() {
                return Err(format!("Duplicate function type '{name}'").into());
            }
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
            self.func_types_map.insert(name, func_ty);
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
            if self.func_types_map.get(&name).is_some() {
                todo!();
            }
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
                let [val, width] = <[&str; 2]>::try_from(s.split('i').collect::<Vec<_>>()).unwrap();
                let width = width
                    .parse::<u32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let val = if val.starts_with("-") {
                    let val = val
                        .parse::<i64>()
                        .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                    if width < 64
                        && (val < -((1 << width) - 1) / 2 - 1 || val >= ((1 << width) - 1) / 2)
                    {
                        return Err(self.error_at_span(span,
                          &format!("Signed constant {val} exceeds the bit width {width} of the integer type")));
                    }
                    val as u64
                } else {
                    let val = val
                        .parse::<u64>()
                        .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                    if width < 64 && val > (1 << width) - 1 {
                        return Err(self.error_at_span(span,
                          &format!("Unsigned constant {val} exceeds the bit width {width} of the integer type")));
                    }
                    val
                };
                let tyidx = self.m.insert_ty(Ty::Integer(width)).unwrap();
                Ok(Operand::Const(
                    self.m
                        .insert_const(Const::Int(tyidx, val))
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
            ASTOperand::ConstFloat(span) => {
                // unwrap must succeed if the parser is correct.
                let s = self.lexer.span_str(span).strip_suffix("float").unwrap();
                let val = s
                    .parse::<f32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let tyidx = self.m.insert_ty(Ty::Float(FloatTy::Float)).unwrap();
                Ok(Operand::Const(
                    self.m
                        .insert_const(Const::Float(tyidx, val as f64))
                        .map_err(|e| self.error_at_span(span, &e.to_string()))?,
                ))
            }
            ASTOperand::ConstDouble(span) => {
                // unwrap must succeed if the parser is correct.
                let s = self.lexer.span_str(span).strip_suffix("double").unwrap();
                let val = s
                    .parse::<f64>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let tyidx = self.m.insert_ty(Ty::Float(FloatTy::Double)).unwrap();
                Ok(Operand::Const(
                    self.m
                        .insert_const(Const::Float(tyidx, val))
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
                    self.error_at_span(span, &format!("Undefined local operand '%{idx}'"))
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
            ASTType::Float(span) => Ok(self
                .m
                .insert_ty(Ty::Float(FloatTy::Float))
                .map_err(|e| self.error_at_span(span, &e.to_string()))?),
            ASTType::Double(span) => Ok(self
                .m
                .insert_ty(Ty::Float(FloatTy::Double))
                .map_err(|e| self.error_at_span(span, &e.to_string()))?),
            ASTType::Ptr => Ok(self.m.ptr_tyidx()),
            ASTType::Void => Ok(self.m.void_tyidx()),
        }
    }

    /// Push `inst` to the end of [self.m] and add an assignment of a local operand (e.g. `%3 =
    /// ...`). Note: this must be called *before* inserting the instruction, as it allowws this
    /// function to capture cases where users try referencing the variable itself (e.g. `%0 = %0`
    /// will be caught as an error by this function).
    fn push_assign(&mut self, inst: Inst, span: Span) -> Result<(), Box<dyn Error>> {
        let idx = self.lexer.span_str(span)[1..]
            .parse::<usize>()
            .map_err(|e| self.error_at_span(span, &e.to_string()))?;
        let idx = InstIdx::new(idx).map_err(|e| self.error_at_span(span, &e.to_string()))?;
        self.m.push(inst).unwrap();
        match self.inst_idx_map.insert(idx, self.m.last_inst_idx()) {
            None => Ok(()),
            Some(_) => Err(format!("Local operand '%{idx}' redefined").into()),
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

struct ASTFuncType {
    name: Span,
    arg_tys: Vec<ASTType>,
    is_varargs: bool,
    rtn_ty: ASTType,
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
    Guard {
        operand: ASTOperand,
        is_true: bool,
        live_vars: Vec<Span>,
    },
    ICall {
        assign: Option<Span>,
        func_type: Span,
        target: ASTOperand,
        args: Vec<ASTOperand>,
    },
    ICmp {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        pred: Predicate,
        lhs: ASTOperand,
        rhs: ASTOperand,
    },
    FCmp {
        assign: Span,
        type_: ASTType,
        pred: FloatPredicate,
        lhs: ASTOperand,
        rhs: ASTOperand,
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
    SIToFP {
        assign: Span,
        type_: ASTType,
        val: ASTOperand,
    },
    FPExt {
        assign: Span,
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
    Select {
        assign: Span,
        cond: ASTOperand,
        trueval: ASTOperand,
        falseval: ASTOperand,
    },
    Proxy {
        assign: Span,
        val: ASTOperand,
    },
}

#[derive(Debug)]
enum ASTOperand {
    Local(Span),
    ConstInt(Span),
    ConstPtr(Span),
    ConstFloat(Span),
    ConstDouble(Span),
}

#[derive(Debug)]
enum ASTType {
    Int(Span),
    Float(Span),
    Double(Span),
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
        let i16_tyidx = m.insert_ty(Ty::Integer(16)).unwrap();
        let op1 = m
            .push_and_make_operand(LoadTraceInputInst::new(0, i16_tyidx).into())
            .unwrap();
        let op2 = m
            .push_and_make_operand(LoadTraceInputInst::new(16, i16_tyidx).into())
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
            func_type ft1(i8, i32, ...) -> i64
            func_decl f1()
            func_decl f2(i8) -> i32
            func_decl f3(i8, i32, ...) -> i64
            func_decl f4(...)
            entry:
              %0: i32 = load_ti 0
              %1: i32 = trunc %0
              %2: i32 = add %0, %1
              %4: i1 = eq %1, %2
              tloop_start
              guard true, %4, [%0, %1, %2]
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
              %1999: float = load_ti 0
              %20: float = fadd %1999, %1999
              %21: float = fdiv %1999, %1999
              %22: float = fmul %1999, %1999
              %23: float = frem %1999, %1999
              %24: float = fsub %1999, %1999
              %25: i32 = lshr %0, %1
              %26: i32 = sdiv %0, %1
              %27: i32 = srem %0, %1
              %28: i32 = udiv %0, %1
              %29: i32 = urem %0, %1
              %30: i8 = load_ti 4
              %31: i16 = load_ti 5
              %32: i32 = load_ti 5
              %33: i64 = load_ti 6
              %34: i8 = add %30, 255i8
              %35: i16 = add %31, 32768i16
              %36: i32 = add %32, 2147483648i32
              %37: i64 = add %33, 9223372036854775808i64
              *%9 = 0x0
              *%9 = 0xFFFFFFFF
              %38: i1 = ne %1, %2
              %40: i1 = ugt %1, %2
              %41: i1 = uge %1, %2
              %42: i1 = ult %1, %2
              %43: i1 = ule %1, %2
              %44: i1 = sgt %1, %2
              %45: i1 = sge %1, %2
              %46: i1 = slt %1, %2
              %47: i1 = sle %1, %2
              %48: i32 = load_ti 7
              %49: float = si_to_fp %48
              %50: double = fp_ext %49
              %51: double = fadd 1double, 2.345double
              %52: float = fadd 1float, 2.345float
              %53: i64 = icall<ft1> %9(%5, %7, %0)
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

        let f1_tyidx = m
            .insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_tyidx(), false)))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f1".to_owned(), f1_tyidx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let i32_tyidx = m.insert_ty(Ty::Integer(32)).unwrap();
        let f2_tyidx = m
            .insert_ty(Ty::Func(FuncTy::new(
                vec![m.int8_tyidx()],
                i32_tyidx,
                false,
            )))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f2".to_owned(), f2_tyidx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let i64_tyidx = m.insert_ty(Ty::Integer(64)).unwrap();
        let f3_tyidx = m
            .insert_ty(Ty::Func(FuncTy::new(
                vec![m.int8_tyidx(), i32_tyidx],
                i64_tyidx,
                true,
            )))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f3".to_owned(), f3_tyidx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);

        let f4_tyidx = m
            .insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_tyidx(), true)))
            .unwrap();
        m.insert_func_decl(FuncDecl::new("f4".to_owned(), f4_tyidx))
            .unwrap();
        assert_eq!(m.func_decls_len(), 4);
    }

    #[test]
    fn func_types() {
        let mut m = Module::from_str(
            "
              func_type f1()
              func_type f2(i8) -> i32
              func_type f3(i8, i32, ...) -> i64
              func_type f4(...)
        ",
        );
        // We don't have a "lookup a function type" function, but we can check that if we
        // "insert" identical IR function declarations to the module that no new function
        // types are actually added. We thus need to add all the non-function-types we're
        // interested in first.
        let i32_tyidx = m.insert_ty(Ty::Integer(32)).unwrap();
        let i64_tyidx = m.insert_ty(Ty::Integer(64)).unwrap();
        let types_len = m.types_len();

        m.insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_tyidx(), false)))
            .unwrap();
        assert_eq!(m.types_len(), types_len);

        m.insert_ty(Ty::Func(FuncTy::new(
            vec![m.int8_tyidx()],
            i32_tyidx,
            false,
        )))
        .unwrap();
        assert_eq!(m.types_len(), types_len);

        m.insert_ty(Ty::Func(FuncTy::new(
            vec![m.int8_tyidx(), i32_tyidx],
            i64_tyidx,
            true,
        )))
        .unwrap();
        assert_eq!(m.types_len(), types_len);

        m.insert_ty(Ty::Func(FuncTy::new(Vec::new(), m.void_tyidx(), true)))
            .unwrap();
        assert_eq!(m.types_len(), types_len);
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

    #[test]
    #[should_panic(expected = "Duplicate function type 't1'")]
    fn duplicate_func_type() {
        Module::from_str(
            "
          func_type t1()
          func_type t1()
          entry:
            %0: i8 = load_ti 0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "No such function type 't2'")]
    fn no_such_func_type() {
        Module::from_str(
            "
          func_type t1()
          entry:
            %0: ptr = load_ti 0
            icall<t2> %0()
        ",
        );
    }

    #[test]
    #[should_panic(expected = "ICmp instructions must assign to an i1, not 'i8'")]
    fn icmp_assign_to_non_i1() {
        Module::from_str(
            "
          entry:
            %0: i8 = ult 1i8, 2i8
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Signed constant -129 exceeds the bit width 8 of the integer type")]
    fn invalid_numbers1() {
        Module::from_str(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = add %0, -128i8
            %2: i8 = add %0, -129i8
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Unsigned constant 256 exceeds the bit width 8 of the integer type")]
    fn invalid_numbers2() {
        Module::from_str(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = add %0, 255i8
            %2: i8 = add %0, 256i8
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Binop result type incorrect for one or more operands")]
    fn invalid_binop_result_ty() {
        Module::from_str(
            "
          entry:
            %0: float = fadd 1.2double, 3.4double
            ",
        );
    }

    #[test]
    #[should_panic(expected = "No such local variable %1")]
    fn invalid_guard_inst_ref() {
        Module::from_str(
            "
          entry:
            %0: i1 = load_ti 0
            guard true, %0, [%1]
            ",
        );
    }
}
