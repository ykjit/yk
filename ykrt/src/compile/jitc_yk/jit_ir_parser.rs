//! A basic parser for JIT IR.
//!
//! This module -- which is only intended to be compiled-in in `cfg(test)` -- adds a `from_str`
//! method to [Module] which takes in JIT IR as a string, parses it, and produces a [Module]. This
//! makes it possible to write JIT IR tests using JIT IR concrete syntax.

use super::{
    aot_ir::Predicate,
    jit_ir::{
        AddInst, DirectCallInst, FuncDecl, FuncTy, GuardInfo, GuardInst, IcmpInst, Inst, InstIdx,
        IntegerTy, LoadTraceInputInst, Module, Operand, SRemInst, TestUseInst, TruncInst, Ty,
        TyIdx,
    },
};
use fm::FMBuilder;
use lrlex::{lrlex_mod, DefaultLexerTypes, LRNonStreamingLexer};
use lrpar::{lrpar_mod, NonStreamingLexer, Span};
use regex::Regex;
use std::{collections::HashMap, error::Error, sync::OnceLock};

lrlex_mod!("compile/jitc_yk/jit_ir.l");
lrpar_mod!("compile/jitc_yk/jit_ir.y");

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
            Some(Ok((func_decls, globals, bblocks))) => JITIRParser {
                lexer: &lexer,
                inst_idx_map: HashMap::new(),
            }
            .process(func_decls, globals, bblocks)
            .unwrap(),
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

struct JITIRParser<'lexer, 'input: 'lexer> {
    lexer: &'lexer LRNonStreamingLexer<'lexer, 'input, DefaultLexerTypes<StorageT>>,
    /// A mapping from local operands (e.g. `%3`) in the input text to instruction offsets. For
    /// example if the first instruction in the input is `%3 = ...` the map will be `%3 -> %0`
    /// indicating that whenever we see `%3` in the input we map that to `%0` in the IR we're
    /// generating. To populate this map, each instruction that defines a new local operand needs
    /// to call [Self::add_assign].
    inst_idx_map: HashMap<InstIdx, InstIdx>,
}

impl<'lexer, 'input: 'lexer> JITIRParser<'lexer, 'input> {
    fn process(
        &mut self,
        func_decls: Vec<ASTFuncDecl>,
        _globals: Vec<()>,
        bblocks: Vec<ASTBBlock>,
    ) -> Result<Module, Box<dyn Error>> {
        let mut m = Module::new_testing();
        self.process_func_decls(&mut m, func_decls)?;

        for bblock in bblocks.into_iter() {
            for inst in bblock.insts {
                match inst {
                    ASTInst::Add {
                        assign,
                        type_: _,
                        lhs,
                        rhs,
                    } => {
                        let inst =
                            AddInst::new(self.process_operand(lhs)?, self.process_operand(rhs)?);
                        self.add_assign(m.len(), assign)?;
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::Call {
                        assign,
                        name: name_span,
                        args,
                    } => {
                        let name = &self.lexer.span_str(name_span)[1..];
                        let fd_idx = m.find_func_decl_idx_by_name(name);
                        let inst =
                            DirectCallInst::new(&mut m, fd_idx, self.process_operands(args)?)
                                .map_err(|e| self.error_at_span(name_span, &e.to_string()))?;
                        if let Some(x) = assign {
                            self.add_assign(m.len(), x)?;
                        }
                        m.push(inst.into()).unwrap();
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
                        self.add_assign(m.len(), assign)?;
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::Guard { operand, is_true } => {
                        let gidx = m
                            .push_guardinfo(GuardInfo::new(Vec::new(), Vec::new()))
                            .unwrap();
                        let inst = GuardInst::new(self.process_operand(operand)?, is_true, gidx);
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::LoadTraceInput { assign, type_, off } => {
                        let off = self
                            .lexer
                            .span_str(off)
                            .parse::<u32>()
                            .map_err(|e| self.error_at_span(off, &e.to_string()))?;
                        let inst = LoadTraceInputInst::new(off, self.process_type(&mut m, type_)?);
                        self.add_assign(m.len(), assign)?;
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::SRem {
                        assign,
                        type_: _,
                        lhs,
                        rhs,
                    } => {
                        let inst =
                            SRemInst::new(self.process_operand(lhs)?, self.process_operand(rhs)?);
                        self.add_assign(m.len(), assign)?;
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::TestUse(op) => {
                        let inst = TestUseInst::new(self.process_operand(op)?);
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::TraceLoopStart => {
                        m.push(Inst::TraceLoopStart).unwrap();
                    }
                    ASTInst::Trunc {
                        assign,
                        type_,
                        operand,
                    } => {
                        let inst = TruncInst::new(
                            &self.process_operand(operand)?,
                            self.process_type(&mut m, type_)?,
                        );
                        self.add_assign(m.len(), assign)?;
                        m.push(inst.into()).unwrap();
                    }
                }
            }
        }
        Ok(m)
    }

    fn process_func_decls(
        &mut self,
        m: &mut Module,
        func_decls: Vec<ASTFuncDecl>,
    ) -> Result<(), Box<dyn Error>> {
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
                    mapped.push(self.process_type(m, x)?);
                }
                mapped
            };
            let rtn_ty = self.process_type(m, rtn_ty)?;
            let func_ty = m
                .insert_ty(Ty::Func(FuncTy::new(arg_tys, rtn_ty, is_varargs)))
                .map_err(|e| self.error_at_span(name_span, &e.to_string()))?;
            m.insert_func_decl(FuncDecl::new(name, func_ty))
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

    fn process_type(&mut self, m: &mut Module, type_: ASTType) -> Result<TyIdx, Box<dyn Error>> {
        match type_ {
            ASTType::Int(span) => {
                let width = self.lexer.span_str(span)[1..]
                    .parse::<u32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let ty = IntegerTy::new(width);
                m.insert_ty(Ty::Integer(ty))
                    .map_err(|e| self.error_at_span(span, &e.to_string()))
            }
            ASTType::Void => Ok(m.void_ty_idx()),
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
    Add {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
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
    LoadTraceInput {
        assign: Span,
        type_: ASTType,
        off: Span,
    },
    SRem {
        assign: Span,
        #[allow(dead_code)]
        type_: ASTType,
        lhs: ASTOperand,
        rhs: ASTOperand,
    },
    TestUse(ASTOperand),
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
}

#[derive(Debug)]
enum ASTType {
    Void,
    Int(Span),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::jitc_yk::jit_ir::{FuncTy, Ty};

    #[test]
    fn roundtrip() {
        let mut m = Module::new_testing();
        let i16_ty_idx = m.insert_ty(Ty::Integer(IntegerTy::new(16))).unwrap();
        let op1 = m
            .push_and_make_operand(LoadTraceInputInst::new(0, i16_ty_idx).into())
            .unwrap();
        let op2 = m
            .push_and_make_operand(LoadTraceInputInst::new(16, i16_ty_idx).into())
            .unwrap();
        let op3 = m
            .push_and_make_operand(AddInst::new(op1.clone(), op2.clone()).into())
            .unwrap();
        let op4 = m
            .push_and_make_operand(AddInst::new(op1.clone(), op3.clone()).into())
            .unwrap();
        m.push(TestUseInst::new(op3).into()).unwrap();
        m.push(TestUseInst::new(op4).into()).unwrap();
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
              %3: i16 = srem %1, %2
              %4: i16 = eq %1, %2
              tloop_start
              guard %4, true
              call @f1()
              %5: i8 = load_ti 1
              %6: i32 = call @f2(%1)
              %7: i32 = load_ti 2
              %8: i64 = call @f3(%5, %7, %0)
              call @f4(%0, %1)
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

        let i32_ty_idx = m.insert_ty(Ty::Integer(IntegerTy::new(32))).unwrap();
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

        let i64_ty_idx = m.insert_ty(Ty::Integer(IntegerTy::new(64))).unwrap();
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
