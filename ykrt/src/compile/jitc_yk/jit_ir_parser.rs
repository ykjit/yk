//! A basic parser for JIT IR.
//!
//! This module -- which is only intended to be compiled-in in `cfg(test)` -- adds a `from_str`
//! method to [Module] which takes in JIT IR as a string, parses it, and produces a [Module]. This
//! makes it possible to write JIT IR tests using JIT IR concrete syntax.

use super::jit_ir::{
    AddInst, InstIdx, IntegerTy, LoadTraceInputInst, Module, Operand, TestUseInst, Ty, TyIdx,
};
use fm::FMBuilder;
use lrlex::{lrlex_mod, DefaultLexerTypes, LRNonStreamingLexer};
use lrpar::{lrpar_mod, NonStreamingLexer, Span};
use regex::Regex;
use std::{error::Error, sync::OnceLock};

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
            Some(Ok((globals, bblocks))) => JITIRParser { lexer: &lexer }
                .process(globals, bblocks)
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
}

impl<'lexer, 'input: 'lexer> JITIRParser<'lexer, 'input> {
    fn process(
        &self,
        _globals: Vec<()>,
        bblocks: Vec<ASTBBlock>,
    ) -> Result<Module, Box<dyn Error>> {
        let mut m = Module::new_testing();
        for bblock in bblocks {
            for inst in bblock.insts {
                match inst {
                    ASTInst::Add { type_: _, lhs, rhs } => {
                        let inst =
                            AddInst::new(self.process_operand(lhs)?, self.process_operand(rhs)?);
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::LoadTraceInput { type_, off } => {
                        let off = self
                            .lexer
                            .span_str(off)
                            .parse::<u32>()
                            .map_err(|e| self.error_at_span(off, &e.to_string()))?;
                        let inst = LoadTraceInputInst::new(off, self.process_type(&mut m, type_)?);
                        m.push(inst.into()).unwrap();
                    }
                    ASTInst::TestUse(op) => {
                        let inst = TestUseInst::new(self.process_operand(op)?);
                        m.push(inst.into()).unwrap();
                    }
                }
            }
        }
        Ok(m)
    }

    fn process_operand(&self, op: ASTOperand) -> Result<Operand, Box<dyn Error>> {
        match op {
            ASTOperand::Local(span) => {
                let idx = self.lexer.span_str(span)[1..]
                    .parse::<usize>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                Ok(Operand::Local(
                    InstIdx::new(idx).map_err(|e| self.error_at_span(span, &e.to_string()))?,
                ))
            }
        }
    }

    fn process_type(&self, m: &mut Module, type_: ASTType) -> Result<TyIdx, Box<dyn Error>> {
        match type_ {
            ASTType::Int(span) => {
                let width = self.lexer.span_str(span)[1..]
                    .parse::<u32>()
                    .map_err(|e| self.error_at_span(span, &e.to_string()))?;
                let ty = IntegerTy::new(width);
                m.insert_ty(Ty::Integer(ty))
                    .map_err(|e| self.error_at_span(span, &e.to_string()))
            }
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

#[derive(Debug)]
struct ASTBBlock {
    #[allow(dead_code)]
    label: Span,
    insts: Vec<ASTInst>,
}

#[derive(Debug)]
enum ASTInst {
    Add {
        #[allow(dead_code)]
        type_: ASTType,
        lhs: ASTOperand,
        rhs: ASTOperand,
    },
    LoadTraceInput {
        type_: ASTType,
        off: Span,
    },
    TestUse(ASTOperand),
}

#[derive(Debug)]
enum ASTOperand {
    Local(Span),
}

#[derive(Debug)]
enum ASTType {
    Int(Span),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::jitc_yk::jit_ir::Ty;

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
}
