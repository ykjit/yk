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
use std::error::Error;

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
            Some(Ok((globals, bblocks))) => process(lexer, globals, bblocks).unwrap(),
            _ => panic!("Could not produce JIT Module."),
        }
    }

    /// Assert that for the given IR input, as a string, `ir_input`, when transformed by
    /// `ir_transform(Module)` it produces a [Module] that when printed corresponds to the [fm]
    /// pattern `transformed_ptn`.
    pub(crate) fn assert_ir_transform_eq<F>(ir_input: &str, ir_transform: F, transformed_ptn: &str)
    where
        F: FnOnce(Module) -> Module,
    {
        let m = Self::from_str(ir_input);
        let m = ir_transform(m);
        let fmm = FMBuilder::new(transformed_ptn).unwrap().build().unwrap();
        if let Err(e) = fmm.matches(&m.to_string()) {
            panic!("{e}");
        }
    }
}

fn process(
    lexer: LRNonStreamingLexer<DefaultLexerTypes<StorageT>>,
    _globals: Vec<()>,
    bblocks: Vec<ASTBBlock>,
) -> Result<Module, Box<dyn Error>> {
    let mut m = Module::new_testing();
    for bblock in bblocks {
        for inst in bblock.insts {
            match inst {
                ASTInst::Add { type_: _, lhs, rhs } => {
                    let inst =
                        AddInst::new(process_operand(&lexer, lhs)?, process_operand(&lexer, rhs)?);
                    m.push(inst.into()).unwrap();
                }
                ASTInst::LoadTraceInput { type_, off } => {
                    let off = lexer
                        .span_str(off)
                        .parse::<u32>()
                        .map_err(|e| error_at_span(&lexer, off, &e.to_string()))?;
                    let inst = LoadTraceInputInst::new(off, process_type(&lexer, &mut m, type_)?);
                    m.push(inst.into()).unwrap();
                }
                ASTInst::TestUse(op) => {
                    let inst = TestUseInst::new(process_operand(&lexer, op)?);
                    m.push(inst.into()).unwrap();
                }
            }
        }
    }
    Ok(m)
}

fn process_operand(
    lexer: &LRNonStreamingLexer<DefaultLexerTypes<StorageT>>,
    op: ASTOperand,
) -> Result<Operand, Box<dyn Error>> {
    match op {
        ASTOperand::Local(span) => {
            let idx = lexer.span_str(span)[1..]
                .parse::<usize>()
                .map_err(|e| error_at_span(lexer, span, &e.to_string()))?;
            Ok(Operand::Local(
                InstIdx::new(idx).map_err(|e| error_at_span(lexer, span, &e.to_string()))?,
            ))
        }
    }
}

fn process_type(
    lexer: &LRNonStreamingLexer<DefaultLexerTypes<StorageT>>,
    m: &mut Module,
    type_: ASTType,
) -> Result<TyIdx, Box<dyn Error>> {
    match type_ {
        ASTType::Int(span) => {
            let width = lexer.span_str(span)[1..]
                .parse::<u32>()
                .map_err(|e| error_at_span(lexer, span, &e.to_string()))?;
            let ty = IntegerTy::new(width);
            m.insert_ty(Ty::Integer(ty))
                .map_err(|e| error_at_span(lexer, span, &e.to_string()))
        }
    }
}

/// Return an error message pinpointing `span` as the culprit.
fn error_at_span(
    lexer: &LRNonStreamingLexer<DefaultLexerTypes<StorageT>>,
    span: Span,
    msg: &str,
) -> Box<dyn Error> {
    let ((line_off, col), _) = lexer.line_col(span);
    let code = lexer
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
}
