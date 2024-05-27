%start Module

%%

Module -> Result<(Vec<ASTFuncDecl>, Vec<()>, Vec<ASTBBlock>), Box<dyn Error>>:
    FuncDecls Globals BBlocks {
      Ok(($1?, $2?, $3?))
    }
  ;

FuncDecls -> Result<Vec<ASTFuncDecl>, Box<dyn Error>>:
    FuncDecl FuncDecls { flatten($1, $2) }
  | { Ok(Vec::new()) }
  ;

FuncDecl -> Result<ASTFuncDecl, Box<dyn Error>>:
    "FUNC_DECL" "ID" "(" FuncArgs ")" FuncRtnType {
      let (arg_tys, is_varargs) = $4?;
      Ok(ASTFuncDecl{name: $2?.span(), arg_tys, is_varargs, rtn_ty: $6?})
    }
  ;

FuncArgs -> Result<(Vec<ASTType>, bool), Box<dyn Error>>:
    NormalFuncArgs "," "..." { Ok(($1?, true)) }
  | NormalFuncArgs "," Type { Ok((flattenr($1, $3)?, false)) }
  | Type { Ok((vec![$1?], false)) }
  | "..." { Ok((Vec::new(), true)) }
  | { Ok((Vec::new(), false)) }
  ;

NormalFuncArgs -> Result<Vec<ASTType>, Box<dyn Error>>:
    NormalFuncArgs "," Type { flattenr($1, $3) }
  | Type { Ok(vec![$1?]) }
  ;

FuncRtnType -> Result<ASTType, Box<dyn Error>>:
    "->" Type { $2 }
  | { Ok(ASTType::Void) }
  ;

Globals -> Result<Vec<()>, Box<dyn Error>>:
    "GLOBAL" Globals { todo!() }
  | { Ok(Vec::new()) }
  ;

BBlocks -> Result<Vec<ASTBBlock>, Box<dyn Error>>:
    BBlock BBlocks { flatten($1, $2) }
  | { Ok(Vec::new()) }
  ;

BBlock -> Result<ASTBBlock, Box<dyn Error>>:
    "LABEL" Insts { Ok(ASTBBlock{ label: $1?.span(), insts: $2?}) }
  ;

Insts -> Result<Vec<ASTInst>, Box<dyn Error>>:
    Inst Insts { flatten($1, $2) }
  | { Ok(Vec::new()) }
  ;

Inst -> Result<ASTInst, Box<dyn Error>>:
    "GUARD" Operand "," "TRUE" {
      Ok(ASTInst::Guard{operand: $2?, is_true: true})
    }
  | "GUARD" Operand "," "FALSE" {
      Ok(ASTInst::Guard{operand: $2?, is_true: false})
    }
  | "LOCAL_OPERAND" ":" Type "=" "LOAD_TI" "INT" {
      Ok(ASTInst::LoadTraceInput{assign: $1?.span(), type_: $3?, off: $6?.span()})
    }
  | "LOCAL_OPERAND" ":" Type "=" "ADD" Operand "," Operand  {
      Ok(ASTInst::Add{assign: $1?.span(), type_: $3?, lhs: $6?, rhs: $8?})
    }
  | "LOCAL_OPERAND" ":" Type "=" "CALL" "GLOBAL" "(" CallArgs ")" {
      Ok(ASTInst::Call{assign: Some($1?.span()), name: $6?.span(), args: $8?})
    }
  | "CALL" "GLOBAL" "(" CallArgs ")" {
      Ok(ASTInst::Call{assign: None, name: $2?.span(), args: $4?})
    }
  | "LOCAL_OPERAND" ":" Type "=" "EQ" Operand "," Operand  {
      Ok(ASTInst::Eq{assign: $1?.span(), type_: $3?, lhs: $6?, rhs: $8?})
    }
  | "LOCAL_OPERAND" ":" Type "=" "SREM" Operand "," Operand {
      Ok(ASTInst::SRem{assign: $1?.span(), type_: $3?, lhs: $6?, rhs: $8?})
    }
  | "LOCAL_OPERAND" ":" Type "=" "TRUNC" Operand {
      Ok(ASTInst::Trunc{assign: $1?.span(), type_: $3?, operand: $6? })
    }
  | "TEST_USE" Operand { Ok(ASTInst::TestUse($2?)) }
  | "TLOOP_START" { Ok(ASTInst::TraceLoopStart) }
  ;

Operand -> Result<ASTOperand, Box<dyn Error>>:
    "LOCAL_OPERAND" { Ok(ASTOperand::Local($1?.span())) }
  ;

Type -> Result<ASTType, Box<dyn Error>>:
    "INT_TYPE" { Ok(ASTType::Int($1?.span())) }
  | "PTR" { Ok(ASTType::Ptr) }
  ;

CallArgs -> Result<Vec<ASTOperand>, Box<dyn Error>>:
    CallArgs "," Operand { flattenr($1, $3) }
  | Operand { Ok(vec![$1?]) }
  | { Ok(Vec::new()) }
  ;

%%

use crate::compile::jitc_yk::jit_ir_parser::*;
use std::error::Error;

fn flatten<T>(lhs: Result<T, Box<dyn Error>>, rhs: Result<Vec<T>, Box<dyn Error>>)
  -> Result<Vec<T>, Box<dyn Error>>
{
    let lhs = lhs?;
    let mut rhs = rhs?;
    let mut out = Vec::with_capacity(rhs.len() + 1);
    out.push(lhs);
    out.append(&mut rhs);
    Ok(out)
}

fn flattenr<T>(lhs: Result<Vec<T>, Box<dyn Error>>, rhs: Result<T, Box<dyn Error>>)
  -> Result<Vec<T>, Box<dyn Error>>
{
    let mut lhs = lhs?;
    let rhs = rhs?;
    lhs.push(rhs);
    Ok(lhs)
}
