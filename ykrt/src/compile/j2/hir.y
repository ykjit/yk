%start Module
%expect-unused Unmatched "UNMATCHED"

%%

Module -> Result<(Vec<AstExtern>, Vec<AstInst>), Box<dyn Error>>:
    Externs Insts {
      Ok(($1?, $2?))
    }
  ;

Externs -> Result<Vec<AstExtern>, Box<dyn Error>>:
    Externs Extern { flattenr($1, $2) }
  | { Ok(Vec::new()) }
  ;

Extern -> Result<AstExtern, Box<dyn Error>>:
    "EXTERN" "ID" "(" FuncArgs ")" FuncRtnType {
      let (arg_tys, has_varargs) = $4?;
      let ty = AstFuncTy { arg_tys, has_varargs, rtn_ty: $6? };
      Ok(AstExtern{name: $2?.span(), ty})
    }
  ;

FuncArgs -> Result<(Vec<AstTy>, bool), Box<dyn Error>>:
    NormalFuncArgs "," "..." { Ok(($1?, true)) }
  | NormalFuncArgs "," Ty { Ok((flattenr($1, $3)?, false)) }
  | Ty { Ok((vec![$1?], false)) }
  | "..." { Ok((Vec::new(), true)) }
  | { Ok((Vec::new(), false)) }
  ;

NormalFuncArgs -> Result<Vec<AstTy>, Box<dyn Error>>:
    NormalFuncArgs "," Ty { flattenr($1, $3) }
  | Ty { Ok(vec![$1?]) }
  ;

FuncRtnType -> Result<AstTy, Box<dyn Error>>:
    "->" Ty { $2 }
  | { Ok(AstTy::Void) }
  ;

Insts -> Result<Vec<AstInst>, Box<dyn Error>>:
    Insts Inst { flattenr($1, $2) }
  | { Ok(Vec::new()) }
  ;

Inst -> Result<AstInst, Box<dyn Error>>:
    "BLACKBOX" "LOCAL" { Ok(AstInst::Blackbox($2?.span())) }
  | "CALL" "ID" "LOCAL" "(" Locals ")" {
      Ok(AstInst::Call { local: None, ty: None, extern_: $2?.span(), tgt: $3?.span(), args: $5? })
    }
  | "EXIT" "[" Locals "]" {
      Ok(AstInst::Exit { locals: $3? })
    }
  | "GUARD" "FALSE" "," "LOCAL" "," "[" Locals "]" {
       Ok(AstInst::Guard { expect: false, cond: $4?.span(), entry_vars: $7? })
    }
  | "GUARD" "TRUE" "," "LOCAL" "," "[" Locals "]" {
      Ok(AstInst::Guard { expect: true, cond: $4?.span(), entry_vars: $7? })
    }
  | "LOCAL" ":" Ty "=" "CALL" "ID" "LOCAL" "(" Locals ")" {
      Ok(AstInst::Call { local: Some($1?.span()), ty: Some($3?), extern_: $6?.span(), tgt: $7?.span(), args: $9? })
    }
  | "LOCAL" ":" Ty "=" "ABS" "LOCAL" {
      Ok(AstInst::Abs { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "ADD" "LOCAL" "," "LOCAL" {
       Ok(AstInst::Add { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "AND" "LOCAL" "," "LOCAL" {
      Ok(AstInst::And { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "ASHR" "LOCAL" "," "LOCAL" {
      Ok(AstInst::AShr { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "ARG" "[" ArgList "]" {
      Ok(AstInst::Arg { local: $1?.span(), ty: $3?, vlocs: $7? })
    }
  | "LOCAL" ":" Ty "=" Const {
       Ok(AstInst::Const { local: $1?.span(), ty: $3?, kind: $5? })
    }
  | "LOCAL" ":" Ty "=" "CTPOP" "LOCAL" {
      Ok(AstInst::CtPop { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "="  "DYNPTRADD" "LOCAL" "," "LOCAL" "," "INT" {
       Ok(AstInst::DynPtrAdd { local: $1?.span(), ty: $3?, ptr: $6?.span(), num_elems: $8?.span(), elem_size: $10?.span() })
    }
  | "LOCAL" ":" Ty "=" "FADD" "LOCAL" "," "LOCAL" {
       Ok(AstInst::FAdd { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "FCMP" FPred "LOCAL" "," "LOCAL" {
      Ok(AstInst::FCmp{ local: $1?.span(), ty: $3?, pred: $6?, lhs: $7?.span(), rhs: $9?.span() })
    }
  | "LOCAL" ":" Ty "=" "FDIV" "LOCAL" "," "LOCAL" {
       Ok(AstInst::FDiv { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "FMUL" "LOCAL" "," "LOCAL" {
       Ok(AstInst::FMul { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "FSUB" "LOCAL" "," "LOCAL" {
       Ok(AstInst::FSub { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "FPEXT" "LOCAL" {
      Ok(AstInst::FPExt { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "FPTOSI" "LOCAL" {
      Ok(AstInst::FPToSI { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "GLOBAL" {
      Ok(AstInst::Global { local: $1?.span(), ty: $3?, name: $5?.span() })
    }
  | "LOCAL" ":" Ty "=" "ICMP" IPred "LOCAL" "," "LOCAL" {
      Ok(AstInst::ICmp{ local: $1?.span(), ty: $3?, pred: $6?, lhs: $7?.span(), rhs: $9?.span() })
    }
  | "LOCAL" ":" Ty "=" "LOAD" "LOCAL" {
      Ok(AstInst::Load { local: $1?.span(), ty: $3?, ptr: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "INTTOPTR" "LOCAL" {
      Ok(AstInst::IntToPtr { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "LSHR" "LOCAL" "," "LOCAL" {
      Ok(AstInst::LShr { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "MUL" "LOCAL" "," "LOCAL" {
       Ok(AstInst::Mul { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "OR" "LOCAL" "," "LOCAL" {
       Ok(AstInst::Or { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "PTRADD" "LOCAL" "," "INT" {
      Ok(AstInst::PtrAdd { local: $1?.span(), ty: $3?, ptr: $6?.span(), off: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "PTRTOINT" "LOCAL" {
      Ok(AstInst::PtrToInt { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "RETURN" {
      Ok(AstInst::Return)
    }
  | "LOCAL" ":" Ty "=" "SELECT" "LOCAL" "," "LOCAL" "," "LOCAL" {
       Ok(AstInst::Select { local: $1?.span(), ty: $3?, cond: $6?.span(), truev: $8?.span(), falsev: $10?.span() })
    }
  | "LOCAL" ":" Ty "=" "SEXT" "LOCAL" {
      Ok(AstInst::SExt { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "SHL" "LOCAL" "," "LOCAL" {
      Ok(AstInst::Shl { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "SITOFP" "LOCAL" {
      Ok(AstInst::SIToFP { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "SREM" "LOCAL" "," "LOCAL" {
       Ok(AstInst::SRem { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "TRUNC" "LOCAL" {
      Ok(AstInst::Trunc { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  | "LOCAL" ":" Ty "=" "SUB" "LOCAL" "," "LOCAL" {
       Ok(AstInst::Sub { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "STORE" "LOCAL" "," "LOCAL" {
      Ok(AstInst::Store { val: $2?.span(), ptr: $4?.span() })
    }
  | "LOCAL" ":" Ty "=" "UDIV" "LOCAL" "," "LOCAL" {
      Ok(AstInst::UDiv { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "XOR" "LOCAL" "," "LOCAL" {
       Ok(AstInst::Xor { local: $1?.span(), ty: $3?, lhs: $6?.span(), rhs: $8?.span() })
    }
  | "LOCAL" ":" Ty "=" "ZEXT" "LOCAL" {
      Ok(AstInst::ZExt { local: $1?.span(), ty: $3?, val: $6?.span() })
    }
  ;

ArgList -> Result<Vec<AstVLoc>, Box<dyn Error>>:
    ArgList "," VLoc { flattenr($1, $3) }
  | VLoc { Ok(vec![$1?]) }
  ;

Const -> Result<AstConst, Box<dyn Error>>:
    "CONST_DOUBLE" { Ok(AstConst::Double($1?.span())) }
  | "CONST_FLOAT" { Ok(AstConst::Float($1?.span())) }
  | "INT" { Ok(AstConst::Int($1?.span())) }
  ;

FPred -> Result<FPred, Box<dyn Error>>:
    "FALSE" { Ok(FPred::False) }
  | "OEQ" { Ok(FPred::Oeq) }
  | "OGT" { Ok(FPred::Ogt) }
  | "OGE" { Ok(FPred::Oge) }
  | "OLT" { Ok(FPred::Olt) }
  | "OLE" { Ok(FPred::Ole) }
  | "ONE" { Ok(FPred::One) }
  | "ORD" { Ok(FPred::Ord) }
  | "UEQ" { Ok(FPred::Ueq) }
  | "UGT" { Ok(FPred::Ugt) }
  | "UGE" { Ok(FPred::Uge) }
  | "ULT" { Ok(FPred::Ult) }
  | "ULE" { Ok(FPred::Ule) }
  | "UNE" { Ok(FPred::Une) }
  | "UNO" { Ok(FPred::Uno) }
  | "TRUE" { Ok(FPred::True) }
  ;

Locals -> Result<Vec<Span>, Box<dyn Error>>:
    LocalsList { $1 }
  | { Ok(vec!()) }
  ;

LocalsList -> Result<Vec<Span>, Box<dyn Error>>:
    Locals "," "LOCAL" { flattenspan($1, $3?.span()) }
  | "LOCAL" { Ok(vec![$1?.span()]) }
  ;

IPred -> Result<IPred, Box<dyn Error>>:
    "EQ" { Ok(IPred::Eq) }
  | "NE" { Ok(IPred::Ne) }
  | "UGT" { Ok(IPred::Ugt) }
  | "UGE" { Ok(IPred::Uge) }
  | "ULT" { Ok(IPred::Ult) }
  | "ULE" { Ok(IPred::Ule) }
  | "SGT" { Ok(IPred::Sgt) }
  | "SGE" { Ok(IPred::Sge) }
  | "SLT" { Ok(IPred::Slt) }
  | "SLE" { Ok(IPred::Sle) }
  ;

Ty -> Result<AstTy, Box<dyn Error>>:
    "INT_TY" { Ok(AstTy::Int($1?.span())) }
  | "FLOAT_TY" { Ok(AstTy::Float) }
  | "DOUBLE_TY" { Ok(AstTy::Double) }
  | "PTR" { Ok(AstTy::Ptr) }
  ;

VLoc -> Result<AstVLoc, Box<dyn Error>>:
    "REG" { Ok(AstVLoc::AutoReg) }
  | "REG" "STRING" { Ok(AstVLoc::Reg($2?.span())) }
  | "STACK" { Ok(AstVLoc::AutoStack) }
  | "STACK" "INT" { Ok(AstVLoc::Stack($2?.span())) }
  | "STACKOFF" "INT" { Ok(AstVLoc::StackOff($2?.span())) }
  ;

Unmatched -> ():
    "UNMATCHED" { }
  ;

%%

use crate::compile::j2::{hir::IPred, hir_parser::*};
use std::error::Error;

fn flattenr<T>(lhs: Result<Vec<T>, Box<dyn Error>>, rhs: Result<T, Box<dyn Error>>)
  -> Result<Vec<T>, Box<dyn Error>>
{
    let mut lhs = lhs?;
    let rhs = rhs?;
    lhs.push(rhs);
    Ok(lhs)
}

fn flattenspan(lhs: Result<Vec<Span>, Box<dyn Error>>, rhs: Span) -> Result<Vec<Span>, Box<dyn Error>>
{
    let mut lhs = lhs?;
    lhs.push(rhs);
    Ok(lhs)
}
