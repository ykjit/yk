%start Module

%%

Module -> Result<(Vec<()>, Vec<ASTBBlock>), Box<dyn Error>>:
    Globals BBlocks {
      Ok(($1?, $2?))
    }
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
    "LOCAL_OPERAND" ":" Type "=" "LOAD_TI" "INT" "," Type  {
      Ok(ASTInst::LoadTraceInput{type_: $3?, off: $6?.span()})
    }
  | "LOCAL_OPERAND" ":" Type "=" "ADD" Operand "," Operand  {
      Ok(ASTInst::Add{type_: $3?, lhs: $6?, rhs: $8?})
    }
  | "TEST_USE" Operand { Ok(ASTInst::TestUse($2?)) }
  ;

Operand -> Result<ASTOperand, Box<dyn Error>>:
    "LOCAL_OPERAND" { Ok(ASTOperand::Local($1?.span())) }
  ;

Type -> Result<ASTType, Box<dyn Error>>:
    "INT_TYPE" { Ok(ASTType::Int($1?.span())) }
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
