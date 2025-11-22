//! The JIT IR trace builder.
//!
//! This takes in an (AOT IR, execution trace) pair and constructs a JIT IR trace from it.

use super::YkSideTraceInfo;
use super::aot_ir::{self, BBlockId, BinOp, Module};
use super::jit_ir::TraceEndFrame;
use super::{
    AOT_MOD,
    arbbitint::ArbBitInt,
    jit_ir::{self, Const, Operand, PackedOperand, ParamIdx, TraceKind},
};
use crate::compile::jitc_yk::aot_ir::{DeoptSafepoint, TyIdx};
use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{CompilationError, CompiledTrace},
    log::stats::TimingState,
    mt::{MT, TraceId},
    trace::{AOTTraceIterator, TraceAction},
};
use std::{collections::HashMap, ffi::CString, sync::Arc};
use ykaddr::addr::symbol_to_ptr;

/// Caller-saved registers in DWARF notation.
static CALLER_CLOBBER_REG: [u16; 9] = [0, 1, 2, 4, 5, 8, 9, 10, 11];

/// Given an execution trace and AOT IR, creates a JIT IR trace.
pub(crate) struct TraceBuilder {
    /// The AOT IR.
    aot_mod: &'static Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstId, jit_ir::Operand>,
    /// BBlock containing the current control point (i.e. the control point that started this trace).
    cp_block: Option<aot_ir::BBlockId>,
    /// Inlined calls.
    ///
    /// For a valid trace, this always contains at least one element, otherwise the trace returned
    /// out of the function that started tracing, which is problematic.
    frames: Vec<InlinedFrame>,
    /// If `Some`, the block we started inlining at, and the successor block to stop outlining at.
    outline_info: Option<(BBlockId, BBlockId)>,
    /// Current count of recursive calls to the function in which outlining was started. Will be 0
    /// if `outline_target_blk` is None.
    recursion_count: usize,
    /// Values promoted to trace-level constants by `yk_promote_*` and `yk_idempotent_promote_*`.
    /// Values are stored as native-endian sequences of bytes: the AOT code must be examined to
    /// determine the size of a given a value at a given point. Currently (and probably forever)
    /// only values that are a multiple of 8 bits are supported.
    promotions: Box<[u8]>,
    /// The trace's current position in the promotions array.
    promote_idx: usize,
    /// The dynamically recorded debug strings, in the order that the corresponding
    /// `yk_debug_str()` calls were encountered in the trace.
    debug_strs: Vec<String>,
    /// The trace's current position in the [Self::debug_strs] vector.
    debug_str_idx: usize,
    /// Local variables that we have inferred to be constant.
    inferred_consts: HashMap<jit_ir::InstIdx, jit_ir::ConstIdx>,
    /// Did this trace end in another frame?
    endframe: TraceEndFrame,
    /// Determines if interpreter recursion was detected and we need to stop processing any
    /// remaining blocks. Set to `false` by default and changed to `true` when tracing has left the
    /// interpreter frame and we need to ignore any remaining blocks.
    finish_early: bool,
    /// Info regarding the most recently seen recursive call to the interpreter.
    last_interp_call: Option<(BBlockId, &'static DeoptSafepoint)>,
}

impl TraceBuilder {
    /// Create a trace builder.
    ///
    /// Arguments:
    ///  - `aot_mod`: The AOT IR module that the trace flows through.
    ///  - `mtrace`: The mapped trace.
    ///  - `promotions`: Values promoted to constants during runtime.
    ///  - `debug_archors`: Debug strs recorded during runtime.
    fn new(
        _mt: &Arc<MT>,
        tracekind: TraceKind,
        aot_mod: &'static Module,
        ctrid: TraceId,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        endframe: TraceEndFrame,
    ) -> Result<Self, CompilationError> {
        Ok(Self {
            aot_mod,
            jit_mod: jit_ir::Module::new(tracekind, ctrid, aot_mod.global_decls_len())?,
            local_map: HashMap::new(),
            cp_block: None,
            // We have to insert a placeholder frame to represent the place we started tracing, as
            // we don't know where that is yet. We'll update it as soon as we do.
            frames: vec![InlinedFrame {
                funcidx: None,
                callinst: None,
                safepoint: None,
                args: Vec::new(),
            }],
            outline_info: None,
            recursion_count: 0,
            promotions,
            promote_idx: 0,
            debug_strs,
            debug_str_idx: 0,
            inferred_consts: HashMap::new(),
            endframe,
            finish_early: false,
            last_interp_call: None,
        })
    }

    // Given a mapped block, find the AOT block ID, or return `None` if it is unmapped or it's IR
    // is unavailable (for example it was marked `yk_outline`).
    fn lookup_aot_block(&self, tb: &TraceAction) -> Option<aot_ir::BBlockId> {
        match tb {
            TraceAction::MappedAOTBBlock { funcidx, bbidx } => {
                if !self.aot_mod.func((*funcidx).into()).is_declaration() {
                    Some(aot_ir::BBlockId::new(
                        (*funcidx).into(),
                        aot_ir::BBlockIdx::new(*bbidx),
                    ))
                } else {
                    None
                }
            }
            TraceAction::UnmappableBBlock => None,
            TraceAction::Promotion => todo!(),
        }
    }

    /// Create the prolog of the trace.
    fn create_trace_header(
        &mut self,
        blk: &'static aot_ir::BBlock,
    ) -> Result<(), CompilationError> {
        // Find the control point call to retrieve the live variables from its safepoint.
        let safepoint = match self.jit_mod.tracekind() {
            TraceKind::HeaderOnly | TraceKind::HeaderAndBody | TraceKind::DifferentFrames => {
                let mut inst_iter = blk.insts.iter().enumerate().rev();
                let mut safepoint = None;
                for (_, inst) in inst_iter.by_ref() {
                    // Is it a call to the control point, then insert loads for the trace inputs. These
                    // directly reference registers or stack slots in the parent frame and thus don't
                    // necessarily result in machine code during codegen.
                    if inst.is_control_point(self.aot_mod) {
                        safepoint = Some(inst.safepoint().unwrap());
                        break;
                    }
                }
                // If we don't find a safepoint here something has gone wrong with the AOT IR.
                safepoint.unwrap().clone()
            }
            TraceKind::Sidetrace(_) => unreachable!(),
            TraceKind::Connector(ctr) => ctr.safepoint().as_ref().unwrap().clone(),
        };
        self.jit_mod.safepoint = Some(safepoint.clone());
        let (rec, _) = AOT_STACKMAPS
            .as_ref()
            .unwrap()
            .get(usize::try_from(safepoint.id).unwrap());

        debug_assert!(safepoint.lives.len() == rec.live_vals.len());
        for idx in 0..safepoint.lives.len() {
            let aot_op = &safepoint.lives[idx];
            let input_tyidx = self.handle_type(aot_op.type_(self.aot_mod))?;

            // Get the location for this input variable.
            let var = &rec.live_vals[idx];
            if var.len() > 1 {
                todo!("Deal with multi register locations");
            }

            // Rewrite registers to their spill locations. We need to do this as we no longer
            // push/pop registers around the control point to reduce its overhead. We know that
            // for every live variable in a caller-saved register there must exist a spill offset
            // in that location's extras.
            let loc = match &var[0] {
                yksmp::Location::Register(reg, size, v) => {
                    let mut newloc = None;
                    for offset in v {
                        if *offset < 0 {
                            newloc = Some(yksmp::Location::Indirect(6, i32::from(*offset), *size));
                            break;
                        }
                    }
                    if let Some(loc) = newloc {
                        loc
                    } else if CALLER_CLOBBER_REG.contains(reg) {
                        panic!("No spill offset for caller-saved register.")
                    } else {
                        var[0].clone()
                    }
                }
                _ => var[0].clone(),
            };

            let param_inst = jit_ir::ParamInst::new(ParamIdx::try_from(idx)?, input_tyidx).into();
            self.jit_mod.push(param_inst)?;
            self.jit_mod.push_param(loc);
            self.local_map.insert(
                aot_op.to_inst_id(),
                jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
            );
            self.jit_mod
                .push_body_start_var(jit_ir::Operand::Var(self.jit_mod.last_inst_idx()));
        }
        self.jit_mod.push(jit_ir::Inst::TraceHeaderStart)?;
        Ok(())
    }

    /// Process (skip) promotions and debug strings inside an otherwise outlined block.
    fn process_promotions_and_debug_strs_only(
        &mut self,
        bid: &aot_ir::BBlockId,
    ) -> Result<(), CompilationError> {
        let blk = self.aot_mod.bblock(bid);

        for inst in blk.insts.iter() {
            match inst {
                aot_ir::Inst::Promote { tyidx, .. } => {
                    self.promote_idx +=
                        usize::try_from(self.aot_mod.type_(*tyidx).bytew()).unwrap();
                }
                aot_ir::Inst::Call {
                    callee: callee_fidx,
                    ..
                } => {
                    let callee = self.aot_mod.func(*callee_fidx);
                    if callee.is_idempotent() {
                        let aot_ir::Ty::Func(fty) = self.aot_mod.type_(callee.tyidx()) else {
                            panic!()
                        };
                        self.promote_idx +=
                            usize::try_from(self.aot_mod.type_(fty.ret_ty()).bytew()).unwrap();
                    }
                }
                aot_ir::Inst::DebugStr { .. } => {
                    // Skip this debug string.
                    self.debug_str_idx += 1;
                }
                _ => (),
            }
        }
        Ok(())
    }

    /// Walk over a traced AOT block, translating the constituent instructions into the JIT module.
    ///
    /// When all is well:
    ///
    ///  - Returns `Ok(Some(new_prevbb))` if the block was terminated by a `ret` and tracebuilder's
    ///    `prevbid` needs to be updated as a result.
    ///
    ///  - Returns `Ok(None)` for all other block terminators.
    fn process_block(
        &mut self,
        bid: &aot_ir::BBlockId,
        prevbb: &Option<aot_ir::BBlockId>,
        nextbb: Option<aot_ir::BBlockId>,
    ) -> Result<Option<Option<BBlockId>>, CompilationError> {
        // We have already checked this (and aborted building the trace) at the time we encountered
        // an `Inst::Ret` that would make the frame stack empty.
        assert!(!self.frames.is_empty());

        // unwrap safe: can't trace a block not in the AOT module.
        let blk = self.aot_mod.bblock(bid);

        // Decide how to translate each AOT instruction.
        for (iidx, inst) in blk.insts.iter().enumerate() {
            match inst {
                aot_ir::Inst::Br { .. } => Ok(()),
                aot_ir::Inst::Load {
                    ptr,
                    tyidx,
                    volatile,
                } => self.handle_load(bid, iidx, ptr, tyidx, *volatile),
                // FIXME: ignore remaining instructions after a call.
                aot_ir::Inst::Call {
                    callee,
                    args,
                    safepoint,
                } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_call(inst, bid, iidx, callee, args, safepoint.as_ref(), nextinst)
                }
                aot_ir::Inst::IndirectCall {
                    ftyidx,
                    callop,
                    args,
                    safepoint,
                } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_indirectcall(
                        inst, bid, iidx, ftyidx, callop, args, nextinst, safepoint,
                    )
                }
                aot_ir::Inst::Store { tgt, val, volatile } => {
                    self.handle_store(bid, iidx, tgt, val, *volatile)
                }
                aot_ir::Inst::PtrAdd {
                    ptr,
                    const_off,
                    dyn_elem_counts,
                    dyn_elem_sizes,
                    ..
                } => {
                    self.handle_ptradd(bid, iidx, ptr, *const_off, dyn_elem_counts, dyn_elem_sizes)
                }
                aot_ir::Inst::BinaryOp { lhs, binop, rhs } => {
                    self.handle_binop(bid, iidx, *binop, lhs, rhs)
                }
                aot_ir::Inst::ICmp { lhs, pred, rhs, .. } => {
                    self.handle_icmp(bid, iidx, lhs, pred, rhs)
                }
                aot_ir::Inst::CondBr {
                    cond,
                    true_bb,
                    safepoint,
                    ..
                } => self.handle_condbr(bid, safepoint, nextbb.as_ref().unwrap(), cond, true_bb),
                aot_ir::Inst::Cast {
                    cast_kind,
                    val,
                    dest_tyidx,
                } => self.handle_cast(bid, iidx, cast_kind, val, dest_tyidx),
                aot_ir::Inst::Ret { val } => {
                    let retframe = self.handle_ret(bid, iidx, val)?;
                    // In the case of a `ret` terminator, we need to communicate back to the caller
                    // the block that should become the new `prevbid` going forward.
                    //
                    // This is nasty.
                    let ret_prevbb: Option<Option<BBlockId>> =
                        if let Some(ref ci) = retframe.callinst {
                            Some(Some(BBlockId::new(ci.funcidx(), ci.bbidx())))
                        } else {
                            Some(None)
                        };
                    // Early return OK because `ret` is a terminator and there can be no further
                    // instructions in the block.
                    return Ok(ret_prevbb);
                }
                aot_ir::Inst::Switch {
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                    safepoint,
                } => self.handle_switch(
                    bid,
                    iidx,
                    safepoint,
                    nextbb.as_ref().unwrap(),
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                ),
                aot_ir::Inst::Phi {
                    incoming_bbs,
                    incoming_vals,
                    ..
                } => {
                    assert_eq!(
                        prevbb.as_ref().map(|x| x.funcidx()),
                        Some(bid.funcidx()),
                        "{:?} {:?}",
                        self.jit_mod.ctrid(),
                        self.jit_mod.insts_len()
                    );
                    self.handle_phi(
                        bid,
                        iidx,
                        &prevbb.as_ref().unwrap().bbidx(),
                        incoming_bbs,
                        incoming_vals,
                    )
                }
                aot_ir::Inst::Select {
                    cond,
                    trueval,
                    falseval,
                } => self.handle_select(bid, iidx, cond, trueval, falseval),
                aot_ir::Inst::LoadArg { arg_idx, .. } => {
                    // Map passed in arguments to their respective LoadArg instructions.
                    let jitop = &self.frames.last().unwrap().args[*arg_idx];
                    let aot_iid = aot_ir::InstId::new(
                        bid.funcidx(),
                        bid.bbidx(),
                        aot_ir::BBlockInstIdx::new(iidx),
                    );
                    self.local_map.insert(aot_iid, jitop.unpack(&self.jit_mod));
                    Ok(())
                }
                aot_ir::Inst::FCmp { lhs, pred, rhs, .. } => {
                    self.handle_fcmp(bid, iidx, lhs, pred, rhs)
                }
                aot_ir::Inst::Promote {
                    val,
                    tyidx,
                    safepoint,
                } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_promote(bid, iidx, val, safepoint, tyidx, nextinst)
                }
                aot_ir::Inst::FNeg { val } => self.handle_fneg(bid, iidx, val),
                aot_ir::Inst::DebugStr { .. } => {
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_debug_str(bid, iidx, nextinst)
                }
                aot_ir::Inst::ExtractValue { op, tyidx, indices } => {
                    self.handle_extractvalue(bid, iidx, op, tyidx, indices)
                }
                _ => Err(CompilationError::General(format!(
                    "Unimplemented: {inst:?}"
                ))),
            }?;
        }
        Ok(None)
    }

    /// Link the AOT IR to the last instruction pushed into the JIT IR.
    ///
    /// This must be called after adding a JIT IR instruction which has a return value.
    fn link_iid_to_last_inst(&mut self, bid: &aot_ir::BBlockId, aot_inst_idx: usize) {
        let aot_iid = aot_ir::InstId::new(
            bid.funcidx(),
            bid.bbidx(),
            aot_ir::BBlockInstIdx::new(aot_inst_idx),
        );
        // The unwrap is safe because we've already inserted an element at this index and proven
        // that the index is in bounds.
        self.local_map
            .insert(aot_iid, jit_ir::Operand::Var(self.jit_mod.last_inst_idx()));
    }

    fn copy_inst(
        &mut self,
        jit_inst: jit_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        if jit_inst.def_type(&self.jit_mod).is_some() {
            // If the AOT instruction defines a new value, then add it to the local map.
            self.jit_mod.push(jit_inst)?;
            self.link_iid_to_last_inst(bid, aot_inst_idx);
        } else {
            self.jit_mod.push(jit_inst)?;
        }
        Ok(())
    }

    /// Translate a global variable use.
    fn handle_global(
        &mut self,
        idx: aot_ir::GlobalDeclIdx,
    ) -> Result<jit_ir::GlobalDeclIdx, CompilationError> {
        let aot_global = self.aot_mod.global_decl(idx);
        let jit_global = jit_ir::GlobalDecl::new(
            CString::new(aot_global.name()).unwrap(),
            aot_global.is_threadlocal(),
            idx,
        );
        self.jit_mod.insert_global_decl(jit_global)
    }

    /// Translate a constant value.
    fn handle_const(
        &mut self,
        aot_const: &aot_ir::Const,
    ) -> Result<jit_ir::Const, CompilationError> {
        let aot_const = aot_const.unwrap_val();
        let bytes = aot_const.bytes();
        match self.aot_mod.type_(aot_const.tyidx()) {
            aot_ir::Ty::Integer(x) => {
                // FIXME: It would be better if the AOT IR had converted these integers in advance
                // rather than doing this dance here.
                let v = match x.bitw() {
                    1 | 8 => {
                        debug_assert_eq!(bytes.len(), 1);
                        u64::from(bytes[0])
                    }
                    16 => {
                        debug_assert_eq!(bytes.len(), 2);
                        u64::from(u16::from_ne_bytes([bytes[0], bytes[1]]))
                    }
                    32 => {
                        debug_assert_eq!(bytes.len(), 4);
                        u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    64 => {
                        debug_assert_eq!(bytes.len(), 8);
                        u64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    _ => todo!("{}", x.bitw()),
                };
                let jit_tyidx = self.jit_mod.insert_ty(jit_ir::Ty::Integer(x.bitw()))?;
                Ok(jit_ir::Const::Int(
                    jit_tyidx,
                    ArbBitInt::from_u64(x.bitw(), v),
                ))
            }
            aot_ir::Ty::Float(fty) => {
                let jit_tyidx = self.jit_mod.insert_ty(jit_ir::Ty::Float(fty.clone()))?;
                // unwrap cannot fail if the AOT IR is valid.
                let val = f64::from_ne_bytes(bytes[0..8].try_into().unwrap());
                Ok(jit_ir::Const::Float(jit_tyidx, val))
            }
            aot_ir::Ty::Ptr => {
                let val: usize;
                #[cfg(target_arch = "x86_64")]
                {
                    debug_assert_eq!(bytes.len(), 8);
                    val = usize::from_ne_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]);
                }
                Ok(jit_ir::Const::Ptr(val))
            }
            x => todo!("{x:?}"),
        }
    }

    /// Translate an operand.
    fn handle_operand(
        &mut self,
        op: &aot_ir::Operand,
    ) -> Result<jit_ir::Operand, CompilationError> {
        match op {
            aot_ir::Operand::Local(iid) => Ok(self.local_map[iid].decopy(&self.jit_mod).clone()),
            aot_ir::Operand::Const(cidx) => {
                let jit_const = self.handle_const(self.aot_mod.const_(*cidx))?;
                Ok(jit_ir::Operand::Const(
                    self.jit_mod.insert_const(jit_const)?,
                ))
            }
            aot_ir::Operand::Global(gd_idx) => {
                let load = jit_ir::LookupGlobalInst::new(self.handle_global(*gd_idx)?)?;
                self.jit_mod.push_and_make_operand(load.into())
            }
            aot_ir::Operand::Func(fidx) => {
                // We reduce a function operand to a constant pointer.
                let aot_func = self.aot_mod.func(*fidx);
                let fname = aot_func.name();
                let vaddr = symbol_to_ptr(fname).unwrap();
                let cidx = self
                    .jit_mod
                    .insert_const(jit_ir::Const::Ptr(vaddr as usize))
                    .unwrap();
                Ok(jit_ir::Operand::Const(cidx))
            }
        }
    }

    /// Translate a type.
    fn handle_type(&mut self, aot_type: &aot_ir::Ty) -> Result<jit_ir::TyIdx, CompilationError> {
        let jit_ty = match aot_type {
            aot_ir::Ty::Void => jit_ir::Ty::Void,
            aot_ir::Ty::Integer(x) => jit_ir::Ty::Integer(x.bitw()),
            aot_ir::Ty::Ptr => jit_ir::Ty::Ptr,
            aot_ir::Ty::Func(ft) => {
                let mut jit_args = Vec::new();
                for aot_arg_tyidx in ft.arg_tyidxs() {
                    let jit_ty = self.handle_type(self.aot_mod.type_(*aot_arg_tyidx))?;
                    jit_args.push(jit_ty);
                }
                let jit_retty = self.handle_type(self.aot_mod.type_(ft.ret_ty()))?;
                jit_ir::Ty::Func(jit_ir::FuncTy::new(jit_args, jit_retty, ft.is_vararg()))
            }
            aot_ir::Ty::Struct(st) => {
                let mut fields = Vec::new();
                for aotty in st.field_tyidxs() {
                    let jit_ty = self.handle_type(self.aot_mod.type_(*aotty))?;
                    fields.push(jit_ty);
                }
                jit_ir::Ty::Struct(jit_ir::StructTy::new(
                    st.bit_size(),
                    fields,
                    st.field_bit_offs().clone(),
                ))
            }
            aot_ir::Ty::Float(ft) => {
                let inner = match ft {
                    aot_ir::FloatTy::Float => jit_ir::FloatTy::Float,
                    aot_ir::FloatTy::Double => jit_ir::FloatTy::Double,
                };
                jit_ir::Ty::Float(inner)
            }
            aot_ir::Ty::Unimplemented(s) => jit_ir::Ty::Unimplemented(s.to_owned()),
        };
        self.jit_mod.insert_ty(jit_ty)
    }

    /// Translate a function.
    fn handle_func(
        &mut self,
        aot_idx: aot_ir::FuncIdx,
    ) -> Result<jit_ir::FuncDeclIdx, CompilationError> {
        let aot_func = self.aot_mod.func(aot_idx);
        let jit_func = jit_ir::FuncDecl::new(
            aot_func.name().to_owned(),
            self.handle_type(self.aot_mod.type_(aot_func.tyidx()))?,
        );
        self.jit_mod.insert_func_decl(jit_func)
    }

    /// Translate binary operations such as add, sub, mul, etc.
    fn handle_binop(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        binop: aot_ir::BinOp,
        lhs: &aot_ir::Operand,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let lhs = self.handle_operand(lhs)?;
        let rhs = self.handle_operand(rhs)?;
        let inst = jit_ir::BinOpInst::new(lhs, binop, rhs).into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    /// Create a guard.
    ///
    /// The guard fails if `cond` is not `expect`.
    fn create_guard(
        &mut self,
        bid: &aot_ir::BBlockId,
        cond: Option<&jit_ir::Operand>,
        expect: bool,
        safepoint: &'static aot_ir::DeoptSafepoint,
    ) -> Result<jit_ir::Inst, CompilationError> {
        // Assign this branch's stackmap to the current frame.
        self.frames.last_mut().unwrap().safepoint = Some(safepoint);

        // Collect the safepoint IDs and live variables from this conditional branch and the
        // previous frames to store inside the guard.
        // Unwrap-safe as each frame at this point must have a safepoint associated with it.
        let mut live_vars = Vec::new(); // (AOT var, JIT var) pairs
        let mut callframes = Vec::new();
        for frame in &self.frames {
            let safepoint = frame.safepoint.unwrap();
            // All the `unwrap`s are safe as we've filled in the necessary information during trace
            // building.
            callframes.push(jit_ir::InlinedFrame::new(
                frame.callinst.clone(),
                frame.funcidx.unwrap(),
                frame.safepoint.unwrap(),
            ));

            // Collect live variables.
            for op in safepoint.lives.iter() {
                match op {
                    aot_ir::Operand::Local(iid) => {
                        let inst = self.aot_mod.inst(iid);
                        if let Some(aot_ir::Ty::Struct(_)) = inst.def_type(self.aot_mod) {
                            panic!(
                                "can't deopt struct yet: {}",
                                inst.display(self.aot_mod, None)
                            );
                        }
                        match self.local_map[iid] {
                            jit_ir::Operand::Var(liidx) => {
                                // If, as often happens, a guard has in its live set the boolean
                                // variable used as the condition, we can convert this into a
                                // constant. For example if we have e.g.:
                                //
                                // ```
                                // %10: i1 = ...
                                // guard true, %10 [...: %10]
                                // ```
                                //
                                // then if the guard fails we know %10 was false and the guard is
                                // therefore equivalent to:
                                //
                                // ```
                                // %10: i1 = ...
                                // guard true, %10 [...: 0i1]
                                // ```
                                //
                                // Rewriting it in this form makes code further down the chain
                                // simpler, because it means it doesn't have to be clever when
                                // analysing a guard's live variables.
                                match cond {
                                    Some(Operand::Var(cond_idx)) if *cond_idx == liidx => {
                                        let cidx = if expect {
                                            self.jit_mod.false_constidx()
                                        } else {
                                            self.jit_mod.true_constidx()
                                        };
                                        live_vars.push((
                                            iid.clone(),
                                            PackedOperand::new(&jit_ir::Operand::Const(cidx)),
                                        ));
                                    }
                                    _ => {
                                        live_vars.push((
                                            iid.clone(),
                                            PackedOperand::new(&jit_ir::Operand::Var(liidx)),
                                        ));
                                    }
                                }
                            }
                            jit_ir::Operand::Const(cidx) => {
                                live_vars.push((
                                    iid.clone(),
                                    PackedOperand::new(&jit_ir::Operand::Const(cidx)),
                                ));
                            }
                        }
                    }
                    _ => panic!(), // IR malformed.
                }
            }
        }

        let gi = jit_ir::GuardInfo::new(*bid, live_vars, callframes, safepoint.id);
        let gi_idx = self.jit_mod.push_guardinfo(gi)?;

        if cond.is_none() {
            // This is a deopt instruction which always fails and thus doesn't have a condition.
            return Ok(jit_ir::Inst::Deopt(gi_idx));
        }

        // Can we infer a constant from this?
        //
        // If this is a `guard true` and the condition is a `eq` with a constant, then we have
        // inferred a constant *after* the guard.
        if expect {
            match cond.unwrap() {
                jit_ir::Operand::Var(iidx) => {
                    // Using `inst_nocopy()` here because `Inst::Const` can arise.
                    if let jit_ir::Inst::ICmp(icmp) = self.jit_mod.inst_nocopy(*iidx).unwrap()
                        && let jit_ir::Operand::Const(const_rhs) = icmp.rhs(&self.jit_mod)
                        && let jit_ir::Operand::Var(var_lhs) = icmp.lhs(&self.jit_mod)
                        && icmp.predicate() == jit_ir::Predicate::Equal
                    {
                        // Check we store a canonical (de-copied) instruction index.
                        assert!(self.jit_mod.inst_nocopy(var_lhs).is_some());
                        self.inferred_consts.insert(var_lhs, const_rhs);
                    }
                }
                jit_ir::Operand::Const(_) => (),
            };
        }

        Ok(jit_ir::GuardInst::new(cond.unwrap().clone(), expect, gi_idx).into())
    }

    /// Translate a conditional `Br` instruction.
    fn handle_condbr(
        &mut self,
        bid: &aot_ir::BBlockId,
        safepoint: &'static aot_ir::DeoptSafepoint,
        next_bb: &aot_ir::BBlockId,
        cond: &aot_ir::Operand,
        true_bb: &aot_ir::BBlockIdx,
    ) -> Result<(), CompilationError> {
        let jit_cond = self.handle_operand(cond)?;
        let guard =
            self.create_guard(bid, Some(&jit_cond), *true_bb == next_bb.bbidx(), safepoint)?;
        self.jit_mod.push(guard)?;
        Ok(())
    }

    /// If all is well, returns `Some(frame)`, where `frame` is the frame we just returned from.
    fn handle_ret(
        &mut self,
        _bid: &aot_ir::BBlockId,
        _aot_inst_idx: usize,
        val: &Option<aot_ir::Operand>,
    ) -> Result<InlinedFrame, CompilationError> {
        // If this `unwrap` fails, it means that early return detection in `mt.rs` is not working
        // as expected.
        let frame = self.frames.pop().unwrap();
        if !self.frames.is_empty() {
            if let Some(val) = val {
                let op = self.handle_operand(val)?;
                self.local_map
                    .insert(frame.callinst.as_ref().unwrap().clone(), op);
            }
            Ok(frame)
        } else {
            if let TraceKind::Sidetrace(_) = self.jit_mod.tracekind() {
                // Even though we currently catch side-traces that leave the main interpreter loop
                // in `mt.rs`, it can happen that a trace re-enters the main interpreter loop after
                // leaving it briefly without encountering a non-null location. As we only check
                // the frame addresses at non-null locations, this special case slips through and
                // we have to handle it here.
                // FIXME: Process side-traces that return from the starting frame in the same way
                // we process normal traces, by emitting a return instruction.
                return Err(CompilationError::General(
                    "Returning out of sidetrace currently unsupported.".into(),
                ));
            }
            // We've returned out of the function that started tracing. Stop processing any
            // remaining blocks and emit a return instruction that naturally returns from a
            // compiled trace into the interpreter.
            let safepoint = frame.safepoint.unwrap();
            // We currently don't support passing values back during early returns.
            assert!(val.is_none());
            self.jit_mod.push(jit_ir::Inst::Return(safepoint.id))?;
            self.finish_early = true;
            Ok(frame)
        }
    }

    /// Translate a `ICmp` instruction.
    fn handle_icmp(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        pred: &aot_ir::Predicate,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst =
            jit_ir::ICmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
                .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_fcmp(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        pred: &aot_ir::FloatPredicate,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst =
            jit_ir::FCmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
                .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    /// Translate a `Load` instruction.
    fn handle_load(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        tyidx: &aot_ir::TyIdx,
        volatile: bool,
    ) -> Result<(), CompilationError> {
        if let aot_ir::Ty::Struct(_) = self.aot_mod.type_(*tyidx) {
            // Our register allocator can't handle multiple registers for a single instruction.
            // Luckily, we can rewrite struct loads and the `extractvalue` instructions that follow
            // it into something simpler. For this we first ignore this load and replace it with
            // its operand in the local map. When handling the `extractvalue` instruction we
            // replace it with a ptradd and a load. For example, this
            // ```
            // %0: {i8, i64} = load %ptr
            // %1: i8 = extractvalue %0, 0
            // %2: i64 = extractvalue %0, 1
            // ```
            // becomes
            // ```
            // %1: ptr = ptr_add %ptr, 0
            // %2: i8 = load %1
            // %3: ptr = ptr_add %ptr, 8
            // %4: i64 = load %3
            // ```
            // This also works when the load was inlined from another call, as the `ret` naturally
            // forwards the loads operand from the local map.
            let aot_iit = aot_ir::InstId::new(
                bid.funcidx(),
                bid.bbidx(),
                aot_ir::BBlockInstIdx::new(aot_inst_idx),
            );
            let jitop = self.handle_operand(ptr)?;
            self.local_map.insert(aot_iit, jitop);
            return Ok(());
        }
        let inst = jit_ir::LoadInst::new(
            self.handle_operand(ptr)?,
            self.handle_type(self.aot_mod.type_(*tyidx))?,
            volatile,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_indirectcall(
        &mut self,
        inst: &'static aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ftyidx: &aot_ir::TyIdx,
        callop: &aot_ir::Operand,
        args: &[aot_ir::Operand],
        nextinst: &'static aot_ir::Inst,
        safepoint: &'static aot_ir::DeoptSafepoint,
    ) -> Result<(), CompilationError> {
        debug_assert!(!inst.is_debug_call(self.aot_mod));

        let jit_callop = self.handle_operand(callop)?;
        if let jit_ir::Operand::Var(callee_iidx) = jit_callop
            && let Some(cidx) = self.inferred_consts.get(&callee_iidx)
        {
            // The callee is constant. We can treat this *indirect* call as if it were a
            // *direct* call, and maybe even inline it.
            let Const::Ptr(vaddr) = self.jit_mod.const_(*cidx) else {
                panic!();
            };
            // Find the function the constant pointer is referring to.
            let dli = ykaddr::addr::dladdr(*vaddr).unwrap();
            assert_eq!(dli.dli_saddr(), *vaddr);
            let callee = self.aot_mod.funcidx(dli.dli_sname().unwrap());
            if self.aot_mod.func(callee).is_idempotent() {
                // ykllvm doesn't insert idempotent recorder calls for indirect calls, so if we
                // allow this to proceed, it's not going to do the right thing.
                todo!();
            }
            return self.direct_call_impl(
                bid,
                aot_inst_idx,
                &callee,
                args,
                Some(safepoint),
                nextinst,
            );
        }

        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            jit_args.push(self.handle_operand(arg)?);
        }

        // While we can work out if this is a mappable call or not by loooking at the next block,
        // this unfortunately doesn't tell us whether the call has been annotated with
        // "yk_outline". For now we have to be conservative here and outline all indirect calls.
        // One solution would be to stop including the AOT IR for functions annotated with
        // "yk_outline". Any mappable, indirect call is then guaranteed to be inline safe.
        self.outline_until_after_call(bid, nextinst);

        let jit_tyidx = self.handle_type(self.aot_mod.type_(*ftyidx))?;
        let inst =
            jit_ir::IndirectCallInst::new(&mut self.jit_mod, jit_tyidx, jit_callop, jit_args)?;
        let idx = self.jit_mod.push_indirect_call(inst)?;
        self.copy_inst(jit_ir::Inst::IndirectCall(idx), bid, aot_inst_idx)
    }

    /// Handle a *direct* call instruction.
    fn handle_call(
        &mut self,
        inst: &'static aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        callee: &aot_ir::FuncIdx,
        args: &[aot_ir::Operand],
        safepoint: Option<&'static aot_ir::DeoptSafepoint>,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        // Ignore special functions that we neither want to inline nor copy.
        if inst.is_debug_call(self.aot_mod) {
            return Ok(());
        }
        // Ignore software tracer calls. Software tracer inserts
        // `__yk_trace_basicblock` instruction calls into the beginning of
        // every basic block. These calls can be ignored as they are
        // only used to collect runtime information for the tracer itself.
        if AOT_MOD.func(*callee).name() == "__yk_trace_basicblock" {
            return Ok(());
        }

        if inst.is_control_point(self.aot_mod) {
            return Ok(());
        }

        self.direct_call_impl(bid, aot_inst_idx, callee, args, safepoint, nextinst)
    }

    fn direct_call_impl(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        callee: &aot_ir::FuncIdx,
        args: &[aot_ir::Operand],
        safepoint: Option<&'static aot_ir::DeoptSafepoint>,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            match self.handle_operand(arg)? {
                jit_ir::Operand::Const(c) => {
                    // We don't want to do constant propagation here as it makes our life harder
                    // creating guards. Instead we simply create a `Const` instruction here and
                    // reference that.
                    let inst = jit_ir::Inst::Const(c);
                    self.jit_mod.push(inst)?;
                    let op = jit_ir::Operand::Var(self.jit_mod.last_inst_idx());
                    jit_args.push(op);
                }
                op => jit_args.push(op),
            }
        }

        // Check if this is a recursive call by scanning the call stack for the callee.
        let is_recursive = self.frames.iter().any(|f| f.funcidx == Some(*callee));
        let func = self.aot_mod.func(*callee);
        if !func.is_declaration()
            && !func.is_outline()
            && !func.is_idempotent()
            && !func.contains_call_to(self.aot_mod, "llvm.va_start")
            && !is_recursive
        {
            // This is a mappable call that we want to inline.
            debug_assert!(safepoint.is_some());
            // Assign safepoint to the current frame.
            // Unwrap is safe as there's always at least one frame.
            self.frames.last_mut().unwrap().safepoint = safepoint;
            // Create a new frame for the inlined call and pass in the arguments of the caller.
            let aot_iid = aot_ir::InstId::new(
                bid.funcidx(),
                bid.bbidx(),
                aot_ir::BBlockInstIdx::new(aot_inst_idx),
            );
            self.frames.push(InlinedFrame {
                funcidx: Some(*callee),
                callinst: Some(aot_iid),
                safepoint: None,
                args: jit_args.iter().map(PackedOperand::new).collect::<Vec<_>>(),
            });
            Ok(())
        } else {
            // This call can't be inlined. It is either unmappable (a declaration or an indirect
            // call) or the compiler annotated it with `yk_outline`.
            self.outline_until_after_call(bid, nextinst);
            let jit_func_decl_idx = self.handle_func(*callee)?;

            // If the callee is marked idempotent, retrieve the dynamically captured runtime value
            // and stash it inside the JIT IR call instruction. Later if the optimiser can prove
            // the arguments to the call are constant, the call can be replaced with this value.
            //
            // Note that functions marked `yk_idempotent` must not contain (or call functions that
            // contain) promotions. Similarly, `yk_idempotent` functions must not directly or
            // transitively call other functions marked `yk_idempotent`. It is UB to do either of
            // these things (because the promote buffer would be consumed in the wrong order).
            let idem_const = if func.is_idempotent() {
                let func_tyidx = self.jit_mod.func_decl(jit_func_decl_idx).tyidx();
                let jit_ir::Ty::Func(jit_fty) = self.jit_mod.type_(func_tyidx) else {
                    panic!()
                };
                Some(self.promote_bytes_to_const(jit_fty.ret_tyidx())?)
            } else {
                None
            };

            let inst = jit_ir::DirectCallInst::new(
                &mut self.jit_mod,
                jit_func_decl_idx,
                jit_args,
                idem_const,
            )?
            .into();
            if self.frames.first().unwrap().funcidx == Some(*callee) {
                // Store the block id and safepoint for the most recently seen recursive
                // interpreter call.
                self.last_interp_call = Some((*bid, safepoint.unwrap()));
            }
            self.copy_inst(inst, bid, aot_inst_idx)
        }
    }

    fn handle_store(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        tgt: &aot_ir::Operand,
        val: &aot_ir::Operand,
        volatile: bool,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::StoreInst::new(
            self.handle_operand(tgt)?,
            self.handle_operand(val)?,
            volatile,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_ptradd(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        const_off: isize,
        dyn_elem_counts: &[aot_ir::Operand],
        dyn_elem_sizes: &[usize],
    ) -> Result<(), CompilationError> {
        // First apply the constant offset.
        let mut jit_ptr = self.handle_operand(ptr)?;
        if const_off != 0 {
            let off_i32 = i32::try_from(const_off).map_err(|_| {
                CompilationError::LimitExceeded("ptradd constant offset doesn't fit in i32".into())
            })?;
            jit_ptr = self
                .jit_mod
                .push_and_make_operand(jit_ir::PtrAddInst::new(jit_ptr, off_i32).into())?;
        } else {
            jit_ptr = match jit_ptr {
                Operand::Var(iidx) => self
                    .jit_mod
                    .push_and_make_operand(jit_ir::Inst::Copy(iidx))?,
                _ => todo!(),
            }
        }

        // Now apply any dynamic indices.
        //
        // Each offset is the number of elements multiplied by the byte size of an element.
        for (num_elems, elem_size) in dyn_elem_counts.iter().zip(dyn_elem_sizes) {
            let num_elems = self.handle_operand(num_elems)?;
            // If the element count is not the same width as LLVM's GEP index type, then we have to
            // sign extend it up (or truncate it down) to the right size. To date I've been unable
            // to get clang to emit code that would require an extend or truncate, so for now it's
            // a todo.
            if num_elems.byte_size(&self.jit_mod) * 8 != usize::from(self.aot_mod.ptr_off_bitsize())
            {
                todo!();
            }
            let elem_size = u16::try_from(*elem_size).map_err(|_| {
                CompilationError::LimitExceeded("ptradd elem size doesn't fit in u16".into())
            })?;
            jit_ptr = self.jit_mod.push_and_make_operand(
                jit_ir::DynPtrAddInst::new(jit_ptr, num_elems, elem_size).into(),
            )?;
        }
        self.link_iid_to_last_inst(bid, aot_inst_idx);
        Ok(())
    }

    fn handle_cast(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        cast_kind: &aot_ir::CastKind,
        val: &aot_ir::Operand,
        dest_tyidx: &aot_ir::TyIdx,
    ) -> Result<(), CompilationError> {
        let inst = match cast_kind {
            aot_ir::CastKind::SExt => jit_ir::SExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::ZeroExtend => jit_ir::ZExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::Trunc => jit_ir::TruncInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::SIToFP => jit_ir::SIToFPInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::FPExt => jit_ir::FPExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::FPToSI => jit_ir::FPToSIInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::BitCast => jit_ir::BitCastInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::PtrToInt => jit_ir::PtrToIntInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::IntToPtr => {
                jit_ir::IntToPtrInst::new(&self.handle_operand(val)?).into()
            }
            aot_ir::CastKind::UIToFP => jit_ir::UIToFPInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
        };
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_switch(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        safepoint: &'static aot_ir::DeoptSafepoint,
        next_bb: &aot_ir::BBlockId,
        test_val: &aot_ir::Operand,
        default_dest: &aot_ir::BBlockIdx,
        case_values: &[u64],
        case_dests: &[aot_ir::BBlockIdx],
    ) -> Result<(), CompilationError> {
        if case_values.is_empty() {
            // Degenerate switch. Not sure it can even happen.
            panic!();
        }

        let jit_tyidx = self.handle_type(test_val.type_(self.aot_mod))?;
        let bitw = self.jit_mod.type_(jit_tyidx).bitw().unwrap();

        // Find out which cases we need to guard. This can be either all (if we hit a default
        // case), some (if multiple cases map to the same block), or one (if only one case maps to
        // the next block in the trace).
        let (expect, check_vals) = match case_dests.iter().position(|&cd| cd == next_bb.bbidx()) {
            Some(_) => {
                (
                    true,
                    // Multiple `case_values` might map to the same block, so we need to guard all
                    // of the values which point to that block.
                    case_dests
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| *x == &next_bb.bbidx())
                        .map(|(i, _)| case_values[i])
                        .collect::<Vec<_>>(),
                )
            }
            None => {
                // If this assertion fails then the basic block that was executed next wasn't an
                // arm of the switch and something has gone wrong.
                assert_eq!(&next_bb.bbidx(), default_dest);
                // The default case was traced.
                //
                // We need a guard that expresses that `test_val` was not any of the `case_values`.
                // We encode this in the JIT IR like:
                //
                //     member = test_val == case_val1 || ... || test_val == case_valN
                //     guard member == 0
                //
                // OPT: This could be much more efficient if we introduced a special "none of these
                // values" guard. It could codegen the comparisons into a bitfield and efficiently
                // bitwise OR a whole word's worth of comparison outcomes at once.
                //
                // OPT: Also depending on the shape of the cases you may be able to optimise. e.g.
                // if they are a consecutive run, you could do a range check instead of all of
                // these comparisons.
                (false, Vec::from(case_values))
            }
        };

        let mut cmps_opnds = Vec::new();
        for cv in check_vals {
            // Build a constant of the case value.
            let jit_const = jit_ir::Const::Int(jit_tyidx, ArbBitInt::from_u64(bitw, cv));
            let jit_const_opnd = jit_ir::Operand::Const(self.jit_mod.insert_const(jit_const)?);

            // Do the comparison.
            let jit_test_val = self.handle_operand(test_val)?;
            let cmp = jit_ir::ICmpInst::new(jit_test_val, jit_ir::Predicate::Equal, jit_const_opnd);
            cmps_opnds.push(self.jit_mod.push_and_make_operand(cmp.into())?);
        }

        // OR together all the equality tests if there are more than one.
        let mut jit_cond = None;
        for cmp in cmps_opnds {
            if jit_cond.is_none() {
                jit_cond = Some(cmp);
            } else {
                // unwrap can't fail due to the above.
                let lhs = jit_cond.take().unwrap();
                let or = jit_ir::BinOpInst::new(lhs, BinOp::Or, cmp);
                jit_cond = Some(self.jit_mod.push_and_make_operand(or.into())?);
            }
        }

        // Guard the result of ORing all the comparisons together.
        // unwrap can't fail: we already disregarded degenerate switches with no
        // non-default cases.
        let guard = self.create_guard(bid, Some(&jit_cond.unwrap()), expect, safepoint)?;
        self.copy_inst(guard, bid, aot_inst_idx)
    }

    fn handle_phi(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        prev_bb: &aot_ir::BBlockIdx,
        incoming_bbs: &[aot_ir::BBlockIdx],
        incoming_vals: &[aot_ir::Operand],
    ) -> Result<(), CompilationError> {
        // If the IR is well-formed the indexing and unwrap() here will not fail.
        let chosen_val = &incoming_vals[incoming_bbs.iter().position(|bb| bb == prev_bb).unwrap()];
        let aot_iit = aot_ir::InstId::new(
            bid.funcidx(),
            bid.bbidx(),
            aot_ir::BBlockInstIdx::new(aot_inst_idx),
        );
        let op = match self.handle_operand(chosen_val)? {
            jit_ir::Operand::Const(c) => {
                // We don't want to do constant propagation here as it makes our life harder
                // creating guards. Instead we simply create a `Const` instruction here and
                // reference that.
                let inst = jit_ir::Inst::Const(c);
                self.jit_mod.push(inst)?;
                jit_ir::Operand::Var(self.jit_mod.last_inst_idx())
            }
            op => op,
        };
        self.local_map.insert(aot_iit, op);
        Ok(())
    }

    fn handle_select(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        cond: &aot_ir::Operand,
        trueval: &aot_ir::Operand,
        falseval: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::SelectInst::new(
            self.handle_operand(cond)?,
            self.handle_operand(trueval)?,
            self.handle_operand(falseval)?,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_extractvalue(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        op: &aot_ir::Operand,
        extracttyidx: &TyIdx,
        indices: &[usize],
    ) -> Result<(), CompilationError> {
        assert_eq!(indices.len(), 1);
        let jitop = self.handle_operand(op)?;
        match jitop {
            Operand::Var(jitiidx) => {
                if let jit_ir::Inst::Call(_) = self.jit_mod.inst(jitiidx) {
                    // We didn't manage to inline the call generating this struct, so we need to
                    // emit the `extractvalue` instruction as is and handle it in the codegen (most
                    // likely by processing the call and the (two) `extractvalue` instructions in
                    // one go).
                    let inst = jit_ir::ExtractValueInst::new(
                        self.handle_operand(op)?,
                        self.handle_type(self.aot_mod.type_(*extracttyidx))?,
                        indices.into(),
                    )
                    .into();
                    return self.copy_inst(inst, bid, aot_inst_idx);
                }
            }
            _ => panic!(),
        }

        // Try to rewrite the `extractvalue` instruction into an equivalent ptradd/load
        // instructions.
        if let aot_ir::Operand::Local(id) = op {
            match self.aot_mod.inst(id).def_type(self.aot_mod).unwrap() {
                aot_ir::Ty::Struct(st) => {
                    let off = st.field_bit_offs()[indices[0]] / 8;
                    let ptradd = jit_ir::PtrAddInst::new(jitop, i32::try_from(off).unwrap());
                    self.jit_mod.push(ptradd.into())?;
                    let load = jit_ir::LoadInst::new(
                        jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
                        self.handle_type(self.aot_mod.type_(*extracttyidx))?,
                        false,
                    );
                    self.jit_mod.push(load.into())?;
                    self.link_iid_to_last_inst(bid, aot_inst_idx);
                }
                _ => panic!(),
            }
        } else {
            panic!();
        }
        Ok(())
    }

    /// Turn outlining until the specified call has been consumed from the trace.
    ///
    /// `terminst` is the terminating instruction immediately after the call, guaranteed to be
    /// present due to ykllvm's block splitting pass.
    ///
    /// `bid` is the [BBlockId] containing the call.
    fn outline_until_after_call(&mut self, bid: &BBlockId, terminst: &'static aot_ir::Inst) {
        match terminst {
            aot_ir::Inst::Br { succ } => {
                // We can only stop outlining when we see the succesor block and we are not in
                // the middle of recursion.
                let succbid = BBlockId::new(bid.funcidx(), *succ);
                self.outline_info = Some((bid.to_owned(), succbid));
                self.recursion_count = 0;
            }
            aot_ir::Inst::CondBr { .. } => {
                // Currently, the successor of a call is always an unconditional branch due to
                // the block spitting pass. However, there's a FIXME in that pass which could
                // lead to conditional branches showing up here. Leave a todo here so we know
                // when this happens.
                todo!()
            }
            _ => panic!(),
        }
    }

    /// Consume bytes from `self.promotions` and build a constant from them of type `tyidx`.
    fn promote_bytes_to_const(
        &mut self,
        tyidx: jit_ir::TyIdx,
    ) -> Result<jit_ir::ConstIdx, CompilationError> {
        let ty = self.jit_mod.type_(tyidx);
        let c = match ty {
            jit_ir::Ty::Void => unreachable!(),
            jit_ir::Ty::Integer(bitw) => {
                let bytew = ty.byte_size().unwrap();
                let v = match *bitw {
                    64 => u64::from_ne_bytes(
                        self.promotions[self.promote_idx..self.promote_idx + bytew]
                            .try_into()
                            .unwrap(),
                    ),
                    32 => u64::from(u32::from_ne_bytes(
                        self.promotions[self.promote_idx..self.promote_idx + bytew]
                            .try_into()
                            .unwrap(),
                    )),
                    x => todo!("{x}"),
                };
                self.promote_idx += bytew;
                Const::Int(tyidx, ArbBitInt::from_u64(*bitw, v))
            }
            jit_ir::Ty::Ptr => {
                let bytew = ty.byte_size().unwrap();
                assert_eq!(bytew, std::mem::size_of::<usize>());
                let v = usize::from_ne_bytes(
                    self.promotions[self.promote_idx..self.promote_idx + bytew]
                        .try_into()
                        .unwrap(),
                );
                self.promote_idx += bytew;
                Const::Ptr(v)
            }
            jit_ir::Ty::Func(_) => todo!(),
            jit_ir::Ty::Float(_) => todo!(),
            jit_ir::Ty::Struct(_) => todo!(),
            jit_ir::Ty::Unimplemented(_) => todo!(),
        };
        self.jit_mod.insert_const(c)
    }

    fn handle_promote(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
        safepoint: &'static aot_ir::DeoptSafepoint,
        tyidx: &aot_ir::TyIdx,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        self.outline_until_after_call(bid, nextinst);
        match self.handle_operand(val)? {
            jit_ir::Operand::Var(ref_iidx) => {
                self.jit_mod.push(jit_ir::Inst::Copy(ref_iidx))?;
                self.link_iid_to_last_inst(bid, aot_inst_idx);

                // Insert a guard to ensure the trace only runs if the value we encounter is the
                // same each time.
                let tyidx = self.handle_type(self.aot_mod.type_(*tyidx))?;
                // Create the constant from the runtime value.
                let cidx = self.promote_bytes_to_const(tyidx)?;
                let cmp_instr = jit_ir::ICmpInst::new(
                    jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
                    aot_ir::Predicate::Equal,
                    jit_ir::Operand::Const(cidx),
                );
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_instr.into())?;
                let guard = self.create_guard(bid, Some(&jit_cond), true, safepoint)?;
                self.copy_inst(guard, bid, aot_inst_idx)
            }
            jit_ir::Operand::Const(_cidx) => todo!(),
        }
    }

    fn handle_fneg(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::FNegInst::new(self.handle_operand(val)?).into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_debug_str(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        self.outline_until_after_call(bid, nextinst);
        let msg = self.debug_strs[self.debug_str_idx].to_owned();
        let inst =
            jit_ir::Inst::DebugStr(jit_ir::DebugStrInst::new(self.jit_mod.push_debug_str(msg)?));
        self.copy_inst(inst, bid, aot_inst_idx)?;
        self.debug_str_idx += 1;
        Ok(())
    }

    /// Entry point for building an IR trace.
    ///
    /// Consumes the trace builder, returning a JIT module.
    ///
    /// # Panics
    ///
    /// If `ta_iter` produces no elements.
    fn build(
        mut self,
        mt: &Arc<MT>,
        ta_iter: Box<dyn AOTTraceIterator>,
    ) -> Result<jit_ir::Module, CompilationError> {
        // Collect the trace first so we can peek at the last element to find the control point
        // block during side-tracing.
        // FIXME: This is a workaround so we can peek at the last block in a trace in order to
        // extract information from the control point when we are side-tracing. Ideally, we extract
        // this information at AOT compile time and serialise it into the module (or block), so we
        // don't have to do this.
        mt.stats.timing_state(TimingState::TraceMapping);
        let tas = ta_iter
            .map(|x| x.map_err(|e| CompilationError::General(e.to_string())))
            .collect::<Result<Vec<_>, _>>()?;

        // Peek to the last block (needed for side-tracing).
        let lastblk = match &tas.last() {
            Some(b) => self.lookup_aot_block(b),
            _ => todo!(),
        };

        let mut trace_iter = tas.into_iter().peekable();

        mt.stats.timing_state(TimingState::Compiling);

        // The previously processed block.
        // `None` when either:
        //  - this is the first block in the trace.
        //  - the previous block was unmappable.
        let mut prev_bid = None;
        // The previously processed *mappable* block.
        // Only `None` if this is the first block in the trace
        let mut prev_mappable_bid: Option<BBlockId> = None;

        if let TraceKind::Sidetrace(sti) = self.jit_mod.tracekind() {
            let sti = Arc::clone(sti);
            // Set the previous block to the last block in the parent trace. Required in order to
            // process phi nodes.
            prev_bid = Some(sti.bid);

            // Setup the trace builder for side-tracing.
            // Create loads for the live variables that will be passed into the side-trace from the
            // parent trace.
            for (idx, (aotid, loc)) in sti.lives().iter().enumerate() {
                let aotinst = self.aot_mod.inst(aotid);
                let aotty = aotinst.def_type(self.aot_mod).unwrap();
                let tyidx = self.handle_type(aotty)?;
                let param_inst = jit_ir::ParamInst::new(ParamIdx::try_from(idx)?, tyidx).into();
                self.jit_mod.push(param_inst)?;
                self.jit_mod.push_param(loc.clone());
                self.local_map.insert(
                    aotid.clone(),
                    jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
                );
            }
            self.cp_block = lastblk;
            self.frames = sti
                .callframes()
                .iter()
                .map(|x| InlinedFrame {
                    funcidx: Some(x.funcidx),
                    callinst: x.callinst.clone(),
                    safepoint: Some(x.safepoint),
                    // We don't need to copy the arguments since they are only required for LoadArg
                    // instructions which we won't see at the beginning of a sidetrace.
                    args: vec![],
                })
                .collect::<Vec<_>>();

            // When side-tracing a switch guard failure, we need to reprocess the switch statement
            // (and only the switch statement) in order to emit a guard at the beginning of the
            // side-trace to check that the case requiring execution is that case the trace
            // captures.
            //
            // Note that it is not necessary to emit such a guard into side-traces stemming from
            // regular conditionals, since a conditional has only two successors. The parent trace
            // captures one, so by construction the side trace must capture the other.
            let prevbb = self.aot_mod.bblock(prev_bid.as_ref().unwrap());
            if let aot_ir::Inst::Switch {
                test_val,
                default_dest,
                case_values,
                case_dests,
                safepoint,
            } = &prevbb.insts.last().unwrap()
            {
                // Skip any residual bits of the block containing the switch that *could* appear at
                // the start of the trace. It appears that this can happen when the switch dispatch
                // is codegenned to multiple blocks (e.g. cascading conditionals).
                while &self.lookup_aot_block(trace_iter.peek().unwrap()).unwrap()
                    == prev_bid.as_ref().unwrap()
                {
                    let _ = trace_iter.next().unwrap(); // skip that block.
                }
                let nextbb = self.lookup_aot_block(trace_iter.peek().unwrap()).unwrap();
                self.handle_switch(
                    prev_bid.as_ref().unwrap(), // this is safe, we've just created this above
                    prevbb.insts.len() - 1,
                    safepoint,
                    &nextbb,
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                )?;
            }
        }
        // The variable `prev_bid` contains the block of the guard that initiated side-tracing (for
        // normal traces this is set to `None`). When hardware tracing, we capture this block again
        // as part of the side-trace. However, since we've already processed this block in the
        // parent trace, we must not process it again in the side-trace.
        //
        // Typically, the mapper would strip this block for us, but for codegen related reasons,
        // e.g. a switch statement codegenning to many machine blocks, it's possible for multiple
        // duplicates of this same block to show up here, which all need to be skipped.
        if prev_bid.is_some() {
            while self.lookup_aot_block(trace_iter.peek().unwrap()) == prev_bid {
                trace_iter.next().unwrap();
            }
        }

        match self.jit_mod.tracekind() {
            TraceKind::HeaderOnly
            | TraceKind::HeaderAndBody
            | TraceKind::Connector(_)
            | TraceKind::DifferentFrames => {
                // Find the block containing the control point call. This is the (sole) predecessor of the
                // first (guaranteed mappable) block in the trace. Note that empty traces are handled in
                // the tracing phase so the `unwrap` is safe.
                let prev = match trace_iter.peek().unwrap() {
                    TraceAction::MappedAOTBBlock {
                        funcidx: func_idx,
                        bbidx,
                    } => {
                        debug_assert!(*bbidx > 0);
                        // It's `- 1` due to the way the ykllvm block splitting pass works.
                        TraceAction::MappedAOTBBlock {
                            funcidx: *func_idx,
                            bbidx: bbidx - 1,
                        }
                    }
                    TraceAction::UnmappableBBlock => panic!(),
                    TraceAction::Promotion => todo!(),
                };

                self.cp_block = self.lookup_aot_block(&prev);
                self.frames.last_mut().unwrap().funcidx =
                    Some(self.cp_block.as_ref().unwrap().funcidx());
                // This unwrap can't fail. If it does that means the tracer has given us a mappable block
                // that doesn't exist in the AOT module.
                self.create_trace_header(self.aot_mod.bblock(self.cp_block.as_ref().unwrap()))?;
            }
            TraceKind::Sidetrace(_) => (),
        }

        // FIXME: this section of code needs to be refactored.
        #[cfg(tracer_hwt)]
        let mut last_blk_is_return = false;
        while let Some(b) = trace_iter.next() {
            if self.finish_early {
                // We don't need to process any remaining blocks. This is because we are finishing
                // the trace early due to interpreter recursion.
                break;
            }
            match self.lookup_aot_block(&b) {
                Some(bid) => {
                    // MappedAOTBBlock block
                    if let Some(prev_mbid) = &prev_mappable_bid {
                        // Due to the way HWT works, when you return from a call, you see the same
                        // basic block again. We skip it.
                        // FIXME: trace builder should be tracer agnostic.
                        #[cfg(tracer_hwt)]
                        if *prev_mbid == bid {
                            continue;
                        }
                        // Check the control flow is regular, bailing out if we detect e.g.
                        // longjmp().
                        //
                        // FIXME: we are unable to detect signal handlers.
                        // See tests/c/signal_handler_interrupts_trace.c
                        if !bid.static_intraprocedural_successor_of(prev_mbid, self.aot_mod)
                            && !bid.is_entry()
                            && !self.aot_mod.bblock(prev_mbid).is_return()
                        {
                            return Err(CompilationError::General(
                                "irregular control flow detected (unexpected successor)".into(),
                            ));
                        }
                    }
                    if let Some((ref outbid, ref tgtbid)) = self.outline_info {
                        // We are currently outlining.
                        if outbid == &bid {
                            // a longjmp (or similar) may have occurred.
                            return Err(CompilationError::General(
                                "irregular control flow detected (outline recursion)".into(),
                            ));
                        }
                        if bid.funcidx() == tgtbid.funcidx() {
                            // We are inside the same function that started outlining.
                            if bid.is_entry() {
                                // We are recursing into the function that started
                                // outlining.
                                self.recursion_count += 1;
                            }
                            if self.aot_mod.bblock(&bid).is_return() {
                                // We are returning from the function that started
                                // outlining. This may be one of multiple inlined calls, so
                                // we may not be done outlining just yet.
                                self.recursion_count = self.recursion_count.checked_sub(1).unwrap();
                            }
                        }
                        if self.recursion_count == 0 && bid == *tgtbid {
                            // We've reached the successor block of the function/block that
                            // started outlining. We are done and can continue processing
                            // blocks normally.
                            self.outline_info = None;
                            // We've returned from the recursive interpreter call so this info is
                            // no longer needed.
                            self.last_interp_call = None;
                        } else {
                            // We are outlining so just skip this block. However, we still need to
                            // process promoted values to make sure we've processed all promotion
                            // data and haven't messed up the mapping.
                            self.process_promotions_and_debug_strs_only(&bid)?;
                            prev_bid = Some(bid);
                            prev_mappable_bid = Some(bid);
                            continue;
                        }
                    } else {
                        // We are not outlining. Process blocks normally.
                        #[cfg(tracer_hwt)]
                        if last_blk_is_return {
                            // If the last block had a return terminator, we are returning
                            // from a call, which means the HWT will have recorded an
                            // additional caller block that we'll have to skip.
                            // FIXME: This only applies to the HWT as the SWT doesn't
                            // record these extra blocks.
                            last_blk_is_return = false;
                            prev_bid = Some(bid.clone());
                            prev_mappable_bid = Some(bid);
                            continue;
                        }
                        #[cfg(tracer_hwt)]
                        if self.aot_mod.bblock(&bid).is_return() {
                            last_blk_is_return = true;
                        }
                    }

                    // In order to emit guards for conditional branches we need to peek at the next
                    // block.
                    let nextbb = trace_iter.peek().and_then(|x| self.lookup_aot_block(x));
                    let ret_prevbb = self.process_block(&bid, &prev_bid, nextbb)?;
                    if self.cp_block.as_ref() == Some(&bid) {
                        // When using the hardware tracer we will see two control point
                        // blocks here. We must only process one of them. The simplest way
                        // to do this that is compatible with the software tracer is to
                        // start outlining here by setting the outlining stop target to
                        // this blocks successor.
                        let blk = self.aot_mod.bblock(&bid);
                        let br = blk.insts.iter().last();
                        // Unwrap safe: block guaranteed to have instructions.
                        match br.unwrap() {
                            aot_ir::Inst::Br { succ } => {
                                let succbid = BBlockId::new(bid.funcidx(), *succ);
                                self.outline_info = Some((bid.to_owned(), succbid));
                                self.recursion_count = 0;
                            }
                            _ => panic!(),
                        }
                    }
                    // If the block was terminated with a `ret` we have to update `prev_bid`.
                    if let Some(ret_prevbb) = ret_prevbb {
                        prev_bid = ret_prevbb;
                    } else {
                        prev_bid = Some(bid);
                    }
                    prev_mappable_bid = Some(bid);
                }
                None => {
                    // Unmappable block
                    prev_bid = None;
                }
            }
        }

        // If we are still outlining, then it must be `yk_mt_control_point()` that is being
        // outlined, otherwise we may have encountered something fishy, like a longjmp().
        if let Some((ref outbid, _)) = self.outline_info
            && outbid != self.cp_block.as_ref().unwrap()
        {
            return Err(CompilationError::General(
                "irregular control flow detected (trace ended with outline successor pending)"
                    .into(),
            ));
        }

        if !self.finish_early {
            // If we have skipped blocks at the end of a trace (due to interpreter recursion) the
            // promotions and debug_str counts won't add up, so don't check them here.
            assert_eq!(self.promote_idx, self.promotions.len());
            assert_eq!(self.debug_str_idx, self.debug_strs.len());
        }
        let bid = self.cp_block.as_ref().unwrap();
        let blk = self.aot_mod.bblock(bid);
        let cpcall = blk.insts.iter().rev().nth(1).unwrap();
        debug_assert!(cpcall.is_control_point(self.aot_mod));
        let safepoint = cpcall.safepoint().unwrap();
        for idx in 0..safepoint.lives.len() {
            let aot_op = &safepoint.lives[idx];
            let jit_op = &self.local_map[&aot_op.to_inst_id()];
            self.jit_mod.push_header_end_var(jit_op.clone());
        }

        let tracekind = self.jit_mod.tracekind();
        match tracekind {
            TraceKind::HeaderOnly | TraceKind::HeaderAndBody => {
                assert!(matches!(self.endframe, TraceEndFrame::Same));
                self.jit_mod.push(jit_ir::Inst::TraceHeaderEnd(false))?;
            }
            TraceKind::Connector(_) => {
                assert!(matches!(self.endframe, TraceEndFrame::Same));
                self.jit_mod.push(jit_ir::Inst::TraceHeaderEnd(true))?;
            }
            TraceKind::Sidetrace(_) => {
                assert!(matches!(self.endframe, TraceEndFrame::Same));
                self.jit_mod.push(jit_ir::Inst::SidetraceEnd)?;
            }
            TraceKind::DifferentFrames => {
                if let TraceEndFrame::Entered = self.endframe {
                    if let Some((bid, safepoint)) = &self.last_interp_call.take() {
                        let deopt = self.create_guard(bid, None, false, safepoint)?;
                        self.jit_mod.push(deopt)?;
                    } else {
                        // We traced a recursive call to the interpreter inside of another
                        // outlined/unmappable function. Since these functions don't have
                        // safepoints attached to them we currently can't emit the deopt. Thus, at
                        // least for the time being, abort the trace.
                        return Err(CompilationError::General(
                            "Recursive interpreter call inside outlined function.".to_string(),
                        ));
                    }
                }
            }
        };

        Ok(self.jit_mod)
    }
}

/// A local version of [jit_ir::InlinedFrame] that deals with the fact that we build up information
/// about an inlined frame bit-by-bit using `Option`s, all of which will end up as `Some`.
#[derive(Debug, Clone)]
struct InlinedFrame {
    funcidx: Option<aot_ir::FuncIdx>,
    callinst: Option<aot_ir::InstId>,
    safepoint: Option<&'static aot_ir::DeoptSafepoint>,
    args: Vec<PackedOperand>,
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn build(
    mt: &Arc<MT>,
    aot_mod: &'static Module,
    ctrid: TraceId,
    ta_iter: Box<dyn AOTTraceIterator>,
    sti: Option<Arc<YkSideTraceInfo<super::codegen::x64::Register>>>,
    promotions: Box<[u8]>,
    debug_strs: Vec<String>,
    connector_tid: Option<Arc<dyn CompiledTrace>>,
    endframe: TraceEndFrame,
) -> Result<jit_ir::Module, CompilationError> {
    let tracekind = match endframe {
        TraceEndFrame::Same => {
            if let Some(x) = sti {
                TraceKind::Sidetrace(x)
            } else if let Some(connector_tid) = connector_tid {
                TraceKind::Connector(connector_tid)
            } else {
                TraceKind::HeaderOnly
            }
        }
        TraceEndFrame::Entered | TraceEndFrame::Left => TraceKind::DifferentFrames,
    };
    TraceBuilder::new(
        mt, tracekind, aot_mod, ctrid, promotions, debug_strs, endframe,
    )?
    .build(mt, ta_iter)
}
