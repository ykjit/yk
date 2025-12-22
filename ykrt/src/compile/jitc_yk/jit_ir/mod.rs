//! The Intermediate Representation (IR) of a trace.
//!
//!
//! ## General SSA properties
//!
//! The most important part of the IR is the
//! [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) *instruction sequence*
//! (stored in a [Vec]) created by [trace_builder]. Instructions are defined by the [Inst] enum.
//! Those instructions which produce a value (which is nearly all of them!) define a variable whose
//! "name" is the offset into the instruction vector.
//!
//! Since traces are linear, we do not have explicit Φ nodes in the IR. However, there are implicit
//! Φ nodes at the `TLoopStart` instruction: there is only one such instruction per trace; and
//! there are no other implicit Φ nodes.
//!
//! SSA variable "names" are offsets into a vector of instructions. For example, the instruction at
//! position 0 in the vector implicitly defines a variable `%0`.
//!
//!
//! ## Instruction operands
//!
//! Instructions contain zero or more *operands*. These are stored as [PackedOperand]s, which is an
//! efficient, opaque encoding. To operate on an operand we must *unpack* a [PackedOperand] to an
//! [Operand]. This also implicitly deconsts/decopied (see below): because of this aspect of
//! unpacking, it is important to work on [Operand]s, not [PackedOperand]s.
//!
//! The IR has three "special" instructions which are stored as normal instructions, but which are
//! hidden by deconsting/decopying (including when one views the IR):
//!
//!   * `Const`, which represents a constant. What, in the underlying IR, nominally looks
//!     as follows:
//!     ```text
//!     %7: i8 = 1i8
//!     %8: add %2, %7
//!     ```
//!     will be displayed *deconsted* as simply `%8: add %2, 1i8`. Note that this is
//!     deliberately indistinguishable from IR which directly stores `%8: add %2, 1i8`.
//!
//!   * `Copy` which is a (possibly chained) proxy for another instruction. What, in the
//!     underlying IR, nominally looks as follows:
//!     ```text
//!     %7: %6
//!     %8: add %2, %7
//!     ```
//!     will be displayed *decopied* as simply `%8: add %2, %6`. Note that this is
//!     deliberately indistinguishable from IR which directly stores `%8: add %2, %6`.
//!
//!   * `Tombstone` which represents an instruction which is no longer used. These are not
//!     displayed at all.
//!
//! Formally speaking, deconsting/decopying allows us to efficiently encode union sets: it is
//! equivalent to "forwarding" in RPython (see e.g. [this blog
//! post](https://pypy.org/posts/2022/07/toy-optimizer.html)).
//!
//!
//! ## Abbreviations and terminological conventions
//!
//! Because using the IR can often involve getting hold of data nested several layers deep, we also
//! use a number of abbreviations/conventions to keep the length of source down to something
//! manageable (in alphabetical order):
//!
//!  * `cidx`: a [ConstIdx].
//!  * `Const` and `const_`: a "constant"
//!  * `decl`: a "declaration" (e.g. a "function declaration" is a reference to an existing
//!    function somewhere else in the address space)
//!  * `iidx`: an [InstIdx].
//!  * `m`: the name conventionally given to the shared [Module] instance (i.e. `m: Module`)
//!  * `Idx`: "index"
//!  * `Inst`: "instruction"
//!  * `Ty`: "type"
//!  * `tyidx`: a [TyIdx].
//!
//! IR structures can be converted to human-readable strings either because:
//!  1. they implement [std::fmt::Display] directly.
//!  2. or, when they need extra information, they expose a `display()` method, which returns an
//!     object which implements [std::fmt::Display].
//!
//!
//! ## Canonicalisation
//!
//! JIT IR has a canonicalised form: that is the "shape" that later stages can weakly assume the IR
//! will be in. Canonicalisation is a weak promise, not a guarantee: later stages still have to
//! deal with the other cases, but since they're mostly expected not to occur, they may be handled
//! suboptimally if that makes the code easier.
//!
//! ## Side traces
//!
//! While we apply loop peeling to normal traces, this doesn't make sense for side-traces which
//! thus don't have [Inst::TraceHeaderStart], [Inst::TraceHeaderEnd], [Inst::TraceBodyStart], and
//! [Inst::TraceBodyEnd] instructions. Instead, side-traces have a single [Inst::SidetraceEnd]
//! instruction at the end of the trace. The operands for this label are stored in
//! [Module::trace_header_end].

mod dead_code;
#[cfg(test)]
mod parser;
#[cfg(any(debug_assertions, test))]
mod well_formed;

use super::{YkSideTraceInfo, aot_ir, codegen::x64::Register};
use crate::{
    compile::{CompilationError, CompiledTrace, jitc_yk::arbbitint::ArbBitInt},
    mt::TraceId,
};
use indexmap::IndexSet;
use std::{
    ffi::{CString, c_void},
    fmt,
    hash::Hash,
    mem,
    sync::Arc,
};
use strum::{EnumCount, EnumDiscriminants};
#[cfg(not(test))]
use ykaddr::addr::symbol_to_ptr;

// This is simple and can be shared across both IRs.
pub(crate) use super::aot_ir::{BinOp, FloatPredicate, FloatTy, Predicate};

/// Did this trace end in a different frame than it started in?
#[derive(Debug)]
pub(crate) enum TraceEndFrame {
    /// The trace stopped in the same frame that it started in.
    Same,
    /// The trace stopped after entering a new frame.
    Entered,
    /// The trace stopped after leaving the starting frame.
    Left,
}

impl TraceEndFrame {
    pub(crate) fn from_frames(start: *mut c_void, end: *mut c_void) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if end < start {
                Self::Entered
            } else if end > start {
                Self::Left
            } else {
                Self::Same
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        todo!()
    }
}

/// What kind of trace does this module represent?
#[derive(Clone)]
pub(crate) enum TraceKind {
    /// A self-contained looping trace that only has a header.
    HeaderOnly,
    /// A self-contained looping trace with a header and a body.
    HeaderAndBody,
    /// A header-only non-looping trace which jumps to the [CompiledTrace].
    Connector(Arc<dyn CompiledTrace>),
    /// A sidetrace: this will start at the point of a guard and will jump to
    /// [SideTraceInfo::target_ctr].
    Sidetrace(Arc<YkSideTraceInfo<Register>>),
    /// A trace where we stopped tracing in a different frame than we started in.
    DifferentFrames,
}

impl std::fmt::Debug for TraceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceKind::HeaderOnly => write!(f, "HeaderOnly"),
            TraceKind::HeaderAndBody => write!(f, "HeaderAndBody"),
            TraceKind::Connector(_) => write!(f, "Connector"),
            TraceKind::Sidetrace(_) => write!(f, "Sidetrace"),
            TraceKind::DifferentFrames => write!(f, "DifferentFrames"),
        }
    }
}

/// The `Module` is the top-level container for JIT IR.
///
/// The IR is conceptually a list of word-sized instructions containing indices into auxiliary
/// vectors.
///
/// The instruction stream of a [Module] is partially mutable:
/// - you may append new instructions to the end.
/// - you may replace and instruction with another.
/// - you may NOT remove an instruction.
#[derive(Debug)]
pub(crate) struct Module {
    /// What kind of trace does this module represent?
    tracekind: TraceKind,
    pub(crate) safepoint: Option<aot_ir::DeoptSafepoint>,
    /// The ID of the compiled trace this module will lead to.
    ///
    /// See the [Self::ctr_id] method for details.
    ctr_id: TraceId,
    /// The IR trace as a linear sequence of instructions.
    insts: Vec<Inst>,
    /// The arguments pool for [CallInst]s. Indexed by [ArgsIdx].
    args: Vec<PackedOperand>,
    /// The constant pool. Indexed by [ConstIdx].
    consts: IndexSet<ConstIndexSetWrapper>,
    /// The type pool. Indexed by [TyIdx].
    types: IndexSet<Ty>,
    /// The ordered sequence of trace parameters.
    params: Vec<yksmp::Location>,
    /// Addresses of root traces to jump to at the end of a side-trace.
    root_jump_ptr: *const libc::c_void,
    /// The type index of the void type. Cached for convenience.
    void_tyidx: TyIdx,
    /// The type index of a pointer type. Cached for convenience.
    ptr_tyidx: TyIdx,
    /// The type index of a 1-bit integer. Cached for convenience.
    int1_tyidx: TyIdx,
    /// The type index of an 8-bit integer. Cached for convenience.
    #[cfg(test)]
    int8_tyidx: TyIdx,
    /// The [ConstIdx] of the i1 value 1 / "true".
    true_constidx: ConstIdx,
    /// The [ConstIdx] of the i1 value 0 / "false".
    false_constidx: ConstIdx,
    /// The function declaration pool. These are declarations of externally compiled functions that
    /// the JITted trace might need to call. Indexed by [FuncDeclIdx].
    func_decls: IndexSet<FuncDecl>,
    /// The global variable declaration table.
    ///
    /// This is a collection of externally defined global variables that the trace may need to
    /// reference. Because they are externally initialised, these are *declarations*.
    global_decls: IndexSet<GlobalDecl>,
    /// Additional information for guards.
    guard_info: Vec<GuardInfo>,
    /// Indirect calls.
    indirect_calls: Vec<IndirectCallInst>,
    /// Live variables at the beginning of the trace body.
    pub(crate) trace_body_start: Vec<PackedOperand>,
    /// The ordered sequence of operands at the end of the trace body: there will be one per
    /// [Operand] at the start of the loop.
    pub(crate) trace_body_end: Vec<PackedOperand>,
    /// Live variables at the beginning the trace header.
    trace_header_start: Vec<PackedOperand>,
    /// Live variables at the end of the trace header.
    trace_header_end: Vec<PackedOperand>,
    /// The virtual address of the global variable pointer array.
    ///
    /// This is an array (added to the LLVM AOT module and AOT codegenned by ykllvm) containing a
    /// pointer to each global variable in the AOT module. The indices of the elements correspond
    /// with [aot_ir::GlobalDeclIdx]s.
    ///
    /// This is marked `cfg(not(test))` because unit tests are not built with ykllvm, and thus the
    /// array will be absent.
    #[cfg(not(test))]
    globalvar_ptrs: &'static [*const ()],
    /// The dynamically recorded debug strings, in the order that the corresponding
    /// `yk_debug_str()` calls were encountered in the trace.
    ///
    /// Indexed by [DebugStrIdx].
    debug_strs: Vec<String>,
}

impl Module {
    /// Create a new [Module].
    pub(crate) fn new(
        tracekind: TraceKind,
        ctr_id: TraceId,
        global_decls_len: usize,
    ) -> Result<Self, CompilationError> {
        Self::new_internal(tracekind, ctr_id, global_decls_len)
    }

    /// Returns this module's current [TraceKind]. Note: this can change as a result of calling
    /// [Self::set_tracekind]!
    pub(crate) fn tracekind(&self) -> &TraceKind {
        &self.tracekind
    }

    /// Returns this module's current [TraceKind]. Currently the only transition allowed is from
    /// [TraceKind::HeaderOnly] to [TraceKind::HeaderAndBody].
    pub(crate) fn set_tracekind(&mut self, tracekind: TraceKind) {
        match (&self.tracekind, &tracekind) {
            (&TraceKind::HeaderOnly, &TraceKind::HeaderAndBody) => (),
            (&TraceKind::HeaderOnly, &TraceKind::DifferentFrames) => (),
            (from, to) => panic!("Can't transition from a {from:?} trace to a {to:?} trace"),
        }
        self.tracekind = tracekind;
    }

    /// Returns the ID the module will have when it is compiled into a trace.
    pub(crate) fn ctrid(&self) -> TraceId {
        self.ctr_id
    }

    #[cfg(test)]
    pub(crate) fn new_testing() -> Self {
        Self::new_internal(TraceKind::HeaderOnly, TraceId::testing(), 0).unwrap()
    }

    pub(crate) fn new_internal(
        tracekind: TraceKind,
        ctr_id: TraceId,
        global_decls_len: usize,
    ) -> Result<Self, CompilationError> {
        // Create some commonly used types ahead of time. Aside from being convenient, this allows
        // us to find their (now statically known) indices in scenarios where Rust forbids us from
        // holding a mutable reference to the Module (and thus we cannot use [Module::tyidx]).
        let mut types = IndexSet::new();
        let void_tyidx = TyIdx::try_from(types.insert_full(Ty::Void).0)?;
        let ptr_tyidx = TyIdx::try_from(types.insert_full(Ty::Ptr).0)?;
        let int1_tyidx = TyIdx::try_from(types.insert_full(Ty::Integer(1)).0).unwrap();
        #[cfg(test)]
        let int8_tyidx = TyIdx::try_from(types.insert_full(Ty::Integer(8)).0).unwrap();

        let mut consts = IndexSet::new();
        let true_constidx = ConstIdx::try_from(
            consts
                .insert_full(ConstIndexSetWrapper(Const::Int(
                    int1_tyidx,
                    ArbBitInt::from_u64(1, 1),
                )))
                .0,
        )
        .unwrap();
        let false_constidx = ConstIdx::try_from(
            consts
                .insert_full(ConstIndexSetWrapper(Const::Int(
                    int1_tyidx,
                    ArbBitInt::from_u64(1, 0),
                )))
                .0,
        )
        .unwrap();

        // Find the global variable pointer array in the address space.
        //
        // FIXME: consider passing this in to the control point to avoid a dlsym().
        #[cfg(not(test))]
        let globalvar_ptrs = {
            let ptr = symbol_to_ptr(GLOBAL_PTR_ARRAY_SYM).unwrap() as *const *const ();
            unsafe { std::slice::from_raw_parts(ptr, global_decls_len) }
        };
        #[cfg(test)]
        assert_eq!(global_decls_len, 0);

        Ok(Self {
            tracekind,
            safepoint: None,
            ctr_id,
            insts: Vec::new(),
            args: Vec::new(),
            consts,
            types,
            params: Vec::new(),
            root_jump_ptr: std::ptr::null(),
            void_tyidx,
            ptr_tyidx,
            int1_tyidx,
            #[cfg(test)]
            int8_tyidx,
            true_constidx,
            false_constidx,
            func_decls: IndexSet::new(),
            global_decls: IndexSet::new(),
            guard_info: Vec::new(),
            indirect_calls: Vec::new(),
            trace_body_start: Vec::new(),
            trace_body_end: Vec::new(),
            trace_header_start: Vec::new(),
            trace_header_end: Vec::new(),
            #[cfg(not(test))]
            globalvar_ptrs,
            debug_strs: Vec::new(),
        })
    }

    /// Get a pointer to an AOT-compiled global variable by a JIT [GlobalDeclIdx].
    ///
    /// # Panics
    ///
    /// Panics if the address cannot be located.
    #[cfg(not(test))]
    pub(crate) fn globalvar_ptr(&self, idx: GlobalDeclIdx) -> *const () {
        let decl = self.global_decl(idx);
        // If the unwrap fails, then the AOT array was absent and something has gone wrong
        // during AOT codegen.
        self.globalvar_ptrs[usize::from(decl.global_ptr_idx())]
    }

    /// Returns the type index of [Ty::Void].
    pub(crate) fn void_tyidx(&self) -> TyIdx {
        self.void_tyidx
    }

    /// Returns the type index of [Ty::Ptr].
    pub(crate) fn ptr_tyidx(&self) -> TyIdx {
        self.ptr_tyidx
    }

    /// Returns the type index of a 1-bit integer.
    pub(crate) fn int1_tyidx(&self) -> TyIdx {
        self.int1_tyidx
    }

    /// Returns the type index of an 8-bit integer.
    #[cfg(test)]
    pub(crate) fn int8_tyidx(&self) -> TyIdx {
        self.int8_tyidx
    }

    /// Return the instruction at the specified index.
    ///
    /// # Panics
    ///
    /// If `iidx` points to a `Const`, `Copy`, or `Tombstone` instruction.
    pub(crate) fn inst(&self, iidx: InstIdx) -> Inst {
        match self.insts[usize::from(iidx)] {
            Inst::Const(_) | Inst::Copy(_) | Inst::Tombstone => {
                todo!("{:?}", self.insts[usize::from(iidx)])
            }
            x => x,
        }
    }

    /// Return the instruction at the specified index or `None` if it is a `Copy` instruction.
    pub(crate) fn inst_nocopy(&self, iidx: InstIdx) -> Option<Inst> {
        match self.insts[usize::from(iidx)] {
            Inst::Copy(_) => None,
            x => Some(x),
        }
    }

    /// Return the instruction at the specified index "raw", that is without decopying.
    ///
    /// This function has very few uses and unless you explicitly know why you're using it, you
    /// should instead use [Self::inst_no_copies] because not handling `Copy` instructions
    /// correctly leads to undefined behaviour.
    fn inst_raw(&self, iidx: InstIdx) -> Inst {
        self.insts[usize::from(iidx)]
    }

    pub(crate) fn push_indirect_call(
        &mut self,
        inst: IndirectCallInst,
    ) -> Result<IndirectCallIdx, CompilationError> {
        IndirectCallIdx::try_from(self.indirect_calls.len())
            .inspect(|_| self.indirect_calls.push(inst))
    }

    /// Return the indirect call at the specified index.
    pub(crate) fn indirect_call(&self, idx: IndirectCallIdx) -> &IndirectCallInst {
        &self.indirect_calls[usize::from(idx)]
    }

    /// Push an instruction to the end of the [Module].
    pub(crate) fn push(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        InstIdx::try_from(self.insts.len()).inspect(|_| self.insts.push(inst))
    }

    /// Iterate, in order, over the `InstIdx`s of this module skipping `Const`, `Copy`, and
    /// `Tombstone` instructions.
    pub(crate) fn iter_skipping_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (InstIdx, Inst)> + '_ {
        (0..self.insts.len()).filter_map(|i| {
            let inst = &self.insts[i];
            if !matches!(inst, Inst::Const(_) | Inst::Copy(_) | Inst::Tombstone) {
                // The `unchecked_from` is safe because we know from `Self::push` that we can't have
                // exceeded `InstIdx`'s bounds.
                Some((InstIdx::unchecked_from(i), *inst))
            } else {
                None
            }
        })
    }

    /// Replace the instruction at `iidx` with `inst`.
    pub(crate) fn replace(&mut self, iidx: InstIdx, inst: Inst) {
        self.insts[usize::from(iidx)] = inst;
    }

    /// Replace the instruction in `iidx` with an instruction that will generate `op`. In other
    /// words, `Operand::Var(...)` will become `Inst::Copy` and `Operand::Const` will become
    /// `Inst::Const`. This is a convenience function over [Self::replace].
    pub(crate) fn replace_with_op(&mut self, iidx: InstIdx, op: Operand) {
        match op {
            Operand::Var(op_iidx) => self.replace(iidx, Inst::Copy(op_iidx)),
            Operand::Const(cidx) => self.replace(iidx, Inst::Const(cidx)),
        }
    }

    /// Push an instruction to the end of the [Module] and create a local variable [Operand] out of
    /// the value that the instruction defines.
    ///
    /// This is useful for forwarding the local variable a instruction defines as operand of a
    /// subsequent instruction: an idiom used a lot (but not exclusively) in testing.
    ///
    /// Note: it is undefined behaviour to push an instruction that does not define a local
    /// variable.
    pub(crate) fn push_and_make_operand(
        &mut self,
        inst: Inst,
    ) -> Result<Operand, CompilationError> {
        // Assert that `inst` defines a local var.
        debug_assert!(inst.def_type(self).is_some());
        InstIdx::try_from(self.insts.len()).map(|x| {
            self.insts.push(inst);
            Operand::Var(x)
        })
    }

    /// Returns the [InstIdx] of the last instruction in this module. Note that this might be a
    /// `Tombstone` or other "internal" instruction.
    ///
    /// # Panics
    ///
    /// If this module has no instructions.
    pub(crate) fn last_inst_idx(&self) -> InstIdx {
        // Assuming `x > 0`, then if `x` fits within `InstIdx`'s bounds, `x - 1` must also fit, so
        // the `unchecked_from` cannot fail.
        InstIdx::unchecked_from(self.insts.len().checked_sub(1).unwrap())
    }

    pub(crate) fn insts_len(&self) -> usize {
        self.insts.len()
    }

    /// Push a slice of arguments into the args pool.
    ///
    /// # Panics
    ///
    /// If `args` would overflow the index type.
    pub(crate) fn push_args(&mut self, args: Vec<Operand>) -> Result<ArgsIdx, CompilationError> {
        ArgsIdx::try_from(self.args.len())
            .inspect(|_| self.args.extend(args.iter().map(PackedOperand::new)))
    }

    /// Return the argument at args index `idx`.
    ///
    /// # Panics
    ///
    /// If `idx` is out of bounds.
    pub(crate) fn arg(&self, idx: ArgsIdx) -> Operand {
        self.args[usize::from(idx)].unpack(self)
    }

    /// Push the location of a trace parameter and return its `ParamIdx`.
    pub(crate) fn push_param(&mut self, loc: yksmp::Location) -> ParamIdx {
        let off = self.params.len();
        self.params.push(loc);
        ParamIdx::try_from(off).unwrap()
    }

    /// Return the parameter at a given [InstIdx].
    pub(crate) fn param(&self, pidx: ParamIdx) -> &yksmp::Location {
        &self.params[usize::from(pidx)]
    }

    /// Add a [Ty] to the types pool and return its index. If the [Ty] already exists, an existing
    /// index will be returned.
    pub(crate) fn insert_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        let (i, _) = self.types.insert_full(ty);
        TyIdx::try_from(i)
    }

    /// Return the [Ty] for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn type_(&self, idx: TyIdx) -> &Ty {
        self.types.get_index(usize::from(idx)).unwrap()
    }

    /// How many [Ty]s does this module contain?
    #[cfg(test)]
    pub(crate) fn types_len(&self) -> usize {
        self.types.len()
    }

    /// Add a constant to the pool and return its index. If the constant already exists, an
    /// existing index will be returned.
    pub(crate) fn insert_const(&mut self, c: Const) -> Result<ConstIdx, CompilationError> {
        let (i, _) = self.consts.insert_full(ConstIndexSetWrapper(c));
        ConstIdx::try_from(i)
    }

    /// Return the const for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn const_(&self, idx: ConstIdx) -> &Const {
        &self.consts.get_index(usize::from(idx)).unwrap().0
    }

    /// Return the [ConstIdx] of the `i1` value for 1/true.
    pub(crate) fn true_constidx(&self) -> ConstIdx {
        self.true_constidx
    }

    /// Return the [ConstIdx] of the `i1` value for 0/false.
    pub(crate) fn false_constidx(&self) -> ConstIdx {
        self.false_constidx
    }

    /// Add a new [GlobalDecl] to the pool and return its index. If the [GlobalDecl] already
    /// exists, an existing index will be returned.
    pub fn insert_global_decl(
        &mut self,
        gd: GlobalDecl,
    ) -> Result<GlobalDeclIdx, CompilationError> {
        let (i, _) = self.global_decls.insert_full(gd);
        GlobalDeclIdx::try_from(i)
    }

    /// Return the global declaration for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn global_decl(&self, idx: GlobalDeclIdx) -> &GlobalDecl {
        self.global_decls.get_index(usize::from(idx)).unwrap()
    }

    #[cfg(test)]
    pub(crate) fn func_decls_len(&self) -> usize {
        self.func_decls.len()
    }

    /// Add a [FuncDecl] to the function declarations pool and return its index. If the [FuncDecl]
    /// already exists, an existing index will be returned.
    pub(crate) fn insert_func_decl(
        &mut self,
        fd: FuncDecl,
    ) -> Result<FuncDeclIdx, CompilationError> {
        let (i, _) = self.func_decls.insert_full(fd);
        FuncDeclIdx::try_from(i)
    }

    /// Return the [FuncDecl] for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds
    pub(crate) fn func_decl(&self, idx: FuncDeclIdx) -> &FuncDecl {
        self.func_decls.get_index(usize::from(idx)).unwrap()
    }

    /// Return the type of the function declaration.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds
    pub(crate) fn func_type(&self, idx: FuncDeclIdx) -> &FuncTy {
        match self.type_(self.func_decl(idx).tyidx) {
            Ty::Func(ft) => ft,
            _ => unreachable!(),
        }
    }

    /// Find a function declaration by name. This has linear search time and is only intended for
    /// use when testing.
    ///
    /// # Panics
    ///
    /// If there is no function declaration `name`.
    #[cfg(test)]
    pub(crate) fn find_func_decl_idx_by_name(&self, name: &str) -> FuncDeclIdx {
        FuncDeclIdx::try_from(
            self.func_decls
                .iter()
                .position(|x| x.name() == name)
                .unwrap(),
        )
        .unwrap()
    }

    pub(crate) fn push_guardinfo(
        &mut self,
        info: GuardInfo,
    ) -> Result<GuardInfoIdx, CompilationError> {
        GuardInfoIdx::try_from(self.guard_info.len()).inspect(|_| self.guard_info.push(info))
    }

    pub(crate) fn guard_info(&self, gidx: GuardInfoIdx) -> &GuardInfo {
        &self.guard_info[usize::from(gidx)]
    }

    pub(crate) fn trace_header_start(&self) -> &[PackedOperand] {
        &self.trace_header_start
    }

    /// Return the position in the list of live variables at the trace header start, if any, of
    /// `op`.
    pub(crate) fn trace_header_start_position(&self, op: Operand) -> Option<usize> {
        let pop = PackedOperand::new(&op);
        self.trace_header_start.iter().position(|x| x == &pop)
    }

    pub(crate) fn trace_header_end(&self) -> &[PackedOperand] {
        &self.trace_header_end
    }

    /// Return the position in the list of live variables at the trace header end, if any, of
    /// `op`.
    pub(crate) fn trace_header_end_position(&self, op: Operand) -> Option<usize> {
        let pop = PackedOperand::new(&op);
        self.trace_header_end.iter().position(|x| x == &pop)
    }

    pub(crate) fn trace_body_start(&self) -> &[PackedOperand] {
        &self.trace_body_start
    }

    pub(crate) fn push_body_start_var(&mut self, op: Operand) {
        self.trace_header_start.push(PackedOperand::new(&op));
    }

    pub(crate) fn push_debug_str(&mut self, msg: String) -> Result<DebugStrIdx, CompilationError> {
        DebugStrIdx::try_from(self.debug_strs.len()).inspect(|_| self.debug_strs.push(msg))
    }

    /// Return the loop jump operands.
    pub(crate) fn trace_body_end(&self) -> &[PackedOperand] {
        &self.trace_body_end
    }

    pub(crate) fn push_header_end_var(&mut self, op: Operand) {
        self.trace_header_end.push(PackedOperand::new(&op));
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "; compiled trace ID #{}\n", self.ctr_id)?;
        for x in &self.func_decls {
            writeln!(
                f,
                "func_decl {} {}",
                x.name(),
                self.type_(x.tyidx()).display(self)
            )?;
        }
        for g in &self.global_decls {
            let tl = if g.is_threadlocal() { " tls" } else { "" };
            writeln!(
                f,
                "global_decl{} @{}",
                tl,
                g.name.to_str().unwrap_or("<not valid UTF-8>")
            )?;
        }
        write!(f, "\nentry:")?;
        for (iidx, inst) in self.iter_skipping_insts() {
            write!(f, "\n  {}", inst.display(self, iidx))?
        }

        Ok(())
    }
}

/// The declaration of a global variable.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlobalDecl {
    /// The name of the delcaration.
    name: CString,
    /// Whether the declaration is thread-local.
    is_threadlocal: bool,
    /// The declaration's index in the global variable pointer array.
    global_ptr_idx: aot_ir::GlobalDeclIdx,
}

impl GlobalDecl {
    pub(crate) fn new(
        name: CString,
        is_threadlocal: bool,
        global_ptr_idx: aot_ir::GlobalDeclIdx,
    ) -> Self {
        Self {
            name,
            is_threadlocal,
            global_ptr_idx,
        }
    }

    /// Return whether the declaration is a thread local.
    pub(crate) fn is_threadlocal(&self) -> bool {
        self.is_threadlocal
    }

    /// Return the global variable's name.
    #[cfg(not(test))]
    pub(crate) fn name(&self) -> &CString {
        &self.name
    }

    /// Return the declaration's index in the global variable pointer array.
    #[cfg(not(test))]
    pub(crate) fn global_ptr_idx(&self) -> aot_ir::GlobalDeclIdx {
        self.global_ptr_idx
    }
}

/// Bit fiddling.
///
/// In the constants below:
///  * `*_SIZE`: the size of a field in bits.
///  * `*_MASK`: a mask with one bits occupying the field in question.
///  * `*_SHIFT`: the number of bits required to left shift a field's value into position (from the
///    LSB).
///
const OPERAND_IDX_MASK: u32 = (1 << 23) - 1;

/// The largest operand index we can express in 23 bits.
const MAX_OPERAND_IDX: u32 = (1 << 23) - 1;

/// The symbol name of the global variable pointers array.
#[cfg(not(test))]
const GLOBAL_PTR_ARRAY_SYM: &str = "__yk_globalvar_ptrs";

/// A packed 24-bit unsigned integer.
#[repr(Rust, packed)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct U24([u8; 3]);

impl TryFrom<usize> for U24 {
    type Error = ();

    fn try_from(x: usize) -> Result<Self, Self::Error> {
        if x >= 1 << 24 {
            Err(())
        } else {
            let b0 = x & 0xff;
            let b1 = (x & 0xff00) >> 8;
            let b2 = (x & 0xff0000) >> 16;
            Ok(Self([b2 as u8, b1 as u8, b0 as u8]))
        }
    }
}

impl From<U24> for usize {
    fn from(x: U24) -> Self {
        static_assertions::const_assert!(mem::size_of::<usize>() >= 3);
        let b0 = x.0[0] as usize; // most-significant byte.
        let b1 = x.0[1] as usize;
        let b2 = x.0[2] as usize;
        (b0 << 16) | (b1 << 8) | b2
    }
}

/// Helper to create index overflow errors.
///
/// FIXME: all of these should be checked at compile time.
fn index_overflow(typ: &str) -> CompilationError {
    CompilationError::LimitExceeded(format!("index overflow: {typ}"))
}

// Generate common methods for 24-bit index types.
macro_rules! index_24bit {
    ($struct:ident) => {
        impl From<$struct> for usize {
            fn from(x: $struct) -> Self {
                usize::from(x.0)
            }
        }

        impl TryFrom<usize> for $struct {
            type Error = CompilationError;

            fn try_from(v: usize) -> Result<Self, Self::Error> {
                match U24::try_from(v) {
                    Ok(x) => Ok(Self(x)),
                    Err(()) => Err(index_overflow(stringify!($struct))),
                }
            }
        }
    };
}

// Generate common methods for 16-bit index types.
macro_rules! index_16bit {
    ($struct:ident) => {
        #[allow(dead_code)]
        impl $struct {
            /// What is the maximum value this index type can represent?
            pub(crate) fn max() -> Self {
                Self(u16::MAX)
            }

            /// Create an instance of `$struct` without checking whether `v` exceeds the underlying
            /// type's bounds. If it does exceed those bounds, the result will be an instance of
            /// this struct whose values is the `MAX` value the underlying type can represent.
            pub(crate) fn unchecked_from(v: usize) -> Self {
                debug_assert!(v <= usize::from(u16::MAX));
                Self(v as u16)
            }

            pub(crate) fn checked_add(&self, other: usize) -> Result<Self, CompilationError> {
                Self::try_from(usize::from(self.0) + other)
            }

            pub(crate) fn checked_sub(&self, other: usize) -> Result<Self, CompilationError> {
                Self::try_from(usize::from(self.0) - other)
            }
        }

        impl From<$struct> for u16 {
            fn from(s: $struct) -> u16 {
                s.0
            }
        }

        impl From<$struct> for usize {
            fn from(s: $struct) -> usize {
                s.0.into()
            }
        }

        impl TryFrom<usize> for $struct {
            type Error = CompilationError;

            fn try_from(v: usize) -> Result<Self, Self::Error> {
                u16::try_from(v)
                    .map_err(|_| index_overflow(stringify!($struct)))
                    .map(|u| Self(u))
            }
        }

        impl fmt::Display for $struct {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
                write!(f, "{}", self.0)
            }
        }
    };
}

// Generate common methods for 32-bit index types.
macro_rules! index_32bit {
    ($struct:ident) => {
        #[allow(dead_code)]
        impl $struct {
            /// What is the maximum value this index type can represent?
            pub(crate) fn max() -> Self {
                Self(u32::MAX)
            }

            /// Create an instance of `$struct` without checking whether `v` exceeds the underlying
            /// type's bounds. If it does exceed those bounds, the result will be an instance of
            /// this struct whose values is the `MAX` value the underlying type can represent.
            pub(crate) fn unchecked_from(v: usize) -> Self {
                debug_assert!(v <= usize::try_from(u32::MAX).unwrap());
                Self(v as u32)
            }

            pub(crate) fn checked_add(&self, other: usize) -> Result<Self, CompilationError> {
                Self::try_from(usize::try_from(self.0).unwrap() + other)
            }

            pub(crate) fn checked_sub(&self, other: usize) -> Result<Self, CompilationError> {
                Self::try_from(usize::try_from(self.0).unwrap() - other)
            }
        }

        impl From<$struct> for u32 {
            fn from(s: $struct) -> u32 {
                s.0
            }
        }

        impl From<$struct> for usize {
            fn from(s: $struct) -> usize {
                s.0.try_into().unwrap()
            }
        }

        impl TryFrom<usize> for $struct {
            type Error = CompilationError;

            fn try_from(v: usize) -> Result<Self, Self::Error> {
                u32::try_from(v)
                    .map_err(|_| index_overflow(stringify!($struct)))
                    .map(|u| Self(u))
            }
        }

        impl fmt::Display for $struct {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
                write!(f, "{}", self.0)
            }
        }
    };
}

/// A function declaration index.
///
/// One of these is an index into the [Module::func_decls].
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FuncDeclIdx(U24);
index_24bit!(FuncDeclIdx);

/// A type index.
///
/// One of these is an index into the [Module::types].
///
/// A type index uniquely identifies a [Ty] in a [Module]. You can rely on this uniquness
/// property for type checking: you can compare type indices instead of the corresponding [Ty]s.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TyIdx(U24);
index_24bit!(TyIdx);

/// An argument index. This denotes the start of a slice into [Module::args].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct ArgsIdx(u16);
index_16bit!(ArgsIdx);

/// A constant index.
///
/// One of these is an index into the [Module::consts].
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd)]
pub(crate) struct ConstIdx(u32);
index_32bit!(ConstIdx);

/// A guard info index.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GuardInfoIdx(pub(crate) u16);
index_16bit!(GuardInfoIdx);

/// A global variable declaration index.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GlobalDeclIdx(U24);
index_24bit!(GlobalDeclIdx);

/// An indirect call index.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct IndirectCallIdx(U24);
index_24bit!(IndirectCallIdx);

/// An index into [Module::insts].
#[derive(Debug, Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct InstIdx(u32);
index_32bit!(InstIdx);

/// An index into [Module::params].
#[derive(Debug, Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ParamIdx(u16);
index_16bit!(ParamIdx);

/// An index into [Module::debug_strs]
#[derive(Debug, Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct DebugStrIdx(u16);
index_16bit!(DebugStrIdx);

/// A function's type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FuncTy {
    /// Ty indices for the function's parameters.
    param_tyidxs: Vec<TyIdx>,
    /// Ty index of the function's return type.
    ret_tyidx: TyIdx,
    /// Is the function vararg?
    is_vararg: bool,
}

impl FuncTy {
    pub(crate) fn new(param_tyidxs: Vec<TyIdx>, ret_tyidx: TyIdx, is_vararg: bool) -> Self {
        Self {
            param_tyidxs,
            ret_tyidx,
            is_vararg,
        }
    }

    /// Return the number of parameters the function accepts (not including varargs).
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn num_params(&self) -> usize {
        self.param_tyidxs.len()
    }

    /// Return a slice of this function's non-varargs parameters.
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn param_tys(&self) -> &[TyIdx] {
        &self.param_tyidxs
    }

    /// Returns whether the function type has vararg parameters.
    pub(crate) fn is_vararg(&self) -> bool {
        self.is_vararg
    }

    /// Returns the type of the return value.
    pub(crate) fn ret_type<'a>(&self, m: &'a Module) -> &'a Ty {
        m.type_(self.ret_tyidx)
    }

    /// Returns the type index of the return value.
    pub(crate) fn ret_tyidx(&self) -> TyIdx {
        self.ret_tyidx
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StructTy {
    /// Total size in bits of this struct including alignment.
    bit_size: u64,
    /// The types of the fields.
    field_tyidxs: Vec<TyIdx>,
    /// The bit offsets of the fields (taking into account any required padding for alignment).
    field_bit_offs: Vec<usize>,
}

impl StructTy {
    pub(crate) fn new(bit_size: u64, field_tyidxs: Vec<TyIdx>, field_bit_offs: Vec<usize>) -> Self {
        StructTy {
            bit_size,
            field_tyidxs,
            field_bit_offs,
        }
    }
}

/// A type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Ty {
    Void,
    /// A fixed-width integer type.
    ///
    /// Note:
    ///   1. These integers range in size from 1..2^23 (inc.) bits (stored in the `u32`). This is
    ///      inherited [from LLVM's integer type](https://llvm.org/docs/LangRef.html#integer-type).
    ///   2. Signedness is not specified. Interpretation of the bit pattern is delegated to operations
    ///      upon the integer.
    Integer(u32),
    Ptr,
    Func(FuncTy),
    Float(FloatTy),
    Struct(StructTy),
    Unimplemented(String),
}

impl Ty {
    /// Returns the size of the type in bytes, or `None` if asking the size makes no sense.
    ///
    /// To get the size in *bits*, use [Self::bit_size] instead. Multiplying the result of
    /// `byte_size()` by 8 is not correct.
    pub(crate) fn byte_size(&self) -> Option<usize> {
        // u16/u32 -> usize conversions could theoretically fail on some arches (which we probably
        // won't ever support).
        match self {
            Self::Void => Some(0),
            Self::Integer(bits) => Some(usize::try_from(bits.div_ceil(8)).unwrap()),
            Self::Ptr => {
                // FIXME: In theory pointers to different types could be of different sizes. We
                // should really ask LLVM how big the pointer was when it codegenned the
                // interpreter, and on a per-pointer basis.
                //
                // For now we assume (and ykllvm assserts this) that all pointers are void
                // pointer-sized.
                Some(mem::size_of::<*const c_void>())
            }
            Self::Func(_) => None,
            Self::Float(ft) => Some(match ft {
                FloatTy::Float => mem::size_of::<f32>(),
                FloatTy::Double => mem::size_of::<f64>(),
            }),
            Self::Struct(s) => Some(usize::try_from(s.bit_size.div_ceil(8)).unwrap()),
            Self::Unimplemented(_) => None,
        }
    }

    /// Returns the size of the type in bits, or `None` if asking the size makes no sense.
    pub(crate) fn bitw(&self) -> Option<u32> {
        match self {
            Self::Void => None,
            Self::Integer(bitw) => Some(*bitw),
            Self::Ptr => {
                // We make the same assumptions about pointer size as in Self::byte_size().
                u32::try_from(mem::size_of::<*const c_void>() * 8).ok()
            }
            Self::Func(_) => None,
            Self::Float(ft) => match ft {
                FloatTy::Float => u32::try_from(mem::size_of::<f32>() * 8).ok(),
                FloatTy::Double => u32::try_from(mem::size_of::<f64>() * 8).ok(),
            },
            Self::Struct(s) => Some(u32::try_from(s.bit_size).unwrap()),
            Self::Unimplemented(_) => None,
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableTy<'a> {
        DisplayableTy { ty: self, m }
    }
}

pub(crate) struct DisplayableTy<'a> {
    ty: &'a Ty,
    m: &'a Module,
}

impl fmt::Display for DisplayableTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            Ty::Void => write!(f, "void"),
            Ty::Integer(num_bits) => write!(f, "i{}", *num_bits),
            Ty::Ptr => write!(f, "ptr"),
            Ty::Func(x) => {
                let mut args = x
                    .param_tyidxs
                    .iter()
                    .map(|x| self.m.type_(*x).display(self.m).to_string())
                    .collect::<Vec<_>>();
                if x.is_vararg() {
                    args.push("...".to_string());
                }
                if x.ret_tyidx() == self.m.void_tyidx {
                    write!(f, "({})", args.join(", "))
                } else {
                    write!(
                        f,
                        "({}) -> {}",
                        args.join(", "),
                        self.m.type_(x.ret_tyidx()).display(self.m)
                    )
                }
            }
            Ty::Float(ft) => write!(f, "{ft}"),
            Ty::Struct(s) => {
                write!(
                    f,
                    "{{{}}}",
                    s.field_tyidxs
                        .iter()
                        .enumerate()
                        .map(|(i, ti)| format!(
                            "{}: {}",
                            s.field_bit_offs[i],
                            self.m.type_(*ti).display(self.m)
                        ))
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }
            Ty::Unimplemented(_) => write!(f, "?type"),
        }
    }
}

/// An (externally defined, in the AOT code) function declaration.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FuncDecl {
    name: String,
    tyidx: TyIdx,
}

impl FuncDecl {
    pub(crate) fn new(name: String, tyidx: TyIdx) -> Self {
        Self { name, tyidx }
    }

    /// Return the name of this function declaration.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }
}

/// The packed representation of an instruction operand. In general, you should *store*
/// `PackedOperand`s but *operate* on (unpacked) [Operand]s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PackedOperand(
    /// # Encoding
    ///
    /// ```ignore
    ///  1             15
    /// +---+--------------------------+
    /// | k |         index            |
    /// +---+--------------------------+
    /// ```
    ///
    ///  - `k=0`: `index` is a local variable index
    ///  - `k=1`: `index` is a constant index
    ///
    ///  The IR can represent 2^{15} = 32768 locals, and as many constants.
    u32,
);

impl PackedOperand {
    pub fn new(op: &Operand) -> Self {
        match op {
            Operand::Var(lidx) => {
                debug_assert!(u32::from(*lidx) <= MAX_OPERAND_IDX);
                PackedOperand(u32::from(*lidx))
            }
            Operand::Const(constidx) => {
                debug_assert!(u32::from(*constidx) <= MAX_OPERAND_IDX);
                PackedOperand(u32::from(*constidx) | !OPERAND_IDX_MASK)
            }
        }
    }

    /// Unpacks and decopies a [PackedOperand] into a [Operand].
    pub(crate) fn unpack(&self, m: &Module) -> Operand {
        if (self.0 & !OPERAND_IDX_MASK) == 0 {
            let mut iidx = InstIdx(self.0);
            loop {
                match m.inst_raw(iidx) {
                    Inst::Const(x) => {
                        return Operand::Const(x);
                    }
                    Inst::Copy(x) => {
                        iidx = x;
                    }
                    _ => {
                        return Operand::Var(iidx);
                    }
                }
            }
        } else {
            Operand::Const(ConstIdx(self.0 & OPERAND_IDX_MASK))
        }
    }
}

/// An operand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Operand {
    /// This operand references another SSA variable.
    Var(InstIdx),
    /// This operand is a constant.
    Const(ConstIdx),
}

impl Operand {
    /// Returns the size of the operand in bytes. Assumes no padding is required for alignment.
    ///
    /// # Panics
    ///
    /// Panics if asking for the size make no sense for this operand.
    pub(crate) fn byte_size(&self, m: &Module) -> usize {
        match self {
            Self::Var(l) => m.inst_raw(*l).def_byte_size(m),
            Self::Const(cidx) => m.type_(m.const_(*cidx).tyidx(m)).byte_size().unwrap(),
        }
    }

    /// Returns the size of the operand in bits.
    ///
    /// # Panics
    ///
    /// Panics if asking for the size make no sense for this operand.
    pub(crate) fn bitw(&self, m: &Module) -> u32 {
        match self {
            Self::Var(l) => m.inst_raw(*l).def_bitw(m),
            Self::Const(cidx) => m.type_(m.const_(*cidx).tyidx(m)).bitw().unwrap(),
        }
    }

    /// Returns the type index of the operand.
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        match self {
            Self::Var(l) => m.inst_raw(*l).tyidx(m),
            Self::Const(c) => m.const_(*c).tyidx(m),
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableOperand<'a> {
        DisplayableOperand { operand: self, m }
    }

    /// If this [Operand] represents a local instruction, call `f` with its `InstIdx`.
    fn map_iidx<F>(&self, f: &mut F)
    where
        F: FnMut(InstIdx),
    {
        if let Operand::Var(iidx) = self {
            f(*iidx)
        }
    }

    /// If self is [Operand::Var] and references a [Inst::Copy], recursively search until a
    /// non-`Copy` instruction is found.
    pub(crate) fn decopy(&self, m: &Module) -> Self {
        let mut ret = self.clone();
        loop {
            match ret {
                Self::Var(iidx) => {
                    if let Inst::Copy(next_iidx) = m.inst_raw(iidx) {
                        ret = Operand::Var(next_iidx);
                    } else {
                        return ret;
                    }
                }
                Self::Const(_) => return ret,
            }
        }
    }
}

pub(crate) struct DisplayableOperand<'a> {
    operand: &'a Operand,
    m: &'a Module,
}

impl fmt::Display for DisplayableOperand<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.operand {
            Operand::Var(idx) => match self.m.inst_raw(*idx) {
                Inst::Const(c) => {
                    write!(f, "{}", self.m.const_(c).display(self.m))
                }
                Inst::Copy(idx) => {
                    write!(f, "%{idx}")
                }
                _ => write!(f, "%{idx}"),
            },
            Operand::Const(idx) => write!(f, "{}", self.m.const_(*idx).display(self.m)),
        }
    }
}

/// A constant.
///
/// Note that this struct deliberately does not implement `PartialEq` (or `Eq`): two instances of
/// `Const` may represent the same underlying constant, but (because of floats), you as the user
/// need to determine what notion of equality you wish to use on a given const.
#[derive(Clone, Debug)]
pub(crate) enum Const {
    Float(TyIdx, f64),
    /// A constant integer. Note that, as in LLVM IR, this has no inherent signedness.
    Int(TyIdx, ArbBitInt),
    Ptr(usize),
}

impl Const {
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        match self {
            Const::Float(tyidx, _) => *tyidx,
            Const::Int(tyidx, _) => *tyidx,
            Const::Ptr(_) => m.ptr_tyidx,
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableConst<'a> {
        DisplayableConst { const_: self, m }
    }
}

pub(crate) struct DisplayableConst<'a> {
    const_: &'a Const,
    m: &'a Module,
}

impl fmt::Display for DisplayableConst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.const_ {
            Const::Float(tyidx, v) => match self.m.type_(*tyidx) {
                Ty::Float(FloatTy::Float) => write!(f, "{}float", *v as f32),
                Ty::Float(FloatTy::Double) => write!(f, "{v}double"),
                _ => unreachable!(),
            },
            Const::Int(_, x) => {
                write!(f, "{}i{}", x.to_zero_ext_u64().unwrap(), x.bitw())
            }
            Const::Ptr(x) => write!(f, "{:#x}", *x),
        }
    }
}

/// This wrapper is deliberately private to this module and is solely used to allow us to maintain
/// a hashable pool of constants. Because `Const` doesn't implement `PartialEq`, we provide a
/// manual implementation here, knowing that it allows some duplicate constants to be stored in the
/// constant pool. There's no way around this, because of floats.
#[derive(Clone, Debug)]
struct ConstIndexSetWrapper(Const);

impl PartialEq for ConstIndexSetWrapper {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Const::Float(lhs_tyidx, lhs_v), Const::Float(rhs_tyidx, rhs_v)) => {
                // We treat floats as bit patterns: because we can accept duplicates, this is
                // acceptable.
                lhs_tyidx == rhs_tyidx && lhs_v.to_bits() == rhs_v.to_bits()
            }
            (Const::Int(lhs_tyidx, lhs), Const::Int(rhs_tyidx, rhs)) => {
                lhs_tyidx == rhs_tyidx && lhs == rhs
            }
            (Const::Ptr(lhs_v), Const::Ptr(rhs_v)) => lhs_v == rhs_v,
            (_, _) => false,
        }
    }
}

impl Eq for ConstIndexSetWrapper {}

impl Hash for ConstIndexSetWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match &self.0 {
            Const::Float(tyidx, v) => {
                tyidx.hash(state);
                // We treat floats as bit patterns: because we can accept duplicates, this is
                // acceptable.
                v.to_bits().hash(state);
            }
            Const::Int(tyidx, x) => {
                tyidx.hash(state);
                x.hash(state);
            }
            Const::Ptr(v) => v.hash(state),
        }
    }
}

#[derive(Debug)]
/// Stores additional guard information.
pub(crate) struct GuardInfo {
    /// The id of the AOT block that created this guard.
    bid: aot_ir::BBlockId,
    /// Live variables, mapping AOT vars to JIT [Operand]s.
    live_vars: Vec<(aot_ir::InstId, PackedOperand)>,
    /// Inlined frames info.
    /// FIXME With this field, the aotlives field is redundant.
    inlined_frames: Vec<InlinedFrame>,
    /// What AOT safepoint does this guard correspond to? This is used solely for debugging
    /// purposes.
    safepoint_id: u64,
}

impl GuardInfo {
    pub(crate) fn new(
        bid: aot_ir::BBlockId,
        live_vars: Vec<(aot_ir::InstId, PackedOperand)>,
        inlined_frames: Vec<InlinedFrame>,
        safepoint_id: u64,
    ) -> Self {
        Self {
            bid,
            live_vars,
            inlined_frames,
            safepoint_id,
        }
    }

    /// Return the AOT block for this guard.
    pub(crate) fn bid(&self) -> &aot_ir::BBlockId {
        &self.bid
    }

    /// Return the `AOT instruction -> PackedOperand` mapping for this guard.
    pub(crate) fn live_vars(&self) -> &[(aot_ir::InstId, PackedOperand)] {
        &self.live_vars
    }

    /// Return the [InlinedFunc]s for this guard.
    pub(crate) fn inlined_frames(&self) -> &[InlinedFrame] {
        &self.inlined_frames
    }

    pub(crate) fn safepoint_id(&self) -> u64 {
        self.safepoint_id
    }
}

/// An abstract call frame for functions inlined in a trace. This contains enough information for a
/// failing guard to reconstruct real call frames on the stack.
#[derive(Debug, Clone)]
pub(crate) struct InlinedFrame {
    // The call instruction that led to [funcidx] being inlined.
    pub(crate) callinst: Option<aot_ir::InstId>,
    // The function that was inlined to create this abstract frame. Will be `None` for the top-most
    // function.
    pub(crate) funcidx: aot_ir::FuncIdx,
    /// The deopt safepoint for [callinst].
    pub(crate) safepoint: &'static aot_ir::DeoptSafepoint,
}

impl InlinedFrame {
    pub(crate) fn new(
        callinst: Option<aot_ir::InstId>,
        funcidx: aot_ir::FuncIdx,
        safepoint: &'static aot_ir::DeoptSafepoint,
    ) -> InlinedFrame {
        InlinedFrame {
            callinst,
            funcidx,
            safepoint,
        }
    }
}

/// An IR instruction.
#[derive(Clone, Copy, Debug, EnumCount, EnumDiscriminants)]
#[repr(u8)]
pub(crate) enum Inst {
    // "Internal" IR instructions: these don't correspond to IR that a user interpreter can
    // express, but are used either for efficient representation of the IR or testing.
    /// Consume a local variable such that it appears to analyses that the variable is definitely
    /// used and thus cannot be removed. This is useful in testing to ensure that some variables
    /// are not optimised away.
    #[cfg(test)]
    BlackBox(BlackBoxInst),
    /// This instruction does not produce a value itself: it is equivalent to the constant at
    /// `ConstIdx`.
    Const(ConstIdx),
    /// This instruction does not produce a value itself: it is equivalent to the value produced by
    /// `InstIdx`.
    Copy(InstIdx),
    /// This instruction has been permanently removed. Note: this must only be used if you are
    /// entirely sure that the value this instruction once produced is no longer used.
    Tombstone,

    // "Normal" IR instructions.
    BinOp(BinOpInst),
    Load(LoadInst),
    LookupGlobal(LookupGlobalInst),
    Param(ParamInst),
    Call(DirectCallInst),
    IndirectCall(IndirectCallIdx),
    PtrAdd(PtrAddInst),
    DynPtrAdd(DynPtrAddInst),
    Store(StoreInst),
    ICmp(ICmpInst),
    Guard(GuardInst),
    /// Marks the point where side-traces reenter the root trace.
    TraceHeaderStart,
    /// Marks the end of a header. The `bool` will be `true` if this is a [TraceKind::Connector]
    /// trace.
    TraceHeaderEnd(bool),
    /// Marks the place to loop back to at the end of the JITted code.
    TraceBodyStart,
    TraceBodyEnd,
    SidetraceEnd,
    Deopt(GuardInfoIdx),
    Return(u64),

    SExt(SExtInst),
    ZExt(ZExtInst),
    Trunc(TruncInst),
    Select(SelectInst),
    SIToFP(SIToFPInst),
    FPExt(FPExtInst),
    FCmp(FCmpInst),
    FPToSI(FPToSIInst),
    BitCast(BitCastInst),
    FNeg(FNegInst),
    DebugStr(DebugStrInst),
    IntToPtr(IntToPtrInst),
    PtrToInt(PtrToIntInst),
    UIToFP(UIToFPInst),
    ExtractValue(ExtractValueInst),
}

impl Inst {
    /// What is the numeric discriminant of this [Inst]? These are guaranteed to be consecutive
    /// integers from `0..Inst::COUNT`.
    pub(crate) fn discriminant(&self) -> usize {
        debug_assert!((InstDiscriminants::from(self) as usize) < Self::COUNT);
        InstDiscriminants::from(self) as usize
    }

    /// Returns the type of the value that the instruction produces (if any).
    pub(crate) fn def_type<'a>(&self, m: &'a Module) -> Option<&'a Ty> {
        let idx = self.tyidx(m);
        if idx != m.void_tyidx() {
            Some(m.type_(idx))
        } else {
            None
        }
    }

    /// Returns the type index of the value that the instruction produces, or [Ty::Void] if it does
    /// not produce a value.
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        match self {
            #[cfg(test)]
            Self::BlackBox(_) => m.void_tyidx(),
            Self::Const(x) => m.const_(*x).tyidx(m),
            Self::Copy(x) => m.inst_raw(*x).tyidx(m),
            Self::Tombstone => panic!(),

            Self::BinOp(x) => x.tyidx(m),
            Self::IndirectCall(idx) => {
                let inst = m.indirect_call(*idx);
                let ty = m.type_(inst.ftyidx);
                let Ty::Func(fty) = ty else { panic!() };
                fty.ret_tyidx()
            }
            Self::Load(li) => li.tyidx(),
            Self::LookupGlobal(..) => m.ptr_tyidx(),
            Self::Param(li) => li.tyidx(),
            Self::Call(ci) => m.func_type(ci.target()).ret_tyidx(),
            Self::PtrAdd(..) => m.ptr_tyidx(),
            Self::DynPtrAdd(..) => m.ptr_tyidx(),
            Self::Store(..) => m.void_tyidx(),
            Self::ICmp(_) => m.int1_tyidx(),
            Self::Guard(..) => m.void_tyidx(),
            Self::TraceHeaderStart => m.void_tyidx(),
            Self::TraceHeaderEnd(_) => m.void_tyidx(),
            Self::TraceBodyStart => m.void_tyidx(),
            Self::TraceBodyEnd => m.void_tyidx(),
            Self::SidetraceEnd => m.void_tyidx(),
            Self::Deopt(_) => m.void_tyidx(),
            Self::Return(_) => m.void_tyidx(),
            Self::SExt(si) => si.dest_tyidx(),
            Self::ZExt(si) => si.dest_tyidx(),
            Self::BitCast(i) => i.dest_tyidx(),
            Self::Trunc(t) => t.dest_tyidx(),
            Self::Select(s) => s.trueval(m).tyidx(m),
            Self::SIToFP(i) => i.dest_tyidx(),
            Self::FPExt(i) => i.dest_tyidx(),
            Self::FCmp(_) => m.int1_tyidx(),
            Self::FPToSI(i) => i.dest_tyidx(),
            Self::FNeg(i) => i.val(m).tyidx(m),
            Self::DebugStr(..) => m.void_tyidx(),
            Self::PtrToInt(i) => i.dest_tyidx(),
            Self::IntToPtr(_) => m.ptr_tyidx(),
            Self::UIToFP(i) => i.dest_tyidx(),
            Self::ExtractValue(i) => i.tyidx(),
        }
    }

    /// Does this instruction have a load side effect?
    pub(crate) fn has_load_effect(&self, m: &Module) -> bool {
        match self {
            Inst::Copy(x) => m.inst_raw(*x).has_load_effect(m),
            Inst::Load(_) => true,
            Inst::Call(_) | Inst::IndirectCall(_) => true,
            _ => false,
        }
    }

    /// Does this instruction have a store side effect?
    pub(crate) fn has_store_effect(&self, m: &Module) -> bool {
        match self {
            #[cfg(test)]
            Inst::BlackBox(_) => true,
            Inst::Copy(x) => m.inst_raw(*x).has_store_effect(m),
            Inst::Store(_) => true,
            Inst::Call(_) | Inst::IndirectCall(_) => true,
            _ => false,
        }
    }

    /// Is this instruction an "internal" trace instruction (e.g. `TraceHeaderStart` /
    /// `TraceHeaderEnd`)? Such instructions must not be removed, and should be thought of as
    /// similar to a barrier.
    pub(crate) fn is_internal_inst(&self) -> bool {
        matches!(
            self,
            Inst::TraceHeaderStart
                | Inst::TraceHeaderEnd(_)
                | Inst::TraceBodyStart
                | Inst::TraceBodyEnd
                | Inst::SidetraceEnd
                | Inst::Deopt(_)
                | Inst::Return(_)
                | Inst::Param(_)
                | Inst::DebugStr(..)
        )
    }

    /// Is this instruction a guard?
    ///
    /// This is a convenience function to allow an easy mirror of `is_internal_inst`.
    pub(crate) fn is_guard(&self) -> bool {
        matches!(self, Inst::Guard(_))
    }

    /// Apply the function `f` to each of this instruction's [Operand]s iff they are of type
    /// [Operand::Var]. When an instruction references more than one [Operand], the order of
    /// traversal is undefined.
    pub(crate) fn map_operand_vars<F>(&self, m: &Module, f: &mut F)
    where
        F: FnMut(InstIdx),
    {
        match self {
            #[cfg(test)]
            Inst::BlackBox(BlackBoxInst { op }) => {
                op.unpack(m).map_iidx(f);
            }
            Inst::Const(_) => (),
            Inst::Copy(_) => (),
            Inst::Tombstone => (),
            Inst::BinOp(BinOpInst { lhs, binop: _, rhs }) => {
                lhs.unpack(m).map_iidx(f);
                rhs.unpack(m).map_iidx(f);
            }
            Inst::Load(LoadInst { ptr: op, .. }) => op.unpack(m).map_iidx(f),
            Inst::LookupGlobal(_) => (),
            Inst::Param(_) => (),
            Inst::Call(x) => {
                for i in x.iter_args_idx() {
                    m.arg(i).map_iidx(f);
                }
            }
            Inst::IndirectCall(x) => {
                let ici = m.indirect_call(*x);
                let IndirectCallInst { target, .. } = ici;
                target.unpack(m).map_iidx(f);
                for i in ici.iter_args_idx() {
                    m.arg(i).map_iidx(f);
                }
            }
            Inst::PtrAdd(x) => {
                let ptr = x.ptr;
                ptr.unpack(m).map_iidx(f);
            }
            Inst::DynPtrAdd(x) => {
                let ptr = x.ptr;
                ptr.unpack(m).map_iidx(f);
                let num_elems = x.num_elems;
                num_elems.unpack(m).map_iidx(f);
            }
            Inst::Store(StoreInst { ptr: tgt, val, .. }) => {
                tgt.unpack(m).map_iidx(f);
                val.unpack(m).map_iidx(f);
            }
            Inst::ICmp(ICmpInst { lhs, pred: _, rhs }) => {
                lhs.unpack(m).map_iidx(f);
                rhs.unpack(m).map_iidx(f);
            }
            Inst::Guard(x @ GuardInst { cond, .. }) => {
                cond.unpack(m).map_iidx(f);
                for (_, pop) in x.guard_info(m).live_vars() {
                    pop.unpack(m).map_iidx(f);
                }
            }
            Inst::TraceHeaderStart => {
                for x in &m.trace_header_start {
                    x.unpack(m).map_iidx(f);
                }
            }
            Inst::TraceHeaderEnd(_) => {
                for x in &m.trace_header_end {
                    x.unpack(m).map_iidx(f);
                }
            }
            Inst::TraceBodyStart => {
                for x in &m.trace_body_start {
                    x.unpack(m).map_iidx(f);
                }
            }
            Inst::TraceBodyEnd => {
                for x in &m.trace_body_end {
                    x.unpack(m).map_iidx(f);
                }
            }
            Inst::SidetraceEnd => {
                for x in &m.trace_header_end {
                    x.unpack(m).map_iidx(f);
                }
            }
            Inst::Deopt(gidx) => {
                for (_, pop) in m.guard_info[usize::from(*gidx)].live_vars() {
                    pop.unpack(m).map_iidx(f);
                }
            }
            Inst::Return(_) => (),
            Inst::SExt(SExtInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::ZExt(ZExtInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::BitCast(BitCastInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::Trunc(TruncInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::Select(SelectInst {
                cond,
                trueval,
                falseval,
            }) => {
                cond.unpack(m).map_iidx(f);
                trueval.unpack(m).map_iidx(f);
                falseval.unpack(m).map_iidx(f);
            }
            Inst::SIToFP(SIToFPInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::FPExt(FPExtInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::FCmp(FCmpInst { lhs, pred: _, rhs }) => {
                lhs.unpack(m).map_iidx(f);
                rhs.unpack(m).map_iidx(f);
            }
            Inst::FPToSI(FPToSIInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::FNeg(FNegInst { val }) => val.unpack(m).map_iidx(f),
            Inst::DebugStr(..) => (),
            Inst::PtrToInt(PtrToIntInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::IntToPtr(IntToPtrInst { val }) => val.unpack(m).map_iidx(f),
            Inst::UIToFP(UIToFPInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::ExtractValue(ExtractValueInst { op, .. }) => op.unpack(m).map_iidx(f),
        }
    }

    /// Duplicate this [Inst] and, during that process, apply the function `f` to each of this
    /// instruction's [Operand]s iff they are of type [Operand::Var]. When an instruction
    /// references more than one [Operand], the order of traversal is undefined.
    ///
    /// Note: for `TraceBody*` and `TraceHeader*` functions, this function is not idempotent.
    pub(crate) fn dup_and_remap_vars<F>(
        &self,
        m: &mut Module,
        f: F,
    ) -> Result<Self, CompilationError>
    where
        F: Fn(&Module, InstIdx) -> Operand,
    {
        let mapper = |m: &Module, x: &PackedOperand| match x.unpack(m) {
            Operand::Var(iidx) => PackedOperand::new(&f(m, iidx)),
            Operand::Const(_) => *x,
        };
        let op_mapper = |m: &Module, x: &Operand| match x {
            Operand::Var(iidx) => f(m, *iidx),
            Operand::Const(c) => Operand::Const(*c),
        };
        let inst = match self {
            #[cfg(test)]
            Inst::BlackBox(BlackBoxInst { op }) => {
                Inst::BlackBox(BlackBoxInst { op: mapper(m, op) })
            }
            Inst::BinOp(BinOpInst { lhs, binop, rhs }) => Inst::BinOp(BinOpInst {
                lhs: mapper(m, lhs),
                binop: *binop,
                rhs: mapper(m, rhs),
            }),
            Inst::Call(dc) => {
                // Clone and map arguments.
                let args = dc
                    .iter_args_idx()
                    .map(|x| op_mapper(m, &m.arg(x)))
                    .collect::<Vec<_>>();
                let dc = DirectCallInst::new(m, dc.target, args, dc.idem_const)?;
                Inst::Call(dc)
            }
            Inst::IndirectCall(iidx) => {
                let ic = m.indirect_call(*iidx);
                // Clone and map arguments.
                let args = ic
                    .iter_args_idx()
                    .map(|x| op_mapper(m, &m.arg(x)))
                    .collect::<Vec<_>>();
                let icnew = IndirectCallInst::new(m, ic.ftyidx, op_mapper(m, &ic.target(m)), args)?;
                let idx = m.push_indirect_call(icnew)?;
                Inst::IndirectCall(idx)
            }
            Inst::Const(c) => Inst::Const(*c),
            Inst::Copy(iidx) => match f(m, *iidx) {
                Operand::Var(iidx) => Inst::Copy(iidx),
                Operand::Const(cidx) => Inst::Const(cidx),
            },
            Inst::DynPtrAdd(inst) => {
                let ptr = inst.ptr;
                let num_elems = inst.num_elems;
                Inst::DynPtrAdd(DynPtrAddInst {
                    ptr: mapper(m, &ptr),
                    num_elems: mapper(m, &num_elems),
                    elem_size: inst.elem_size,
                })
            }
            Inst::FPToSI(FPToSIInst { val, dest_tyidx }) => Inst::FPToSI(FPToSIInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::FPExt(FPExtInst { val, dest_tyidx }) => Inst::FPExt(FPExtInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::FCmp(FCmpInst { lhs, pred, rhs }) => Inst::FCmp(FCmpInst {
                lhs: mapper(m, lhs),
                pred: *pred,
                rhs: mapper(m, rhs),
            }),
            Inst::Guard(GuardInst { cond, expect, gidx }) => {
                let ginfo = &m.guard_info[usize::from(*gidx)];
                let newlives = ginfo
                    .live_vars()
                    .iter()
                    .map(|(aot, jit)| (aot.clone(), mapper(m, jit)))
                    .collect();
                let inlined_frames = ginfo
                    .inlined_frames()
                    .iter()
                    .map(|x| InlinedFrame::new(x.callinst.clone(), x.funcidx, x.safepoint))
                    .collect::<Vec<_>>();
                let newginfo =
                    GuardInfo::new(*ginfo.bid(), newlives, inlined_frames, ginfo.safepoint_id);
                let newgidx = m.push_guardinfo(newginfo).unwrap();
                Inst::Guard(GuardInst {
                    cond: mapper(m, cond),
                    expect: *expect,
                    gidx: newgidx,
                })
            }
            Inst::ICmp(ICmpInst { lhs, pred, rhs }) => Inst::ICmp(ICmpInst {
                lhs: mapper(m, lhs),
                pred: *pred,
                rhs: mapper(m, rhs),
            }),
            Inst::Load(LoadInst {
                ptr,
                tyidx,
                volatile,
            }) => Inst::Load(LoadInst {
                ptr: mapper(m, ptr),
                tyidx: *tyidx,
                volatile: *volatile,
            }),
            Inst::Param(x) => Inst::Param(*x),
            Inst::LookupGlobal(g) => Inst::LookupGlobal(*g),
            Inst::PtrAdd(inst) => {
                let ptr = inst.ptr;
                Inst::PtrAdd(PtrAddInst {
                    ptr: mapper(m, &ptr),
                    off: inst.off,
                })
            }
            Inst::Select(SelectInst {
                cond,
                trueval,
                falseval,
            }) => Inst::Select(SelectInst {
                cond: mapper(m, cond),
                trueval: mapper(m, trueval),
                falseval: mapper(m, falseval),
            }),
            Inst::SExt(SExtInst { val, dest_tyidx }) => Inst::SExt(SExtInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::SIToFP(SIToFPInst { val, dest_tyidx }) => Inst::SIToFP(SIToFPInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::Store(StoreInst { ptr, val, volatile }) => Inst::Store(StoreInst {
                ptr: mapper(m, ptr),
                val: mapper(m, val),
                volatile: *volatile,
            }),
            Inst::Tombstone => Inst::Tombstone,
            Inst::TraceBodyStart => {
                m.trace_body_start = m.trace_body_start.iter().map(|op| mapper(m, op)).collect();
                Inst::TraceBodyStart
            }
            Inst::TraceBodyEnd => {
                m.trace_body_end = m.trace_body_end.iter().map(|op| mapper(m, op)).collect();
                Inst::TraceBodyEnd
            }
            Inst::TraceHeaderEnd(is_connector) => {
                // Copy the header label into the body while remapping the operands.
                m.trace_header_end = m.trace_header_end.iter().map(|op| mapper(m, op)).collect();
                Inst::TraceHeaderEnd(*is_connector)
            }
            Inst::SidetraceEnd => {
                // Copy the header label into the body while remapping the operands.
                m.trace_header_end = m.trace_header_end.iter().map(|op| mapper(m, op)).collect();
                Inst::SidetraceEnd
            }
            Inst::Deopt(gidx) => {
                let ginfo = &m.guard_info[usize::from(*gidx)];
                let newlives = ginfo
                    .live_vars()
                    .iter()
                    .map(|(aot, jit)| (aot.clone(), mapper(m, jit)))
                    .collect();
                let inlined_frames = ginfo
                    .inlined_frames()
                    .iter()
                    .map(|x| InlinedFrame::new(x.callinst.clone(), x.funcidx, x.safepoint))
                    .collect::<Vec<_>>();
                let newginfo =
                    GuardInfo::new(*ginfo.bid(), newlives, inlined_frames, ginfo.safepoint_id);
                let newgidx = m.push_guardinfo(newginfo).unwrap();
                Inst::Deopt(newgidx)
            }
            Inst::Return(id) => Inst::Return(*id),
            Inst::Trunc(TruncInst { val, dest_tyidx }) => Inst::Trunc(TruncInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::ZExt(ZExtInst { val, dest_tyidx }) => Inst::ZExt(ZExtInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::DebugStr(DebugStrInst { idx }) => Inst::DebugStr(DebugStrInst { idx: *idx }),
            Inst::BitCast(BitCastInst { val, dest_tyidx }) => Inst::BitCast(BitCastInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::FNeg(FNegInst { val }) => Inst::FNeg(FNegInst {
                val: mapper(m, val),
            }),
            Inst::PtrToInt(PtrToIntInst { val, dest_tyidx }) => Inst::PtrToInt(PtrToIntInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::IntToPtr(IntToPtrInst { val }) => Inst::IntToPtr(IntToPtrInst {
                val: mapper(m, val),
            }),
            Inst::TraceHeaderStart => todo!(),
            Inst::UIToFP(UIToFPInst { val, dest_tyidx }) => Inst::UIToFP(UIToFPInst {
                val: mapper(m, val),
                dest_tyidx: *dest_tyidx,
            }),
            Inst::ExtractValue(ExtractValueInst { op, tyidx, indices }) => {
                Inst::ExtractValue(ExtractValueInst {
                    op: mapper(m, op),
                    tyidx: *tyidx,
                    indices: *indices,
                })
            }
        };
        Ok(inst)
    }

    /// Returns the size in bytes of the local variable that this instruction defines (if any).
    ///
    /// # Panics
    ///
    /// Panics if:
    ///  - The instruction defines no local variable.
    ///  - The instruction defines an unsized local variable.
    pub(crate) fn def_byte_size(&self, m: &Module) -> usize {
        if let Some(ty) = self.def_type(m) {
            if let Some(size) = ty.byte_size() {
                size
            } else {
                panic!()
            }
        } else {
            panic!()
        }
    }

    /// Returns the bit width of the local variable that this instruction defines (if any).
    ///
    /// # Panics
    ///
    /// Panics if:
    ///  - The instruction defines no local variable.
    ///  - The instruction defines an unsized local variable.
    pub(crate) fn def_bitw(&self, m: &Module) -> u32 {
        if let Some(ty) = self.def_type(m) {
            if let Some(size) = ty.bitw() {
                size
            } else {
                panic!()
            }
        } else {
            panic!()
        }
    }

    /// Is this instruction "equal" to `other` modulo `Copy` instructions? For example, given:
    ///
    /// ```text
    /// %0: ...
    /// %1: add %0, %0
    /// %2: add %0, %0
    /// %3: %2
    /// %4: add %0, %1
    /// ```
    ///
    /// `%1`, `%2`, and `%3` are all "decopy equal" to each other, but `%4` is not "decopy equal"
    /// to any other instruction.
    ///
    /// # Panics
    ///
    /// If either `self` or `other` are `Inst::Copy`.
    pub(crate) fn decopy_eq(&self, m: &Module, other: Inst) -> bool {
        assert!(!matches!(self, Inst::Copy(..)));
        assert!(!matches!(other, Inst::Copy(..)));
        if std::mem::discriminant(self) != std::mem::discriminant(&other) {
            return false;
        }
        match (*self, other) {
            #[cfg(test)]
            (Self::BlackBox(x), Self::BlackBox(y)) => x.decopy_eq(m, y),
            (Self::Const(x), Self::Const(y)) => x == y,
            (Self::BinOp(x), Self::BinOp(y)) => x.decopy_eq(m, y),
            (Self::Load(x), Self::Load(y)) => x.decopy_eq(m, y),
            (Self::Store(x), Self::Store(y)) => x.decopy_eq(m, y),
            (Self::LookupGlobal(x), Self::LookupGlobal(y)) => x.decopy_eq(y),
            (Self::Param(x), Self::Param(y)) => x.decopy_eq(y),
            (Self::PtrAdd(x), Self::PtrAdd(y)) => x.decopy_eq(m, y),
            (Self::DynPtrAdd(x), Self::DynPtrAdd(y)) => x.decopy_eq(m, y),
            (Self::ICmp(x), Self::ICmp(y)) => x.decopy_eq(m, y),
            (Self::Guard(x), Self::Guard(y)) => x.decopy_eq(m, y),
            (Self::SExt(x), Self::SExt(y)) => x.decopy_eq(m, y),
            (Self::ZExt(x), Self::ZExt(y)) => x.decopy_eq(m, y),
            (Self::BitCast(x), Self::BitCast(y)) => x.decopy_eq(m, y),
            (Self::Trunc(x), Self::Trunc(y)) => x.decopy_eq(m, y),
            (Self::Select(x), Self::Select(y)) => x.decopy_eq(m, y),
            (Self::SIToFP(x), Self::SIToFP(y)) => x.decopy_eq(m, y),
            (Self::FPExt(x), Self::FPExt(y)) => x.decopy_eq(m, y),
            (Self::FCmp(x), Self::FCmp(y)) => x.decopy_eq(m, y),
            (Self::FPToSI(x), Self::FPToSI(y)) => x.decopy_eq(m, y),
            (Self::FNeg(x), Self::FNeg(y)) => x.decopy_eq(m, y),
            (Self::PtrToInt(x), Self::PtrToInt(y)) => x.decopy_eq(m, y),
            (Self::IntToPtr(x), Self::IntToPtr(y)) => x.decopy_eq(m, y),
            (Self::UIToFP(x), Self::UIToFP(y)) => x.decopy_eq(m, y),
            (Self::ExtractValue(x), Self::ExtractValue(y)) => x.decopy_eq(m, y),
            (x, y) => todo!("{x:?} {y:?}"),
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module, iidx: InstIdx) -> DisplayableInst<'a> {
        DisplayableInst {
            inst: self,
            iidx,
            m,
        }
    }
}

pub(crate) struct DisplayableInst<'a> {
    inst: &'a Inst,
    iidx: InstIdx,
    m: &'a Module,
}

impl fmt::Display for DisplayableInst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dt) = self.inst.def_type(self.m) {
            write!(f, "%{}: {} = ", self.iidx, dt.display(self.m))?;
        }
        match self.inst {
            #[cfg(test)]
            Inst::BlackBox(x) => write!(f, "black_box {}", x.operand(self.m).display(self.m)),
            Inst::Const(_) | Inst::Copy(_) | Inst::Tombstone => unreachable!(),

            Inst::BinOp(BinOpInst { lhs, binop, rhs }) => write!(
                f,
                "{} {}, {}",
                binop,
                lhs.unpack(self.m).display(self.m),
                rhs.unpack(self.m).display(self.m)
            ),
            Inst::Load(x) => write!(f, "load {}", x.ptr(self.m).display(self.m)),
            Inst::LookupGlobal(x) => write!(
                f,
                "lookup_global @{}",
                self.m
                    .global_decl(x.global_decl_idx)
                    .name
                    .to_str()
                    .unwrap_or("<not valid UTF-8>")
            ),
            Inst::Call(x) => {
                let idem_const = if let Some(cidx) = x.idem_const {
                    &format!(" <idem_const {}>", self.m.const_(cidx).display(self.m))
                } else {
                    ""
                };
                write!(
                    f,
                    "call @{}({}){}",
                    self.m.func_decl(x.target).name(),
                    (0..x.num_args())
                        .map(|y| format!("{}", x.operand(self.m, y).display(self.m)))
                        .collect::<Vec<_>>()
                        .join(", "),
                    idem_const
                )
            }
            Inst::IndirectCall(x) => {
                let inst = &self.m.indirect_call(*x);
                write!(
                    f,
                    "icall {}({})",
                    inst.target.unpack(self.m).display(self.m),
                    (0..inst.num_args())
                        .map(|y| format!("{}", inst.operand(self.m, y).display(self.m)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Inst::PtrAdd(x) => {
                write!(f, "ptr_add {}, {}", x.ptr(self.m).display(self.m), x.off())
            }
            Inst::DynPtrAdd(x) => {
                write!(
                    f,
                    "dyn_ptr_add {}, {}, {}",
                    x.ptr(self.m).display(self.m),
                    x.num_elems(self.m).display(self.m),
                    x.elem_size()
                )
            }
            Inst::Store(x) => write!(
                f,
                "*{} = {}",
                x.ptr.unpack(self.m).display(self.m),
                x.val.unpack(self.m).display(self.m)
            ),
            Inst::ICmp(x) => write!(
                f,
                "{} {}, {}",
                x.pred,
                x.lhs(self.m).display(self.m),
                x.rhs(self.m).display(self.m)
            ),
            Inst::Guard(x @ GuardInst { cond, expect, gidx }) => {
                let gi = x.guard_info(self.m);
                let live_vars = gi
                    .live_vars()
                    .iter()
                    .map(|(x, y)| {
                        format!(
                            "{}:%{}_{}: {}",
                            usize::from(x.funcidx()),
                            usize::from(x.bbidx()),
                            usize::from(x.iidx()),
                            y.unpack(self.m).display(self.m),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                // The optimisation pass may have removed some guard instructions, so the `gidx`
                // stored in this instruction may no longer be valid. In fact, the codegen ignores
                // it entirely and uses the position in the vector as the guard index. In order to
                // not confuse ourselves when looking at the printed IR, we should do the same
                // here.
                let mut actual_gidx = 0;
                for (_, inst) in self.m.iter_skipping_insts() {
                    if let Inst::Guard(GuardInst {
                        cond: _,
                        expect: _,
                        gidx: other_gidx,
                    }) = inst
                    {
                        if *gidx == other_gidx {
                            break;
                        }
                        actual_gidx += 1;
                    }
                }
                write!(
                    f,
                    "guard {}, {}, [{live_vars}] ; trace_gid {actual_gidx} safepoint_id {}",
                    if *expect { "true" } else { "false" },
                    cond.unpack(self.m).display(self.m),
                    gi.safepoint_id
                )
            }
            Inst::Param(x) => {
                write!(f, "param {:?}", self.m.params[usize::from(x.paramidx())])
            }
            Inst::TraceHeaderStart => {
                // Just marks a location, so we format it to look like a label.
                write!(f, "header_start [")?;
                for var in &self.m.trace_header_start {
                    write!(f, "{}", var.unpack(self.m).display(self.m))?;
                    if var != self.m.trace_header_start.last().unwrap() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Inst::TraceHeaderEnd(is_connector) => {
                // Just marks a location, so we format it to look like a label.
                if *is_connector {
                    write!(f, "connector [")?;
                } else {
                    write!(f, "header_end [")?;
                }
                for var in &self.m.trace_header_end {
                    write!(f, "{}", var.unpack(self.m).display(self.m))?;
                    if var != self.m.trace_header_end.last().unwrap() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Inst::TraceBodyStart => {
                // Just marks a location, so we format it to look like a label.
                write!(f, "body_start [")?;
                for (i, var) in self.m.trace_body_start.iter().enumerate() {
                    write!(f, "{}", var.unpack(self.m).display(self.m))?;
                    if i + 1 < self.m.trace_body_start.len() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Inst::TraceBodyEnd => {
                // Just marks a location, so we format it to look like a label.
                write!(f, "body_end [")?;
                for (i, var) in self.m.trace_body_end.iter().enumerate() {
                    write!(f, "{}", var.unpack(self.m).display(self.m))?;
                    if i + 1 < self.m.trace_body_end.len() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Inst::SidetraceEnd => {
                write!(f, "sidetrace_end {:?} [", self.m.root_jump_ptr)?;
                for (i, var) in self.m.trace_header_end.iter().enumerate() {
                    write!(f, "{}", var.unpack(self.m).display(self.m))?;
                    if i + 1 < self.m.trace_header_end.len() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Inst::Deopt(gidx) => {
                let gi = &self.m.guard_info[usize::from(*gidx)];
                let live_vars = gi
                    .live_vars()
                    .iter()
                    .map(|(x, y)| {
                        format!(
                            "{}:%{}_{}: {}",
                            usize::from(x.funcidx()),
                            usize::from(x.bbidx()),
                            usize::from(x.iidx()),
                            y.unpack(self.m).display(self.m),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "deopt [{live_vars}]")
            }
            Inst::Return(id) => {
                write!(f, "return [safepoint: {id}]")
            }
            Inst::SExt(i) => {
                write!(f, "sext {}", i.val(self.m).display(self.m),)
            }
            Inst::ZExt(i) => {
                write!(f, "zext {}", i.val(self.m).display(self.m),)
            }
            Inst::BitCast(i) => {
                write!(f, "bitcast {}", i.val(self.m).display(self.m),)
            }
            Inst::Trunc(i) => {
                write!(f, "trunc {}", i.val(self.m).display(self.m))
            }
            Inst::Select(s) => write!(
                f,
                "{} ? {} : {}",
                s.cond(self.m).display(self.m),
                s.trueval(self.m).display(self.m),
                s.falseval(self.m).display(self.m)
            ),
            Inst::SIToFP(i) => write!(f, "si_to_fp {}", i.val(self.m).display(self.m)),
            Inst::FPExt(i) => write!(f, "fp_ext {}", i.val(self.m).display(self.m)),
            Inst::FCmp(x) => write!(
                f,
                "{} {}, {}",
                x.pred,
                x.lhs(self.m).display(self.m),
                x.rhs(self.m).display(self.m)
            ),
            Inst::FPToSI(i) => write!(f, "fp_to_si {}", i.val(self.m).display(self.m)),
            Inst::FNeg(i) => write!(f, "fneg {}", i.val(self.m).display(self.m)),
            Inst::DebugStr(i) => write!(f, "; debug_str: {}", i.msg(self.m)),
            Inst::PtrToInt(i) => {
                write!(f, "ptr_to_int {}", i.val(self.m).display(self.m),)
            }
            Inst::IntToPtr(i) => {
                write!(f, "int_to_ptr {}", i.val(self.m).display(self.m),)
            }
            Inst::UIToFP(i) => write!(f, "ui_to_fp {}", i.val(self.m).display(self.m)),
            Inst::ExtractValue(i) => {
                write!(
                    f,
                    "extractvalue {}, {}",
                    i.val(self.m).display(self.m),
                    i.indices
                )
            }
        }
    }
}

macro_rules! inst {
    ($discrim:ident, $inst_type:ident) => {
        impl From<$inst_type> for Inst {
            fn from(inst: $inst_type) -> Inst {
                Inst::$discrim(inst)
            }
        }
    };
}

inst!(BinOp, BinOpInst);
#[cfg(test)]
inst!(BlackBox, BlackBoxInst);
inst!(Load, LoadInst);
inst!(LookupGlobal, LookupGlobalInst);
inst!(Store, StoreInst);
inst!(Param, ParamInst);
inst!(Call, DirectCallInst);
inst!(PtrAdd, PtrAddInst);
inst!(DynPtrAdd, DynPtrAddInst);
inst!(ICmp, ICmpInst);
inst!(Guard, GuardInst);
inst!(SExt, SExtInst);
inst!(ZExt, ZExtInst);
inst!(Trunc, TruncInst);
inst!(Select, SelectInst);
inst!(SIToFP, SIToFPInst);
inst!(FPExt, FPExtInst);
inst!(FCmp, FCmpInst);
inst!(FPToSI, FPToSIInst);
inst!(BitCast, BitCastInst);
inst!(FNeg, FNegInst);
inst!(DebugStr, DebugStrInst);
inst!(PtrToInt, PtrToIntInst);
inst!(IntToPtr, IntToPtrInst);
inst!(UIToFP, UIToFPInst);
inst!(ExtractValue, ExtractValueInst);

/// The operands for a [Instruction::BinOp]
///
/// # Semantics
///
/// Performs a binary operation.
///
/// The naming convention used is based on infix notation, e.g. in `2 + 3`, "2" is the left-hand
/// side (`lhs`), "+" is the binary operator (`binop`), and "3" is the right-hand side (`rhs`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinOpInst {
    /// The left-hand side of the operation.
    pub(crate) lhs: PackedOperand,
    /// The operation to perform.
    binop: BinOp,
    /// The right-hand side of the operation.
    pub(crate) rhs: PackedOperand,
}

impl BinOpInst {
    pub(crate) fn new(lhs: Operand, binop: BinOp, rhs: Operand) -> Self {
        Self {
            lhs: PackedOperand::new(&lhs),
            binop,
            rhs: PackedOperand::new(&rhs),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.lhs(m) == other.lhs(m) && self.binop == other.binop && self.rhs(m) == other.rhs(m)
    }

    pub(crate) fn lhs(&self, m: &Module) -> Operand {
        self.lhs.unpack(m)
    }

    pub(crate) fn binop(&self) -> BinOp {
        self.binop
    }

    pub(crate) fn rhs(&self, m: &Module) -> Operand {
        self.rhs.unpack(m)
    }

    /// Returns the type index of the operands being added.
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        self.lhs.unpack(m).tyidx(m)
    }
}

/// This is a test-only instruction which "consumes" an operand in the sense of "make use of the
/// value". This is useful to make clear in a test that an operand is used at a certain point,
/// which prevents optimisations removing some or all of the things that relate to this operand.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub struct BlackBoxInst {
    op: PackedOperand,
}

#[cfg(test)]
impl BlackBoxInst {
    pub(crate) fn new(op: Operand) -> BlackBoxInst {
        Self {
            op: PackedOperand::new(&op),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.operand(m) == other.operand(m)
    }

    pub(crate) fn operand(&self, m: &Module) -> Operand {
        self.op.unpack(m)
    }
}

/// The operands for a [Inst::Load]
///
/// # Semantics
///
/// Loads a value from a given pointer operand.
///
#[derive(Clone, Copy, Debug)]
pub struct LoadInst {
    /// The pointer to load from.
    ptr: PackedOperand,
    /// The type of the pointee.
    tyidx: TyIdx,
    /// Is this load volatile?
    volatile: bool,
}

impl LoadInst {
    // FIXME: why do we need to provide a type index? Can't we get that from the operand?
    pub(crate) fn new(ptr: Operand, tyidx: TyIdx, volatile: bool) -> LoadInst {
        LoadInst {
            ptr: PackedOperand::new(&ptr),
            tyidx,
            volatile,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.ptr.unpack(m) == other.ptr.unpack(m)
            && self.tyidx == other.tyidx
            && self.volatile == other.volatile
    }

    /// Return the pointer operand.
    pub(crate) fn ptr(&self, m: &Module) -> Operand {
        self.ptr.unpack(m)
    }

    /// Returns the type index of the loaded value.
    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    /// Is this a volatile load?
    pub(crate) fn is_volatile(&self) -> bool {
        // FIXME: We don't yet record this in AOT?!
        false
    }
}

/// The `Param` instruction.
///
/// ## Semantics
///
/// Specifies the [yksmp::Location] of a run-time argument.
#[derive(Clone, Copy, Debug)]
pub struct ParamInst {
    /// The [ParamIdx] of the [yksmp::Location] parameter.
    pidx: ParamIdx,
    /// The type of the resulting local variable.
    tyidx: TyIdx,
}

impl ParamInst {
    pub(crate) fn new(pidx: ParamIdx, tyidx: TyIdx) -> ParamInst {
        Self { pidx, tyidx }
    }

    pub(crate) fn decopy_eq(&self, other: Self) -> bool {
        self.pidx == other.pidx && self.tyidx == other.tyidx
    }

    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    /// Return The [InstIdx] of the [yksmp::Location] parameter.
    pub(crate) fn paramidx(&self) -> ParamIdx {
        self.pidx
    }
}

/// The operands for a [Inst::LookupGlobal]
///
/// # Semantics
///
/// Loads a value from a given global variable.
///
/// FIXME: Codegenning this instruction leads to unoptimial code, since all this does is write a
/// constant pointer into a register only to immediately use that register in the following
/// instruction. We'd rather want to inline the constant into the next instruction. So instead of:
/// ```ignore
/// mov rax, 0x123abc
/// mov [rax], 0x1
/// ```
/// we would get
/// ```ignore
/// mov [0x123abc], 0x1
/// ```
/// However, this requires us to change the JIT IR to allow globals inside operands (we don't want
/// to implement a special global version for each instruction, e.g. LoadGlobal/StoreGlobal/etc).
/// The easiest way to do this is to make globals a subclass of constants, similarly to what LLVM
/// does.
#[derive(Clone, Copy, Debug)]
pub struct LookupGlobalInst {
    /// The pointer to load from.
    global_decl_idx: GlobalDeclIdx,
}

impl LookupGlobalInst {
    #[cfg(not(test))]
    pub(crate) fn new(global_decl_idx: GlobalDeclIdx) -> Result<Self, CompilationError> {
        Ok(Self { global_decl_idx })
    }

    #[cfg(test)]
    pub(crate) fn new(_global_decl_idx: GlobalDeclIdx) -> Result<Self, CompilationError> {
        panic!("Cannot lookup globals in cfg(test) as ykllvm will not have compiled this binary");
    }

    fn decopy_eq(&self, other: Self) -> bool {
        self.global_decl_idx == other.global_decl_idx
    }

    #[cfg(not(test))]
    pub(crate) fn decl<'a>(&self, m: &'a Module) -> &'a GlobalDecl {
        m.global_decl(self.global_decl_idx)
    }

    /// Returns the index of the global to lookup.
    #[cfg(not(test))]
    pub(crate) fn global_decl_idx(&self) -> GlobalDeclIdx {
        self.global_decl_idx
    }
}
///
/// The operands for a [Inst::IndirectCall]
///
/// # Semantics
///
/// Perform an indirect call to an external or AOT function.
#[derive(Clone, Copy, Debug)]
pub struct IndirectCallInst {
    /// The callee.
    target: PackedOperand,
    // Type of the target function.
    ftyidx: TyIdx,
    /// How many arguments in [Module::extra_args] is this call passing?
    num_args: u16,
    /// At what index do the contiguous operands in [Module::extra_args] start?
    args_idx: ArgsIdx,
}

impl IndirectCallInst {
    pub(crate) fn new(
        m: &mut Module,
        ftyidx: TyIdx,
        target: Operand,
        args: Vec<Operand>,
    ) -> Result<IndirectCallInst, CompilationError> {
        let num_args = u16::try_from(args.len()).map_err(|_| {
            CompilationError::LimitExceeded(format!(
                "{} arguments passed but at most {} can be handled",
                args.len(),
                u16::MAX
            ))
        })?;
        let args_idx = m.push_args(args)?;
        Ok(Self {
            target: PackedOperand::new(&target),
            ftyidx,
            num_args,
            args_idx,
        })
    }

    /// Return the [TyIdx] of the callee.
    pub(crate) fn ftyidx(&self) -> TyIdx {
        self.ftyidx
    }

    /// Return the callee [Operand].
    pub(crate) fn target(&self, m: &Module) -> Operand {
        self.target.unpack(m)
    }

    /// How many arguments is this call instruction passing?
    pub(crate) fn num_args(&self) -> usize {
        usize::from(self.num_args)
    }

    /// Return an iterator for each of this direct call instruction's [ArgsIdx].
    pub(crate) fn iter_args_idx(&self) -> impl Iterator<Item = ArgsIdx> {
        (usize::from(self.args_idx)..usize::from(self.args_idx) + usize::from(self.num_args))
            .map(|x| ArgsIdx::try_from(x).unwrap())
    }

    /// Fetch the operand at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the operand index is out of bounds.
    pub(crate) fn operand(&self, m: &Module, idx: usize) -> Operand {
        m.arg(self.args_idx.checked_add(idx).unwrap()).clone()
    }
}

/// The operands for a [Inst::Call]
///
/// # Semantics
///
/// Perform a call to an external or AOT function.
#[derive(Clone, Copy, Debug)]
#[repr(Rust, packed)]
pub struct DirectCallInst {
    /// The callee.
    target: FuncDeclIdx,
    /// At what index do the contiguous operands in [Module::args] start?
    args_idx: ArgsIdx,
    /// How many arguments in [Module::args] is this call passing?
    num_args: u16,
    /// If this is a call to an idempotent function, then this will contain the [ConstIdx] of the
    /// return value (dynamically captured during tracing).
    idem_const: Option<ConstIdx>,
}

impl DirectCallInst {
    pub(crate) fn new(
        m: &mut Module,
        target: FuncDeclIdx,
        args: Vec<Operand>,
        idem_const: Option<ConstIdx>,
    ) -> Result<DirectCallInst, CompilationError> {
        let num_args = u16::try_from(args.len()).map_err(|_| {
            CompilationError::LimitExceeded(format!(
                "{} arguments passed but at most {} can be handled",
                args.len(),
                u16::MAX
            ))
        })?;
        let args_idx = m.push_args(args)?;
        Ok(Self {
            target,
            args_idx,
            num_args,
            idem_const,
        })
    }

    /// Return the [FuncDeclIdx] of the callee.
    pub(crate) fn target(&self) -> FuncDeclIdx {
        self.target
    }

    /// How many arguments is this call instruction passing?
    pub(crate) fn num_args(&self) -> usize {
        usize::from(self.num_args)
    }

    /// Return an iterator for each of this direct call instruction's [ArgsIdx].
    pub(crate) fn iter_args_idx(&self) -> impl Iterator<Item = ArgsIdx> {
        (usize::from(self.args_idx)..usize::from(self.args_idx) + usize::from(self.num_args))
            .map(|x| ArgsIdx::try_from(x).unwrap())
    }

    /// Fetch the operand at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the operand index is out of bounds.
    pub(crate) fn operand(&self, m: &Module, idx: usize) -> Operand {
        m.arg(ArgsIdx::try_from(usize::from(self.args_idx) + idx).unwrap())
    }

    /// Return this function call's idempotent constant, if one exists.
    pub(crate) fn idem_const(&self) -> Option<ConstIdx> {
        self.idem_const
    }
}

/// The operands for a [Inst::Store]
///
/// # Semantics
///
/// Stores a value into a pointer.
#[derive(Clone, Copy, Debug)]
pub struct StoreInst {
    /// The target pointer that we will store `val` into.
    ptr: PackedOperand,
    /// The value to store.
    val: PackedOperand,
    /// Is this store volatile?
    volatile: bool,
}

impl StoreInst {
    pub(crate) fn new(ptr: Operand, val: Operand, volatile: bool) -> Self {
        // FIXME: assert type of pointer
        Self {
            ptr: PackedOperand::new(&ptr),
            val: PackedOperand::new(&val),
            volatile,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.ptr(m) == other.ptr(m)
            && self.val(m) == other.val(m)
            && self.volatile == other.volatile
    }

    /// Returns the value operand: i.e. the thing that is going to be stored.
    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    /// Returns the target pointer: i.e. where to store [self.val()].
    pub(crate) fn ptr(&self, m: &Module) -> Operand {
        self.ptr.unpack(m)
    }

    pub(crate) fn is_volatile(&self) -> bool {
        self.volatile
    }
}

/// The operands for a [Inst::Select]
///
/// # Semantics
///
/// Selects from two values depending on a condition.
#[derive(Clone, Copy, Debug)]
pub struct SelectInst {
    cond: PackedOperand,
    trueval: PackedOperand,
    falseval: PackedOperand,
}

impl SelectInst {
    pub(crate) fn new(cond: Operand, trueval: Operand, falseval: Operand) -> Self {
        Self {
            cond: PackedOperand::new(&cond),
            trueval: PackedOperand::new(&trueval),
            falseval: PackedOperand::new(&falseval),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.cond(m) == other.cond(m)
            && self.trueval(m) == other.trueval(m)
            && self.falseval(m) == other.falseval(m)
    }

    /// Returns the condition.
    pub(crate) fn cond(&self, m: &Module) -> Operand {
        self.cond.unpack(m)
    }

    /// Returns the value for when the condition is true.
    pub(crate) fn trueval(&self, m: &Module) -> Operand {
        self.trueval.unpack(m)
    }

    /// Returns the value for when the condition is false.
    pub(crate) fn falseval(&self, m: &Module) -> Operand {
        self.falseval.unpack(m)
    }
}

/// The operands for a [Inst::FNeg]
///
/// # Semantics
///
/// Negates the floating point value.
#[derive(Clone, Copy, Debug)]
pub struct FNegInst {
    val: PackedOperand,
}

impl FNegInst {
    pub(crate) fn new(val: Operand) -> Self {
        Self {
            val: PackedOperand::new(&val),
        }
    }

    /// Returns the value to be negated.
    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m)
    }
}

/// The operands for a [Inst::DebugStr]
///
/// # Semantics
///
/// In terms of program semantics, this instruction is a no-op. It serves only to allow the
/// insertion of debugging strings into the JIT IR when displayed.
#[derive(Clone, Copy, Debug)]
pub struct DebugStrInst {
    idx: DebugStrIdx,
}

impl DebugStrInst {
    pub(crate) fn new(idx: DebugStrIdx) -> Self {
        Self { idx }
    }

    /// Returns the message.
    pub(crate) fn msg<'a>(&self, m: &'a Module) -> &'a str {
        &m.debug_strs[usize::from(self.idx)]
    }
}

/// The operands for a [Inst::PtrToInt]
///
/// # Semantics
///
/// Converts a pointer to an integer, zero extending or truncating as necessary if the destination
/// integer is not exactly pointer-sized.
#[derive(Clone, Copy, Debug)]
pub struct PtrToIntInst {
    /// The pointer operand to convert.
    val: PackedOperand,
    /// The integer type to convert the pointer to.
    dest_tyidx: TyIdx,
}

impl PtrToIntInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m)
    }
}

/// The operands for a [Inst::IntToPtr]
///
/// # Semantics
///
/// Converts a integer to a pointer, zero extending or truncating as necessary if the source
/// integer is not exactly pointer-sized.
#[derive(Clone, Copy, Debug)]
pub struct IntToPtrInst {
    /// The value to convert to a pointer.
    val: PackedOperand,
}

impl IntToPtrInst {
    pub(crate) fn new(val: &Operand) -> Self {
        Self {
            val: PackedOperand::new(val),
        }
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m)
    }
}

/// The operands for a [Inst::UIToFP].
///
/// # Semantics
///
/// Converts an unsigned integer to a floating point value to an unsigned integer.
#[derive(Clone, Copy, Debug)]
pub struct UIToFPInst {
    /// The unsigned integer to convert from.
    val: PackedOperand,
    /// The type to convert to. Must be a floating point type.
    dest_tyidx: TyIdx,
}

impl UIToFPInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m)
    }
}

/// An instruction that adds a constant offset to a pointer.
///
/// # Semantics
///
/// Returns a pointer value that is the result of adding the specified, signed, constant (byte)
/// offset to the input pointer operand.
///
/// Following LLVM semantics, the operation is permitted to silently wrap if the result doesn't fit
/// in the LLVM pointer indexing type.
#[derive(Clone, Copy, Debug)]
#[repr(Rust, packed)]
pub struct PtrAddInst {
    /// The pointer to offset
    ptr: PackedOperand,
    /// The constant byte offset.
    ///
    /// Depending upon the platform, LLVM `getelementptr` may allow larger offsets than what this
    /// field can express. Due to space constraints we only accept a reduced range of values and
    /// traces requiring values outside of this range cannot be JIT compiled.
    off: i32,
}

impl PtrAddInst {
    pub(crate) fn new(ptr: Operand, off: i32) -> Self {
        Self {
            ptr: PackedOperand::new(&ptr),
            off,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.ptr(m) == other.ptr(m) && self.off == other.off
    }

    pub(crate) fn ptr(&self, m: &Module) -> Operand {
        let ptr = self.ptr;
        ptr.unpack(m)
    }

    pub(crate) fn off(&self) -> i32 {
        self.off
    }
}

/// An instruction that adds a (potentially) dynamic offset to a pointer.
///
/// The dynamic value is computed from an element size and a number of elements.
///
/// # Semantics
///
/// Returns a pointer value that is the result of:
///  - multiplying the constant element size by the (potentially) dynamic number of elements, and
///  - adding the result to the specified pointer.
///
/// Following LLVM semantics, the operation is permitted to silently wrap if the result doesn't fit
/// in the LLVM pointer indexing type.
#[derive(Clone, Copy, Debug)]
#[repr(Rust, packed)]
pub struct DynPtrAddInst {
    /// The pointer to offset
    ptr: PackedOperand,
    /// The (dynamic) number of elements.
    num_elems: PackedOperand,
    /// The element size.
    ///
    /// Depending upon the platform, LLVM `getelementptr` may allow larger element sizes than what
    /// this field can express. Due to space constraints we only accept a reduced range of values
    /// and traces requiring values outside of this range cannot be JIT compiled.
    elem_size: u16,
}

impl DynPtrAddInst {
    pub(crate) fn new(ptr: Operand, num_elems: Operand, elem_size: u16) -> Self {
        Self {
            ptr: PackedOperand::new(&ptr),
            elem_size,
            num_elems: PackedOperand::new(&num_elems),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.ptr(m) == other.ptr(m)
            && self.num_elems(m) == other.num_elems(m)
            && self.elem_size() == other.elem_size()
    }

    pub(crate) fn ptr(&self, m: &Module) -> Operand {
        let ptr = self.ptr;
        ptr.unpack(m)
    }

    pub(crate) fn elem_size(&self) -> u16 {
        self.elem_size
    }

    pub(crate) fn num_elems(&self, m: &Module) -> Operand {
        let num_elems = self.num_elems;
        num_elems.unpack(m)
    }
}

/// The operands for a [Inst::ICmp]
///
/// # Semantics
///
/// Compares two integer operands according to a predicate (e.g. greater-than). Defines a local
/// variable that dictates the truth of the comparison.
///
#[derive(Clone, Copy, Debug)]
pub struct ICmpInst {
    pub(crate) lhs: PackedOperand,
    pub(crate) pred: Predicate,
    pub(crate) rhs: PackedOperand,
}

impl ICmpInst {
    pub(crate) fn new(lhs: Operand, pred: Predicate, rhs: Operand) -> Self {
        Self {
            lhs: PackedOperand::new(&lhs),
            pred,
            rhs: PackedOperand::new(&rhs),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.lhs(m) == other.lhs(m) && self.pred == other.pred && self.rhs(m) == other.rhs(m)
    }

    /// Returns the left-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `x`.
    pub(crate) fn lhs(&self, m: &Module) -> Operand {
        self.lhs.unpack(m)
    }

    /// Returns the right-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `y`.
    pub(crate) fn rhs(&self, m: &Module) -> Operand {
        self.rhs.unpack(m)
    }

    /// Returns the predicate of the comparison.
    ///
    /// E.g. in `x <= y`, it's `<=`.
    pub(crate) fn predicate(&self) -> Predicate {
        self.pred
    }
}

/// The operands for a [Inst::FCmp]
///
/// # Semantics
///
/// Compares two floating point operands according to a predicate (e.g. greater-than). Defines a
/// local variable that dictates the truth of the comparison.
#[derive(Clone, Copy, Debug)]
pub struct FCmpInst {
    pub(crate) lhs: PackedOperand,
    pub(crate) pred: FloatPredicate,
    pub(crate) rhs: PackedOperand,
}

impl FCmpInst {
    pub(crate) fn new(lhs: Operand, pred: FloatPredicate, rhs: Operand) -> Self {
        Self {
            lhs: PackedOperand::new(&lhs),
            pred,
            rhs: PackedOperand::new(&rhs),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.lhs(m) == other.lhs(m) && self.pred == other.pred && self.rhs(m) == other.rhs(m)
    }

    /// Returns the left-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `x`.
    pub(crate) fn lhs(&self, m: &Module) -> Operand {
        self.lhs.unpack(m)
    }

    /// Returns the right-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `y`.
    pub(crate) fn rhs(&self, m: &Module) -> Operand {
        self.rhs.unpack(m)
    }

    /// Returns the predicate of the comparison.
    ///
    /// E.g. in `x <= y`, it's `<=`.
    pub(crate) fn predicate(&self) -> FloatPredicate {
        self.pred
    }
}

/// The operands for a [Inst::Guard]
///
/// # Semantics
///
/// Guards a trace against diverging execution. The remainder of the trace will be compiled under
/// the assumption that (at runtime) the guard condition is true. If the guard condition is false,
/// then execution may not continue, and deoptimisation must occur.
///
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuardInst {
    /// The condition to guard against.
    pub(crate) cond: PackedOperand,
    /// The expected outcome of the condition.
    pub(crate) expect: bool,
    /// Additional information about this guard.
    pub(crate) gidx: GuardInfoIdx,
}

impl GuardInst {
    pub(crate) fn new(cond: Operand, expect: bool, gidx: GuardInfoIdx) -> Self {
        GuardInst {
            cond: PackedOperand::new(&cond),
            expect,
            gidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: GuardInst) -> bool {
        self.cond.unpack(m) == other.cond.unpack(m)
            && self.expect == other.expect
            && self.gidx == other.gidx
    }

    pub(crate) fn cond(&self, m: &Module) -> Operand {
        self.cond.unpack(m)
    }

    pub(crate) fn expect(&self) -> bool {
        self.expect
    }

    pub(crate) fn guard_info<'a>(&self, m: &'a Module) -> &'a GuardInfo {
        &m.guard_info[usize::from(self.gidx)]
    }
}

/// Container for things that provide information required for deopt.
#[derive(Debug)]
pub(crate) enum HasGuardInfo {
    Guard(GuardInst),
    Deopt(GuardInfoIdx),
}

impl HasGuardInfo {
    pub(crate) fn guard_info<'a>(&self, m: &'a Module) -> &'a GuardInfo {
        match self {
            Self::Guard(inst) => inst.guard_info(m),
            Self::Deopt(gidx) => m.guard_info(*gidx),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SExtInst {
    /// The value to extend.
    val: PackedOperand,
    /// The type to extend to.
    dest_tyidx: TyIdx,
}

impl SExtInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BitCastInst {
    /// The value to extend.
    val: PackedOperand,
    /// The type to extend to.
    dest_tyidx: TyIdx,
}

impl BitCastInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ZExtInst {
    /// The value to extend.
    val: PackedOperand,
    /// The type to extend to.
    dest_tyidx: TyIdx,
}

impl ZExtInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TruncInst {
    /// The value to extend.
    val: PackedOperand,
    /// The type to extend to.
    dest_tyidx: TyIdx,
}

impl TruncInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SIToFPInst {
    /// The value to convert.
    val: PackedOperand,
    /// The type to convert to. Must be a floating point type.
    dest_tyidx: TyIdx,
}

impl SIToFPInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

/// The operands for a [Inst::FPToSI]
///
/// # Semantics
///
/// This instruction converts a floating point value to an integer value, rounding towards zero.
///
/// The source float and destination integer need not be the same width, but if the resulting
/// numeric value does not fit, the result is undefined.
#[derive(Clone, Copy, Debug)]
pub struct FPToSIInst {
    /// The value to convert. Must be of floating point type.
    val: PackedOperand,
    /// The type to convert to. Must be an integer type.
    dest_tyidx: TyIdx,
}

impl FPToSIInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FPExtInst {
    /// The value to convert.
    val: PackedOperand,
    /// The type to convert to. Must be a larger floating point type.
    dest_tyidx: TyIdx,
}

impl FPExtInst {
    pub(crate) fn new(val: &Operand, dest_tyidx: TyIdx) -> Self {
        Self {
            val: PackedOperand::new(val),
            dest_tyidx,
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.dest_tyidx == other.dest_tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

/// The operands for a [Inst::ExtractValue]
///
/// # Semantics
///
/// Extracts the value of a member field of an aggregate value, like a struct or an array.
#[derive(Clone, Copy, Debug)]
pub struct ExtractValueInst {
    // The source from which to extract from.
    op: PackedOperand,
    // The type of the source.
    tyidx: TyIdx,
    // The index from which to extract.
    // FIXME: This can be list of indices and thus should be a vec.
    indices: u8,
}

impl ExtractValueInst {
    pub(crate) fn new(op: Operand, tyidx: TyIdx, indices: Vec<usize>) -> Self {
        // We currently only deal with one index so we don't need to store a vec.
        assert!(indices.len() == 1);
        Self {
            op: PackedOperand::new(&op),
            tyidx,
            indices: u8::try_from(indices[0]).unwrap(),
        }
    }

    fn decopy_eq(&self, m: &Module, other: Self) -> bool {
        self.val(m) == other.val(m) && self.tyidx == other.tyidx
    }

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.op.unpack(m)
    }

    /// Returns the type index of the aggregate value.
    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    /// Returns the index of the member field.
    pub(crate) fn index(&self) -> u8 {
        self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn use_case_update_inst() {
        let mut prog: Vec<Inst> = vec![
            ParamInst::new(ParamIdx::try_from(0).unwrap(), TyIdx::try_from(0).unwrap()).into(),
            ParamInst::new(ParamIdx::try_from(8).unwrap(), TyIdx::try_from(0).unwrap()).into(),
            LoadInst::new(
                Operand::Var(InstIdx(0)),
                TyIdx(U24::try_from(0).unwrap()),
                false,
            )
            .into(),
        ];
        prog[2] = LoadInst::new(
            Operand::Var(InstIdx(1)),
            TyIdx(U24::try_from(0).unwrap()),
            false,
        )
        .into();
    }

    /// Ensure that any given instruction fits in 128-bits.
    #[test]
    fn inst_size() {
        assert!(mem::size_of::<Inst>() <= 2 * mem::size_of::<u64>());
    }

    #[test]
    fn vararg_call_args() {
        // Set up a function to call.
        let mut m = Module::new_testing();
        let i32_tyidx = m.insert_ty(Ty::Integer(32)).unwrap();
        let func_ty = Ty::Func(FuncTy::new(vec![i32_tyidx; 3], i32_tyidx, true));
        let func_tyidx = m.insert_ty(func_ty).unwrap();
        let func_decl_idx = m
            .insert_func_decl(FuncDecl::new("foo".to_owned(), func_tyidx))
            .unwrap();

        // Build a call to the function.
        let args = vec![
            Operand::Var(InstIdx(0)),
            Operand::Var(InstIdx(1)),
            Operand::Var(InstIdx(2)),
        ];
        m.push(Inst::Param(ParamInst::new(
            ParamIdx::try_from(0).unwrap(),
            i32_tyidx,
        )))
        .unwrap();
        m.push(Inst::Param(ParamInst::new(
            ParamIdx::try_from(1).unwrap(),
            i32_tyidx,
        )))
        .unwrap();
        m.push(Inst::Param(ParamInst::new(
            ParamIdx::try_from(2).unwrap(),
            i32_tyidx,
        )))
        .unwrap();
        let ci = DirectCallInst::new(&mut m, func_decl_idx, args, None).unwrap();

        // Now request the operands and check they all look as they should.
        assert_eq!(ci.operand(&m, 0), Operand::Var(InstIdx(0)));
        assert_eq!(ci.operand(&m, 1), Operand::Var(InstIdx(1)));
        assert_eq!(ci.operand(&m, 2), Operand::Var(InstIdx(2)));
    }

    #[test]
    #[should_panic]
    fn call_args_out_of_bounds() {
        // Set up a function to call.
        let mut m = Module::new_testing();
        let arg_tyidxs = vec![m.ptr_tyidx(); 3];
        let ret_tyidx = m.insert_ty(Ty::Void).unwrap();
        let func_ty = FuncTy::new(arg_tyidxs, ret_tyidx, false);
        let func_tyidx = m.insert_ty(Ty::Func(func_ty)).unwrap();
        let func_decl_idx = m
            .insert_func_decl(FuncDecl::new("blah".into(), func_tyidx))
            .unwrap();

        // Now build a call to the function.
        let args = vec![
            Operand::Var(InstIdx(0)),
            Operand::Var(InstIdx(1)),
            Operand::Var(InstIdx(2)),
        ];
        let ci = DirectCallInst::new(&mut m, func_decl_idx, args, None).unwrap();

        // Request an operand with an out-of-bounds index.
        ci.operand(&m, 3);
    }

    #[test]
    fn u24_from_usize() {
        assert_eq!(U24::try_from(0x000000), Ok(U24([0x00, 0x00, 0x00])));
        assert_eq!(U24::try_from(0x123456), Ok(U24([0x12, 0x34, 0x56])));
        assert_eq!(U24::try_from(0xffffff), Ok(U24([0xff, 0xff, 0xff])));
        assert!(U24::try_from(0x1000000).is_err());
        assert!(U24::try_from(0x1234567).is_err());
        assert!(U24::try_from(0xfffffff).is_err());
    }

    #[test]
    fn u24_to_usize() {
        assert_eq!(usize::from(U24([0x00, 0x00, 0x00])), 0x000000);
        assert_eq!(usize::from(U24([0x12, 0x34, 0x56])), 0x123456);
        assert_eq!(usize::from(U24([0xff, 0xff, 0xff])), 0xffffff);
    }

    #[test]
    fn u24_round_trip() {
        assert_eq!(usize::from(U24::try_from(0x000000).unwrap()), 0x000000);
        assert_eq!(usize::from(U24::try_from(0x123456).unwrap()), 0x123456);
        assert_eq!(usize::from(U24::try_from(0xffffff).unwrap()), 0xffffff);
    }

    #[test]
    fn index24_fits() {
        assert!(TyIdx::try_from(0).is_ok());
        assert!(TyIdx::try_from(1).is_ok());
        assert!(TyIdx::try_from(0x1234).is_ok());
        assert!(TyIdx::try_from(0x123456).is_ok());
        assert!(TyIdx::try_from(0xffffff).is_ok());
    }

    #[test]
    fn index24_doesnt_fit() {
        assert!(TyIdx::try_from(0x1000000).is_err());
        assert!(TyIdx::try_from(0x1234567).is_err());
        assert!(TyIdx::try_from(0xeddedde).is_err());
        assert!(TyIdx::try_from(usize::MAX).is_err());
    }

    #[test]
    fn index16_fits() {
        assert!(ArgsIdx::try_from(0).is_ok());
        assert!(ArgsIdx::try_from(1).is_ok());
        assert!(ArgsIdx::try_from(0x1234).is_ok());
        assert!(ArgsIdx::try_from(0xffff).is_ok());
    }

    #[test]
    fn index16_doesnt_fit() {
        assert!(ArgsIdx::try_from(0x10000).is_err());
        assert!(ArgsIdx::try_from(0x12345).is_err());
        assert!(ArgsIdx::try_from(0xffffff).is_err());
        assert!(ArgsIdx::try_from(usize::MAX).is_err());
    }

    #[test]
    fn void_type_size() {
        assert_eq!(Ty::Void.byte_size(), Some(0));
    }

    #[test]
    fn stringify_int_consts() {
        let mut m = Module::new_testing();
        let i64_tyidx = m.insert_ty(Ty::Integer(64)).unwrap();
        assert_eq!(
            Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 0))
                .display(&m)
                .to_string(),
            "0i8"
        );
        assert_eq!(
            Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 255))
                .display(&m)
                .to_string(),
            "255i8"
        );
        assert_eq!(
            Const::Int(i64_tyidx, ArbBitInt::from_u64(64, 0))
                .display(&m)
                .to_string(),
            "0i64"
        );
        assert_eq!(
            Const::Int(i64_tyidx, ArbBitInt::from_u64(64, 9223372036854775808))
                .display(&m)
                .to_string(),
            "9223372036854775808i64"
        );
    }

    #[test]
    fn stringify_const_ptr() {
        let m = Module::new_testing();
        let ptr_val = stringify_const_ptr as *const () as usize;
        let cp = Const::Ptr(ptr_val);
        assert_eq!(
            cp.display(&m).to_string(),
            format!("{:#x}", ptr_val as usize)
        );
    }

    #[test]
    fn print_module() {
        let mut m = Module::new_testing();
        m.push_param(yksmp::Location::Register(3, 1, smallvec![]));
        m.push(ParamInst::new(ParamIdx::try_from(0).unwrap(), m.int8_tyidx()).into())
            .unwrap();
        m.insert_global_decl(GlobalDecl::new(
            CString::new("some_global").unwrap(),
            false,
            aot_ir::GlobalDeclIdx::new(0),
        ))
        .unwrap();
        m.insert_global_decl(GlobalDecl::new(
            CString::new("some_thread_local").unwrap(),
            true,
            aot_ir::GlobalDeclIdx::new(1),
        ))
        .unwrap();
        let expect = [
            "; compiled trace ID #0",
            "",
            "global_decl @some_global",
            "global_decl tls @some_thread_local",
            "",
            "entry:",
            "  %0: i8 = param Register(3, 1, [])",
        ]
        .join("\n");
        assert_eq!(m.to_string(), expect);
    }

    #[test]
    fn integer_type_sizes() {
        for i in 1..8 {
            assert_eq!(Ty::Integer(i).byte_size().unwrap(), 1);
        }
        for i in 9..16 {
            assert_eq!(Ty::Integer(i).byte_size().unwrap(), 2);
        }
        assert_eq!(Ty::Integer(127).byte_size().unwrap(), 16);
        assert_eq!(Ty::Integer(128).byte_size().unwrap(), 16);
        assert_eq!(Ty::Integer(129).byte_size().unwrap(), 17);
    }

    #[test]
    fn iter_skipping_insts() {
        let m = Module::from_str(
            "
        entry:
          %0: i8 = param reg
          %1: i8 = param reg
          %2: i8 = param reg
          %3: i8 = param reg
          %4: i8 = param reg
          %5: i8 = param reg
        ",
        );
        let mut iter = m.iter_skipping_insts();
        assert_eq!(Some(0), iter.next().map(|x| usize::from(x.0)));
        assert_eq!(Some(5), iter.next_back().map(|x| usize::from(x.0)));
        assert_eq!(Some(4), iter.next_back().map(|x| usize::from(x.0)));
        assert_eq!(Some(1), iter.next().map(|x| usize::from(x.0)));
        assert_eq!(Some(2), iter.next().map(|x| usize::from(x.0)));
        assert_eq!(Some(3), iter.next().map(|x| usize::from(x.0)));
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
