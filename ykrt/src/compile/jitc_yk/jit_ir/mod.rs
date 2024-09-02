//! The JIT's Intermediate Representation (IR).
//!
//! This is the IR created by [trace_builder] and which is then analysed and optimised. The IR has
//! the following properties:
//!
//!   1. We use vectors rather than direct references. This allows us to squeeze the amount of the
//!      memory used down (and also bypasses issues with representing graph structures in Rust, but
//!      that's slightly accidental).
//!
//!   2. The IR is designed to allow common manipulations to be performed without requiring
//!      reallocation.
//!
//! This second property has the following consequences:
//!
//!   1. We introduce three "special" instructions which are stored as normal instructions, but
//!      which are hidden when one views the IR:
//!        * `ProxyConst`, which represents a constant. What, in the underlying IR, nominally looks
//!          as follows:
//!          ```text
//!          %7: i8 = 1i8
//!          %8: add %2, %7
//!          ```
//!          will be displayed *deconsted* as simply `%8: add %2, 1i8`. Note that this is
//!          deliberately indistinguishable from IR which directly stores `%8: add %2, 1i8`.
//!
//!        * `ProxyInst` which is a (possibly chained) proxy for another instruction. What, in the
//!          underlying IR, nominally looks as follows:
//!          ```text
//!          %7: %6
//!          %8: add %2, %7
//!          ```
//!          will be displayed *deproxied* as simply `%8: add %2, %6`. Note that this is
//!          deliberately indistinguishable from IR which directly stores `%8: add %2, %6`.
//!
//!        * `Tombstone` which represents an instruction which is no longer used. These are not
//!          displayed at all.
//!
//!   2. We never store [Operand]s directly: we only ever store [PackedOperand]s. Whenever we want
//!      to get an [Operand] we *must* use [PackedOperand::unpack]. `unpack` does the necessary
//!      deconsting/deproxying.
//!
//!   3. External users of the IR should expect to always view the IR as deconsted/deproxied.
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

mod dead_code;
#[cfg(test)]
mod parser;
#[cfg(any(debug_assertions, test))]
mod well_formed;

use super::aot_ir;
use crate::compile::CompilationError;
use indexmap::IndexSet;
use std::{
    ffi::{c_void, CString},
    fmt,
    hash::Hash,
    mem,
};
#[cfg(not(test))]
use ykaddr::addr::symbol_to_ptr;

// This is simple and can be shared across both IRs.
pub(crate) use super::aot_ir::{BinOp, FloatPredicate, FloatTy, Predicate};

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
    /// The ID of the compiled trace.
    ///
    /// See the [Self::ctr_id] method for details.
    ctr_id: u64,
    /// The IR trace as a linear sequence of instructions.
    insts: Vec<Inst>,
    /// The arguments pool for [CallInst]s. Indexed by [ArgsIdx].
    args: Vec<PackedOperand>,
    /// The constant pool. Indexed by [ConstIdx].
    consts: IndexSet<ConstIndexSetWrapper>,
    /// The type pool. Indexed by [TyIdx].
    types: IndexSet<Ty>,
    /// The trace-input location pool.
    tilocs: Vec<yksmp::Location>,
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
    /// Live variables at the beginning of the loop.
    loop_start_vars: Vec<Operand>,
    /// Live variables at the end of the loop.
    loop_jump_vars: Vec<Operand>,
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
}

impl Module {
    /// Create a new [Module].
    pub(crate) fn new(ctr_id: u64, global_decls_len: usize) -> Result<Self, CompilationError> {
        Self::new_internal(ctr_id, global_decls_len)
    }

    /// Returns the ID of the module.
    ///
    /// In `cfg(test)` the ID is meaningless: in `cfg(not(test))` the ID is obtained from
    /// [crate::mt::MT::next_compiled_trace_id] and can be used to semi-uniquely distinguish traces
    /// (see [crate::mt::MT::compiled_trace_id] for more details).
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn ctr_id(&self) -> u64 {
        self.ctr_id
    }

    #[cfg(test)]
    pub(crate) fn new_testing() -> Self {
        Self::new_internal(0, 0).unwrap()
    }

    pub(crate) fn new_internal(
        ctr_id: u64,
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
                .insert_full(ConstIndexSetWrapper(Const::Int(int1_tyidx, 1)))
                .0,
        )
        .unwrap();
        let false_constidx = ConstIdx::try_from(
            consts
                .insert_full(ConstIndexSetWrapper(Const::Int(int1_tyidx, 0)))
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
            ctr_id,
            insts: Vec::new(),
            args: Vec::new(),
            consts,
            types,
            tilocs: Vec::new(),
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
            loop_start_vars: Vec::new(),
            loop_jump_vars: Vec::new(),
            #[cfg(not(test))]
            globalvar_ptrs,
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
    /// If `iidx` points to a `Proxy*` instruction.
    pub(crate) fn inst_no_proxies(&self, iidx: InstIdx) -> Inst {
        match self.insts[usize::from(iidx)] {
            Inst::ProxyConst(_) | Inst::ProxyInst(_) => panic!(),
            x => x,
        }
    }

    /// Return the instruction at the specified index, deproxying `ProxyInst` i.e. searching
    /// until a non-`ProxyInst` instruction is found.
    pub(crate) fn inst_deproxy(&self, mut iidx: InstIdx) -> (InstIdx, Inst) {
        loop {
            match self.insts[usize::from(iidx)] {
                Inst::ProxyInst(proxy_iidx) => iidx = proxy_iidx,
                x => return (iidx, x),
            }
        }
    }

    /// Return the instruction at the specified index. Note: unless you are explicitly handling
    /// `Proxy*` instructions in your code you must use [Self::inst_no_proxies] -- not handling
    /// proxies correctly is undefined behaviour. If in doubt, use [Self::inst_no_proxies].
    fn inst_all(&self, iidx: InstIdx) -> &Inst {
        &self.insts[usize::from(iidx)]
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

    /// Iterate, in order, over all `InstIdx`s of this module (including `Proxy*` and `Tombstone`
    /// instructions).
    pub(crate) fn iter_all_inst_idxs(&self) -> impl DoubleEndedIterator<Item = InstIdx> {
        // The `unchecked_from` is safe because we know from `Self::push` that we can't have
        // exceeded `InstIdx`'s bounds.
        (0..self.insts.len()).map(InstIdx::unchecked_from)
    }

    /// An iterator over instructions that skips `Proxy*` and `Tombstone` instructions.
    ///
    /// This implicitly deduplicates the callers view of instructions (since `Proxy*` instructions
    /// are skipped), but note that the indices, while strictly monotonically increasing, may be
    /// non-consecutive (because of skipping).
    pub(crate) fn iter_skipping_insts(&self) -> SkippingInstsIterator<'_> {
        SkippingInstsIterator { m: self, cur: 0 }
    }

    /// Replace the instruction at `iidx` with `inst`.
    pub(crate) fn replace(&mut self, iidx: InstIdx, inst: Inst) {
        self.insts[usize::from(iidx)] = inst;
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
            Operand::Local(x)
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

    /// Return the raw [PackedOperand] at args index `idx`. Unless you have a good reason for doing
    /// so, use [Self::arg] instead.
    ///
    /// # Panics
    ///
    /// If `idx` is out of bounds.
    pub(crate) fn arg_packed(&self, idx: ArgsIdx) -> PackedOperand {
        self.args[usize::from(idx)]
    }

    /// Push the location of a trace input variable.
    pub(crate) fn push_tiloc(&mut self, loc: yksmp::Location) {
        self.tilocs.push(loc);
    }

    pub(crate) fn tilocs(&self) -> &[yksmp::Location] {
        &self.tilocs
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
    pub fn insert_const(&mut self, c: Const) -> Result<ConstIdx, CompilationError> {
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

    pub(crate) fn loop_start_vars(&self) -> &[Operand] {
        &self.loop_start_vars
    }

    pub(crate) fn push_loop_start_var(&mut self, op: Operand) {
        self.loop_start_vars.push(op);
    }

    pub(crate) fn loop_jump_vars(&self) -> &[Operand] {
        &self.loop_jump_vars
    }

    pub(crate) fn push_loop_jump_var(&mut self, op: Operand) {
        self.loop_jump_vars.push(op);
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
            write!(f, "\n    {}", inst.display(iidx, self))?
        }

        Ok(())
    }
}

/// An iterator over instructions that skips `Proxy*` and `Tombstone` instructions.
///
/// This implicitly deduplicates the callers view of instructions (since `Proxy*` instructions are
/// skipped), but note that the indices, while strictly monotonically increasing, may be
/// non-consecutive (because of skipping).
pub(crate) struct SkippingInstsIterator<'a> {
    m: &'a Module,
    cur: usize,
}

impl<'a> Iterator for SkippingInstsIterator<'a> {
    type Item = (InstIdx, &'a Inst);
    /// Return the next instruction index and its associated instruction or `None` if the end has
    /// been reached.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(x) = self.m.insts.get(self.cur) {
            // We know that `self.cur` must fit in `InstIdx`, as otherwise `m.insts` wouldn't have
            // had the instruction in the first place, so the `unchecked_from` is safe.
            let old = InstIdx::unchecked_from(self.cur);
            self.cur += 1;
            match x {
                Inst::ProxyConst(_) | Inst::ProxyInst(_) | Inst::Tombstone => (),
                _ => return Some((old, &self.m.insts[usize::from(old)])),
            }
        }
        None
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
const OPERAND_IDX_MASK: u16 = 0x7fff;

/// The largest operand index we can express in 15 bits.
const MAX_OPERAND_IDX: u16 = (1 << 15) - 1;

/// The symbol name of the global variable pointers array.
#[cfg(not(test))]
const GLOBAL_PTR_ARRAY_SYM: &str = "__yk_globalvar_ptrs";

/// A packed 24-bit unsigned integer.
#[repr(packed)]
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
    CompilationError::LimitExceeded(format!("index overflow: {}", typ))
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
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct ConstIdx(u16);
index_16bit!(ConstIdx);

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

/// An instruction index.
///
/// One of these is an index into the [Module::insts].
#[derive(Debug, Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct InstIdx(u16);
index_16bit!(InstIdx);

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

    /// Return the number of paramaters the function accepts (not including varargs).
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn num_params(&self) -> usize {
        self.param_tyidxs.len()
    }

    /// Return a slice of this function's non-varargs parameters.
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn param_tys(&self) -> &[TyIdx] {
        &self.param_tyidxs
    }

    /// Returns whether the function type has vararg arguments.
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
    Unimplemented(String),
}

impl Ty {
    /// Returns the size of the type in bytes, or `None` if asking the size makes no sense.
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
            Ty::Float(ft) => write!(f, "{}", ft),
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

/// The packed representation of an instruction operand.
///
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
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PackedOperand(u16);

impl PackedOperand {
    pub fn new(op: &Operand) -> Self {
        match op {
            Operand::Local(lidx) => {
                debug_assert!(u16::from(*lidx) <= MAX_OPERAND_IDX);
                PackedOperand(u16::from(*lidx))
            }
            Operand::Const(constidx) => {
                debug_assert!(u16::from(*constidx) <= MAX_OPERAND_IDX);
                PackedOperand(u16::from(*constidx) | !OPERAND_IDX_MASK)
            }
        }
    }

    /// Unpacks a [PackedOperand] into a [Operand].
    pub(crate) fn unpack(&self, m: &Module) -> Operand {
        if (self.0 & !OPERAND_IDX_MASK) == 0 {
            let mut iidx = InstIdx(self.0);
            loop {
                match m.inst_all(iidx) {
                    Inst::ProxyConst(x) => {
                        return Operand::Const(*x);
                    }
                    Inst::ProxyInst(x) => {
                        iidx = *x;
                    }
                    _ => {
                        return Operand::Local(iidx);
                    }
                }
            }
        } else {
            Operand::Const(ConstIdx(self.0 & OPERAND_IDX_MASK))
        }
    }

    /// If this [PackedOperand] represents a local instruction, call `f` with its `InstIdx`. Note
    /// that this does not perform any kind of deproxification.
    fn map_iidx<F>(&self, f: &mut F)
    where
        F: FnMut(InstIdx),
    {
        if (self.0 & !OPERAND_IDX_MASK) == 0 {
            f(InstIdx(self.0))
        }
    }
}

/// An unpacked representation of a operand.
///
/// This exists both as a convenience (as working with packed operands is laborious) and as a means
/// to add type safety when using operands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Operand {
    Local(InstIdx),
    Const(ConstIdx),
}

impl Operand {
    /// Returns the size of the operand in bytes.
    ///
    /// Assumes no padding is required for alignment.
    ///
    /// # Panics
    ///
    /// Panics if asking for the size make no sense for this operand.
    pub(crate) fn byte_size(&self, m: &Module) -> usize {
        match self {
            Self::Local(l) => m.inst_all(*l).def_byte_size(m),
            Self::Const(cidx) => m.type_(m.const_(*cidx).tyidx(m)).byte_size().unwrap(),
        }
    }

    /// Returns the type index of the operand.
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        match self {
            Self::Local(l) => m.inst_all(*l).tyidx(m),
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
        if let Operand::Local(iidx) = self {
            f(*iidx)
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
            Operand::Local(idx) => match self.m.inst_all(*idx) {
                Inst::ProxyConst(c) => {
                    write!(f, "{}", self.m.const_(*c).display(self.m))
                }
                Inst::ProxyInst(idx) => {
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
    /// A constant integer at most 64 bits wide. This can be treated a signed or unsigned integer
    /// depending on the operations that use this constant (the [Ty::Integer] type itself has no
    /// concept of signedness).
    Int(TyIdx, u64),
    Ptr(usize),
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
            (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                lhs_tyidx == rhs_tyidx && lhs_v == rhs_v
            }
            (Const::Ptr(lhs_v), Const::Ptr(rhs_v)) => lhs_v == rhs_v,
            (_, _) => false,
        }
    }
}

impl Eq for ConstIndexSetWrapper {}

impl Hash for ConstIndexSetWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.0 {
            Const::Float(tyidx, v) => {
                tyidx.hash(state);
                // We treat floats as bit patterns: because we can accept duplicates, this is
                // acceptable.
                v.to_bits().hash(state);
            }
            Const::Int(tyidx, v) => {
                tyidx.hash(state);
                v.hash(state);
            }
            Const::Ptr(v) => v.hash(state),
        }
    }
}

impl Const {
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        match self {
            Const::Float(tyidx, _) => *tyidx,
            Const::Int(tyidx, _) => *tyidx,
            Const::Ptr(_) => m.ptr_tyidx,
        }
    }

    /// If this constant is an integer that can be represented in 64 bits, return it as an `i64`.
    pub(crate) fn int_to_u64(&self) -> Option<u64> {
        match self {
            Const::Float(_, _) => None,
            Const::Int(_, x) => Some(*x),
            Const::Ptr(_) => None,
        }
    }

    /// Create an integer of the same underlying type and with the value `x`.
    ///
    /// # Panics
    ///
    /// If `x` doesn't fit into the underlying integer type.
    pub(crate) fn u64_to_int(&self, x: u64) -> Const {
        match self {
            Const::Float(_, _) => panic!(),
            Const::Int(tyidx, _) => Const::Int(*tyidx, x),
            Const::Ptr(_) => panic!(),
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
                Ty::Float(FloatTy::Double) => write!(f, "{}double", v),
                _ => unreachable!(),
            },
            Const::Int(tyidx, x) => {
                let Ty::Integer(width) = self.m.type_(*tyidx) else {
                    panic!()
                };
                write!(f, "{x}i{width}")
            }
            Const::Ptr(x) => write!(f, "{:#x}", *x),
        }
    }
}

#[derive(Debug)]
/// Stores additional guard information.
pub(crate) struct GuardInfo {
    /// Live variables, mapping AOT vars to JIT [Operand]s.
    live_vars: Vec<(aot_ir::InstID, PackedOperand)>,
    // Inlined frames info.
    // FIXME With this field, the aotlives field is redundant.
    inlined_frames: Vec<InlinedFrame>,
}

impl GuardInfo {
    pub(crate) fn new(
        live_vars: Vec<(aot_ir::InstID, PackedOperand)>,
        inlined_frames: Vec<InlinedFrame>,
    ) -> Self {
        Self {
            live_vars,
            inlined_frames,
        }
    }

    /// Return the `AOT instruction -> PackedOperand` mapping for this guard.
    pub(crate) fn live_vars(&self) -> &[(aot_ir::InstID, PackedOperand)] {
        &self.live_vars
    }

    /// Return the [InlinedFunc]s for this guard.
    pub(crate) fn inlined_frames(&self) -> &[InlinedFrame] {
        &self.inlined_frames
    }
}

/// An abstract call frame for functions inlined in a trace. This contains enough information for a
/// failing guard to reconstruct real call frames on the stack.
#[derive(Debug, Clone)]
pub(crate) struct InlinedFrame {
    // The call instruction that led to [funcidx] being inlined.
    pub(crate) callinst: Option<aot_ir::InstID>,
    // The function that was inlined to create this abstract frame. Will be `None` for the top-most
    // function.
    pub(crate) funcidx: aot_ir::FuncIdx,
    /// The deopt safepoint for [callinst].
    pub(crate) safepoint: &'static aot_ir::DeoptSafepoint,
    /// The [Operand]s passed to [funcidx].
    pub(crate) args: Vec<Operand>,
}

impl InlinedFrame {
    pub(crate) fn new(
        callinst: Option<aot_ir::InstID>,
        funcidx: aot_ir::FuncIdx,
        safepoint: &'static aot_ir::DeoptSafepoint,
        args: Vec<Operand>,
    ) -> InlinedFrame {
        InlinedFrame {
            callinst,
            funcidx,
            safepoint,
            args,
        }
    }
}

/// An IR instruction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
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
    ProxyConst(ConstIdx),
    /// This instruction does not produce a value itself: it is equivalent to the value produced by
    /// `InstIdx`.
    #[allow(clippy::enum_variant_names)]
    ProxyInst(InstIdx),
    /// This instruction has been permanently removed. Note: this must only be used if you are
    /// entirely sure that the value this instruction once produced is no longer used.
    Tombstone,

    // "Normal" IR instructions.
    BinOp(BinOpInst),
    Load(LoadInst),
    LookupGlobal(LookupGlobalInst),
    LoadTraceInput(LoadTraceInputInst),
    Call(DirectCallInst),
    IndirectCall(IndirectCallIdx),
    PtrAdd(PtrAddInst),
    DynPtrAdd(DynPtrAddInst),
    Store(StoreInst),
    ICmp(ICmpInst),
    Guard(GuardInst),
    /// Marks the place to loop back to at the end of the JITted code.
    TraceLoopStart,
    TraceLoopJump,

    SExt(SExtInst),
    ZeroExtend(ZeroExtendInst),
    Trunc(TruncInst),
    Select(SelectInst),
    SIToFP(SIToFPInst),
    FPExt(FPExtInst),
    FCmp(FCmpInst),
    FPToSI(FPToSIInst),
}

impl Inst {
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
            Self::ProxyConst(x) => m.const_(*x).tyidx(m),
            Self::ProxyInst(x) => m.inst_all(*x).tyidx(m),
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
            Self::LoadTraceInput(li) => li.tyidx(),
            Self::Call(ci) => m.func_type(ci.target()).ret_tyidx(),
            Self::PtrAdd(..) => m.ptr_tyidx(),
            Self::DynPtrAdd(..) => m.ptr_tyidx(),
            Self::Store(..) => m.void_tyidx(),
            Self::ICmp(_) => m.int1_tyidx(),
            Self::Guard(..) => m.void_tyidx(),
            Self::TraceLoopStart => m.void_tyidx(),
            Self::TraceLoopJump => m.void_tyidx(),
            Self::SExt(si) => si.dest_tyidx(),
            Self::ZeroExtend(si) => si.dest_tyidx(),
            Self::Trunc(t) => t.dest_tyidx(),
            Self::Select(s) => s.trueval(m).tyidx(m),
            Self::SIToFP(i) => i.dest_tyidx(),
            Self::FPExt(i) => i.dest_tyidx(),
            Self::FCmp(_) => m.int1_tyidx(),
            Self::FPToSI(i) => i.dest_tyidx(),
        }
    }

    /// Does this instruction have a "side effect" that means that removing it would change the
    /// behaviour of later instructions even if they do not have a direct dependency on this
    /// instruction? "Side effects" are:
    ///   * Stores to memory locations.
    ///   * Guards.
    fn has_side_effect(&self, m: &Module) -> bool {
        match self {
            #[cfg(test)]
            Inst::BlackBox(_) => true,
            Inst::ProxyConst(_) => false,
            Inst::ProxyInst(x) => m.inst_all(*x).has_side_effect(m),
            Inst::Tombstone => false,
            Inst::BinOp(_) => false,
            Inst::Load(_) => false,
            Inst::LookupGlobal(_) => false,
            Inst::LoadTraceInput(_) => false,
            Inst::Call(_) => true,
            Inst::IndirectCall(_) => true,
            Inst::PtrAdd(_) => false,
            Inst::DynPtrAdd(_) => false,
            Inst::Store(_) => true,
            Inst::ICmp(_) => false,
            Inst::Guard(_) => true,
            Inst::TraceLoopStart => true,
            Inst::TraceLoopJump => true,
            Inst::SExt(_) => false,
            Inst::ZeroExtend(_) => false,
            Inst::Trunc(_) => false,
            Inst::Select(_) => false,
            Inst::SIToFP(_) => false,
            Inst::FPExt(_) => false,
            Inst::FCmp(_) => false,
            Inst::FPToSI(_) => false,
        }
    }

    /// Apply the function `f` to each of this instruction's [Operand]s iff they are of type
    /// [Operand::Local]. When an instruction references more than one [Operand], the order of
    /// traversal is undefined.
    pub(crate) fn map_operand_locals<F>(&self, m: &Module, f: &mut F)
    where
        F: FnMut(InstIdx),
    {
        match self {
            #[cfg(test)]
            Inst::BlackBox(BlackBoxInst { op }) => {
                op.unpack(m).map_iidx(f);
            }
            Inst::ProxyConst(_) => (),
            Inst::ProxyInst(_) => (),
            Inst::Tombstone => (),
            Inst::BinOp(BinOpInst { lhs, binop: _, rhs }) => {
                lhs.unpack(m).map_iidx(f);
                rhs.unpack(m).map_iidx(f);
            }
            Inst::Load(LoadInst { op, .. }) => op.unpack(m).map_iidx(f),
            Inst::LookupGlobal(_) => (),
            Inst::LoadTraceInput(_) => (),
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
            Inst::Store(StoreInst { tgt, val, .. }) => {
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
            Inst::TraceLoopStart => {
                for x in &m.loop_start_vars {
                    x.map_iidx(f);
                }
            }
            Inst::TraceLoopJump => {
                for x in &m.loop_jump_vars {
                    x.map_iidx(f);
                }
            }
            Inst::SExt(SExtInst { val, .. }) => val.unpack(m).map_iidx(f),
            Inst::ZeroExtend(ZeroExtendInst { val, .. }) => val.unpack(m).map_iidx(f),
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
        }
    }

    /// Apply the function `f` to each of this instruction's [PackedOperand]s that references a
    /// local instruction. When an instruction references more than one [PackedOperand], the order
    /// of traversal is undefined.
    ///
    /// Note that this function does not perform deproxification, and thus must only be used when
    /// you know that you want to know which local an instruction's operands directly refers to
    /// (e.g. for dead code elimination).
    pub(crate) fn map_packed_operand_locals<F>(&self, m: &Module, f: &mut F)
    where
        F: FnMut(InstIdx),
    {
        match self {
            #[cfg(test)]
            Inst::BlackBox(BlackBoxInst { op }) => {
                op.map_iidx(f);
            }
            Inst::ProxyConst(_) => (),
            Inst::ProxyInst(iidx) => f(*iidx),
            Inst::Tombstone => (),
            Inst::BinOp(BinOpInst { lhs, binop: _, rhs }) => {
                lhs.map_iidx(f);
                rhs.map_iidx(f);
            }
            Inst::Load(LoadInst { op, .. }) => op.map_iidx(f),
            Inst::LookupGlobal(_) => (),
            Inst::LoadTraceInput(_) => (),
            Inst::Call(x) => {
                for i in x.iter_args_idx() {
                    m.arg_packed(i).map_iidx(f);
                }
            }
            Inst::IndirectCall(x) => {
                let ici = m.indirect_call(*x);
                let IndirectCallInst { target, .. } = ici;
                target.map_iidx(f);
                for i in ici.iter_args_idx() {
                    m.arg_packed(i).map_iidx(f);
                }
            }
            Inst::PtrAdd(x) => {
                let ptr = x.ptr;
                ptr.map_iidx(f);
            }
            Inst::DynPtrAdd(x) => {
                let ptr = x.ptr;
                ptr.map_iidx(f);
                let num_elems = x.num_elems;
                num_elems.map_iidx(f);
            }
            Inst::Store(StoreInst { tgt, val, .. }) => {
                tgt.map_iidx(f);
                val.map_iidx(f);
            }
            Inst::ICmp(ICmpInst { lhs, pred: _, rhs }) => {
                lhs.map_iidx(f);
                rhs.map_iidx(f);
            }
            Inst::Guard(x @ GuardInst { cond, .. }) => {
                cond.map_iidx(f);
                for (_, pop) in x.guard_info(m).live_vars() {
                    pop.map_iidx(f);
                }
            }
            Inst::TraceLoopStart => {
                for val in &m.loop_start_vars {
                    match val {
                        Operand::Local(iidx) => f(*iidx),
                        _ => panic!(),
                    }
                }
            }
            Inst::TraceLoopJump => {
                for val in &m.loop_jump_vars {
                    match val {
                        Operand::Local(iidx) => f(*iidx),
                        Operand::Const(_) => (),
                    }
                }
            }
            Inst::SExt(SExtInst { val, .. }) => val.map_iidx(f),
            Inst::ZeroExtend(ZeroExtendInst { val, .. }) => val.map_iidx(f),
            Inst::Trunc(TruncInst { val, .. }) => val.map_iidx(f),
            Inst::Select(SelectInst {
                cond,
                trueval,
                falseval,
            }) => {
                cond.map_iidx(f);
                trueval.map_iidx(f);
                falseval.map_iidx(f);
            }
            Inst::SIToFP(SIToFPInst { val, .. }) => val.map_iidx(f),
            Inst::FPExt(FPExtInst { val, .. }) => val.map_iidx(f),
            Inst::FCmp(FCmpInst { lhs, pred: _, rhs }) => {
                lhs.map_iidx(f);
                rhs.map_iidx(f);
            }
            Inst::FPToSI(FPToSIInst { val, .. }) => val.map_iidx(f),
        }
    }

    /// Returns the size of the local variable that this instruction defines (if any).
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

    pub(crate) fn display<'a>(&'a self, iidx: InstIdx, m: &'a Module) -> DisplayableInst<'a> {
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
            Inst::ProxyConst(_) | Inst::ProxyInst(_) | Inst::Tombstone => unreachable!(),

            Inst::BinOp(BinOpInst { lhs, binop, rhs }) => write!(
                f,
                "{} {}, {}",
                binop,
                lhs.unpack(self.m).display(self.m),
                rhs.unpack(self.m).display(self.m)
            ),
            Inst::Load(x) => write!(f, "load {}", x.operand(self.m).display(self.m)),
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
                write!(
                    f,
                    "call @{}({})",
                    self.m.func_decl(x.target).name(),
                    (0..x.num_args())
                        .map(|y| format!("{}", x.operand(self.m, y).display(self.m)))
                        .collect::<Vec<_>>()
                        .join(", ")
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
                x.tgt.unpack(self.m).display(self.m),
                x.val.unpack(self.m).display(self.m)
            ),
            Inst::ICmp(x) => write!(
                f,
                "{} {}, {}",
                x.pred,
                x.lhs(self.m).display(self.m),
                x.rhs(self.m).display(self.m)
            ),
            Inst::Guard(
                x @ GuardInst {
                    cond,
                    expect,
                    gidx: _,
                },
            ) => {
                let live_vars = x
                    .guard_info(self.m)
                    .live_vars()
                    .iter()
                    .map(|(x, y)| {
                        format!(
                            "{}:%{}_{}: {}",
                            usize::from(x.funcidx()),
                            usize::from(x.bbidx()),
                            usize::from(x.iidx()),
                            y.unpack(self.m).display(self.m)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "guard {}, {}, [{live_vars}]",
                    if *expect { "true" } else { "false" },
                    cond.unpack(self.m).display(self.m),
                )
            }
            Inst::LoadTraceInput(x) => {
                write!(
                    f,
                    "load_ti {:?}",
                    self.m.tilocs[usize::try_from(x.locidx()).unwrap()]
                )
            }
            Inst::TraceLoopStart => {
                // Just marks a location, so we format it to look like a label.
                write!(f, "tloop_start [")?;
                for var in &self.m.loop_start_vars {
                    write!(f, "{}", var.display(self.m))?;
                    if var != self.m.loop_start_vars.last().unwrap() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]:")
            }
            Inst::TraceLoopJump => {
                // Just marks a location, so we format it to look like a label.
                write!(f, "tloop_jump [")?;
                for var in &self.m.loop_jump_vars {
                    write!(f, "{}", var.display(self.m))?;
                    if var != self.m.loop_jump_vars.last().unwrap() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]:")
            }
            Inst::SExt(i) => {
                write!(
                    f,
                    "sext {}, {}",
                    i.val(self.m).display(self.m),
                    self.m.type_(i.dest_tyidx()).display(self.m)
                )
            }
            Inst::ZeroExtend(i) => {
                write!(
                    f,
                    "zext {}, {}",
                    i.val(self.m).display(self.m),
                    self.m.type_(i.dest_tyidx()).display(self.m)
                )
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
inst!(LoadTraceInput, LoadTraceInputInst);
inst!(Call, DirectCallInst);
inst!(PtrAdd, PtrAddInst);
inst!(DynPtrAdd, DynPtrAddInst);
inst!(ICmp, ICmpInst);
inst!(Guard, GuardInst);
inst!(SExt, SExtInst);
inst!(ZeroExtend, ZeroExtendInst);
inst!(Trunc, TruncInst);
inst!(Select, SelectInst);
inst!(SIToFP, SIToFPInst);
inst!(FPExt, FPExtInst);
inst!(FCmp, FCmpInst);
inst!(FPToSI, FPToSIInst);

/// The operands for a [Instruction::BinOp]
///
/// # Semantics
///
/// Performs a binary operation.
///
/// The naming convention used is based on infix notation, e.g. in `2 + 3`, "2" is the left-hand
/// side (`lhs`), "+" is the binary operator (`binop`), and "3" is the right-hand side (`rhs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BinOpInst {
    /// The left-hand side of the operation.
    pub(crate) lhs: PackedOperand,
    /// The operation to perform.
    pub(crate) binop: BinOp,
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

    /// Returns the type index of the operands being added.
    pub(crate) fn tyidx(&self, m: &Module) -> TyIdx {
        self.lhs.unpack(m).tyidx(m)
    }
}

/// This is a test-only instruction which "consumes" an operand in the sense of "make use of the
/// value". This is useful to make clear in a test that an operand is used at a certain point,
/// which prevents optimisations removing some or all of the things that relate to this operand.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadInst {
    /// The pointer to load from.
    op: PackedOperand,
    /// The type of the pointee.
    tyidx: TyIdx,
    /// Is this load volatile?
    volatile: bool,
}

impl LoadInst {
    // FIXME: why do we need to provide a type index? Can't we get that from the operand?
    pub(crate) fn new(op: Operand, tyidx: TyIdx, volatile: bool) -> LoadInst {
        LoadInst {
            op: PackedOperand::new(&op),
            tyidx,
            volatile,
        }
    }

    /// Return the pointer operand.
    pub(crate) fn operand(&self, m: &Module) -> Operand {
        self.op.unpack(m)
    }

    /// Returns the type index of the loaded value.
    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }
}

/// The `LoadTraceInput` instruction.
///
/// ## Semantics
///
/// Loads a trace input out of the trace input struct. The variable is loaded from the specified
/// offset (`off`) and the resulting local variable is of the type indicated by the `tyidx`.
///
/// FIXME (maybe): If we added a third `TraceInput` storage class to the register allocator, could
/// we kill this instruction kind entirely?
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(packed)]
pub struct LoadTraceInputInst {
    /// The VarLocation of this input.
    locidx: u32,
    /// The type of the resulting local variable.
    tyidx: TyIdx,
}

impl LoadTraceInputInst {
    pub(crate) fn new(locidx: u32, tyidx: TyIdx) -> LoadTraceInputInst {
        Self { locidx, tyidx }
    }

    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    pub(crate) fn locidx(&self) -> u32 {
        self.locidx
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(packed)]
pub struct DirectCallInst {
    /// The callee.
    target: FuncDeclIdx,
    /// At what index do the contiguous operands in [Module::args] start?
    args_idx: ArgsIdx,
    /// How many arguments in [Module::args] is this call passing?
    num_args: u16,
}

impl DirectCallInst {
    pub(crate) fn new(
        m: &mut Module,
        target: FuncDeclIdx,
        args: Vec<Operand>,
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
}

/// The operands for a [Inst::Store]
///
/// # Semantics
///
/// Stores a value into a pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoreInst {
    /// The target pointer that we will store `val` into.
    tgt: PackedOperand,
    /// The value to store.
    val: PackedOperand,
    /// Is this store volatile?
    volatile: bool,
}

impl StoreInst {
    pub(crate) fn new(tgt: Operand, val: Operand, volatile: bool) -> Self {
        // FIXME: assert type of pointer
        Self {
            tgt: PackedOperand::new(&tgt),
            val: PackedOperand::new(&val),
            volatile,
        }
    }

    /// Returns the value operand: i.e. the thing that is going to be stored.
    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    /// Returns the target operand: i.e. where to store [self.val()].
    pub(crate) fn tgt(&self, m: &Module) -> Operand {
        self.tgt.unpack(m)
    }
}

/// The operands for a [Inst::Select]
///
/// # Semantics
///
/// Selects from two values depending on a condition.
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// An instruction that adds a constant offset to a pointer.
///
/// # Semantics
///
/// Returns a pointer value that is the result of adding the specified, signed, constant (byte)
/// offset to the input pointer operand.
///
/// Following LLVM semantics, the operation is permitted to silently wrap if the result doesn't fit
/// in the LLVM pointer indexing type.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(packed)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(packed)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZeroExtendInst {
    /// The value to extend.
    val: PackedOperand,
    /// The type to extend to.
    dest_tyidx: TyIdx,
}

impl ZeroExtendInst {
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub(crate) fn val(&self, m: &Module) -> Operand {
        self.val.unpack(m)
    }

    pub(crate) fn dest_tyidx(&self) -> TyIdx {
        self.dest_tyidx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn use_case_update_inst() {
        let mut prog: Vec<Inst> = vec![
            LoadTraceInputInst::new(0, TyIdx::try_from(0).unwrap()).into(),
            LoadTraceInputInst::new(8, TyIdx::try_from(0).unwrap()).into(),
            LoadInst::new(
                Operand::Local(InstIdx(0)),
                TyIdx(U24::try_from(0).unwrap()),
                false,
            )
            .into(),
        ];
        prog[2] = LoadInst::new(
            Operand::Local(InstIdx(1)),
            TyIdx(U24::try_from(0).unwrap()),
            false,
        )
        .into();
    }

    /// Ensure that any given instruction fits in 64-bits.
    #[test]
    fn inst_size() {
        assert!(mem::size_of::<Inst>() <= mem::size_of::<u64>());
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
            Operand::Local(InstIdx(0)),
            Operand::Local(InstIdx(1)),
            Operand::Local(InstIdx(2)),
        ];
        m.push(Inst::LoadTraceInput(LoadTraceInputInst::new(0, i32_tyidx)))
            .unwrap();
        m.push(Inst::LoadTraceInput(LoadTraceInputInst::new(1, i32_tyidx)))
            .unwrap();
        m.push(Inst::LoadTraceInput(LoadTraceInputInst::new(2, i32_tyidx)))
            .unwrap();
        let ci = DirectCallInst::new(&mut m, func_decl_idx, args).unwrap();

        // Now request the operands and check they all look as they should.
        assert_eq!(ci.operand(&m, 0), Operand::Local(InstIdx(0)));
        assert_eq!(ci.operand(&m, 1), Operand::Local(InstIdx(1)));
        assert_eq!(ci.operand(&m, 2), Operand::Local(InstIdx(2)));
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
            Operand::Local(InstIdx(0)),
            Operand::Local(InstIdx(1)),
            Operand::Local(InstIdx(2)),
        ];
        let ci = DirectCallInst::new(&mut m, func_decl_idx, args).unwrap();

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
        let i8_tyidx = m.insert_ty(Ty::Integer(8)).unwrap();
        assert_eq!(Const::Int(i8_tyidx, 0).display(&m).to_string(), "0i8");
        assert_eq!(Const::Int(i8_tyidx, 255).display(&m).to_string(), "255i8");
        let i64_tyidx = m.insert_ty(Ty::Integer(64)).unwrap();
        assert_eq!(Const::Int(i64_tyidx, 0).display(&m).to_string(), "0i64");
        assert_eq!(
            Const::Int(i64_tyidx, 9223372036854775808)
                .display(&m)
                .to_string(),
            "9223372036854775808i64"
        );
    }

    #[test]
    fn stringify_const_ptr() {
        let m = Module::new_testing();
        let ptr_val = stringify_const_ptr as usize;
        let cp = Const::Ptr(ptr_val);
        assert_eq!(
            cp.display(&m).to_string(),
            format!("{:#x}", ptr_val as usize)
        );
    }

    #[test]
    fn print_module() {
        let mut m = Module::new_testing();
        m.push_tiloc(yksmp::Location::Register(3, 1, 0, 0));
        m.push(LoadTraceInputInst::new(0, m.int8_tyidx()).into())
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
            "    %0: i8 = load_ti Register(3, 1, 0, 0)",
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
}
