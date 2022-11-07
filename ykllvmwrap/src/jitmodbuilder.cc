// Classes and functions for constructing a new LLVM module from a trace.

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "jitmodbuilder.h"

#include <atomic>
#include <err.h>

using namespace llvm;
using namespace std;

// An atomic counter used to issue compiled traces with unique names.
atomic<uint64_t> NextTraceIdx(0);
uint64_t getNewTraceIdx() {
  uint64_t TraceIdx = NextTraceIdx.fetch_add(1, memory_order_relaxed);
  assert(TraceIdx != numeric_limits<uint64_t>::max());
  return TraceIdx;
}

#define TRACE_FUNC_PREFIX "__yk_compiled_trace_"
#define YK_NEW_CONTROL_POINT "__ykrt_control_point"
#define YK_CONTROL_POINT_ARG_VARS_IDX 2
#define YK_CONTROL_POINT_ARG_FRAMEADDR_IDX 3
#define YK_CONTROL_POINT_NUM_ARGS 4

#define JITFUNC_ARG_INPUTS_STRUCT_IDX 0
#define JITFUNC_ARG_STACKMAP_ADDR_IDX 1
#define JITFUNC_ARG_STACKMAP_LEN_IDX 2
#define JITFUNC_ARG_FRAMEADDR_IDX 3
#define JITFUNC_ARG_LIVEAOTVALS_PTR_IDX 4

#define YK_OUTLINE_FNATTR "yk_outline"

// The first two arguments of a stackmap call are it's id and shadow bytes and
// need to be skipped when scanning the operands for live values.
#define YK_STACKMAP_SKIP_ARGS 2

// Return value telling the caller of the compiled trace that no guard failure
// occurred and it's safe to continue running the fake trace stitching loop.
#define TRACE_RETURN_SUCCESS 0

// The name prefix used for blocks that are branched to when a guard succeeds.
#define GUARD_SUCCESS_BLOCK_NAME "guardsuccess"

// Dump an error message and an LLVM value to stderr and exit with failure.
void dumpValueAndExit(const char *Msg, Value *V) {
  errs() << Msg << ": ";
  V->dump();
  exit(EXIT_FAILURE);
}

// A function name and basic block index pair that identifies a block in the
// AOT LLVM IR.
struct IRBlock {
  // A non-null pointer to the function name.
  char *FuncName;
  // The index of the block in the parent LLVM function.
  size_t BBIdx;
};

// Describes the software or hardware trace to be compiled using LLVM.
class InputTrace {
private:
  // An ordered array of function names. Each non-null element describes the
  // function part of a (function, block) pair that identifies an LLVM
  // BasicBlock. A null element represents unmappable code in the trace.
  char **FuncNames;
  // An ordered array of basic block indices. Each element corresponds with
  // an element (at the same index) in the above `FuncNames` array to make a
  // (function, block) pair that identifies an LLVM BasicBlock.
  size_t *BBs;
  // The length of the `FuncNames` and `BBs` arrays.
  size_t Len;

public:
  InputTrace(char **FuncNames, size_t *BBs, size_t Len)
      : FuncNames(FuncNames), BBs(BBs), Len(Len) {}
  size_t Length() { return Len; }

  // Returns the optional IRBlock at index `Idx` in the trace. No value is
  // returned if element at `Idx` was unmappable. It is undefined behaviour to
  // invoke this method with an out-of-bounds `Idx`.
  const Optional<IRBlock> operator[](size_t Idx) {
    assert(Idx < Len);
    char *FuncName = FuncNames[Idx];
    if (FuncName == nullptr) {
      return Optional<IRBlock>();
    } else {
      return Optional<IRBlock>(IRBlock{FuncName, BBs[Idx]});
    }
  }

  // The same as `operator[]`, but for scenarios where you are certain that the
  // element at position `Idx` cannot be unmappable.
  const IRBlock getUnchecked(size_t Idx) {
    assert(Idx < Len);
    char *FuncName = FuncNames[Idx];
    assert(FuncName != nullptr);
    return IRBlock{FuncName, BBs[Idx]};
  }
};

// Function virtual addresses observed in the input trace.
// Maps a function symbol name to a virtual address.
class FuncAddrs {
  map<string, void *> Map;

public:
  FuncAddrs(char **FuncNames, void **VAddrs, size_t Len) {
    for (size_t I = 0; I < Len; I++) {
      Map.insert({FuncNames[I], VAddrs[I]});
    }
  }

  // Lookup the address of the specified function name or return nullptr on
  // failure.
  void *operator[](const char *FuncName) {
    auto It = Map.find(FuncName);
    if (It == Map.end())
      return nullptr; // Not found.
    return It->second;
  }
};

// Struct to store an active frame, consisting of its location in the AOT
// module and the instruction calling it.
struct FrameInfo {
  size_t BBIdx;
  size_t InstrIdx;
  StringRef Fname;
  CallBase *SMCall;
};

// Struct to store a live AOT value.
struct AOTInfo {
  size_t BBIdx;
  size_t InstrIdx;
  const char *FName;
  size_t FrameIdx;
};

/// Extract all live variables that need to be passed into the control point.
/// FIXME: This is currently an over-approximation and will return some
/// variables that are no longer alive.
std::vector<Value *> getLiveVars(
    DominatorTree &DT, Instruction *Before,
    std::map<Value *, std::tuple<size_t, size_t, Instruction *, size_t>> AOTMap,
    size_t CallDepth) {
  std::vector<Value *> Vec;
  Function *Func = Before->getFunction();
  for (auto &BB : *Func) {
    if (!DT.dominates(cast<Instruction>(Before), &BB)) {
      for (auto &I : BB) {
        if ((!I.getType()->isVoidTy()) &&
            (DT.dominates(&I, cast<Instruction>(Before)))) {
          if (AOTMap.find(&I) == AOTMap.end()) {
            // We sometimes create extra variables that don't have a
            // corresponding AOT variable. For example an AOT switch statement
            // becomes two instructions, `icmp` and `br`. The `icmp`
            // instruction can then be live and appear here. Since its value is
            // inconsequential for the stopgap interpreter we can safely ignore
            // it.
            continue;
          }
          std::tuple<size_t, size_t, Instruction *, size_t> Entry = AOTMap[&I];
          size_t StackFrameIdx = std::get<3>(Entry);
          if (StackFrameIdx > CallDepth) {
            // Due to the over-approximation this function may pick up
            // variables from an already finished stack frame. Adding these
            // variables will lead to the stopgap interpeter trying to
            // initialise them into a stack frame that doesn't exist. We can
            // work around this by disallowing live variables for out-of-range
            // stack frame indexes. While this still means that the stopgap
            // interpreter can intialise some dead variables inside its frames,
            // this is still treated as an over-approximation and won't lead to
            // an error. Ultimately, this whole function needs to be replaced
            // with proper liveness analysis, at which point this workaround
            // won't be needed anymore.
            continue;
          }
          Vec.push_back(&I);
        }
      }
    }
  }
  return Vec;
}

class JITModBuilder {
  // Global variables/functions that were copied over and need to be
  // initialised.
  vector<GlobalVariable *> cloned_globals;
  // The module being traced.
  Module *AOTMod;
  // The new module that is being build.
  Module *JITMod;
  // A pointer to the call to YK_NEW_CONTROL_POINT in the AOT module (once
  // encountered). When this changes from NULL to non-NULL, then we start
  // copying instructions from the AOT module into the JIT module.
  Instruction *NewControlPointCall = nullptr;
  // Stack of inlined calls, required to resume at the correct place in the
  // caller.
  std::vector<tuple<size_t, CallInst *>> InlinedCalls;
  // Instruction at which to continue after a call.
  Optional<tuple<size_t, CallInst *>> ResumeAfter;
  // Active stackframes at each guard. Stores basic block index, instruction
  // index, function name.
  std::vector<FrameInfo> ActiveFrames;
  // The depth of the call-stack.
  size_t CallDepth = 0;
  // The depth of the call-stack during function outlining.
  size_t OutlineCallDepth = 0;
  // Signifies a hole (for which we have no IR) in the trace.
  bool ExpectUnmappable = false;
  // The JITMod's builder.
  llvm::IRBuilder<> Builder;
  // Dead values to recursively delete upon finalisation of the JITMod. This is
  // required because it's not safe to recursively delete values in the middle
  // of creating the JIT module. We don't know if any of those values might be
  // required later in the trace.
  vector<Value *> DeleteDeadOnFinalise;

  // Information about the trace we are compiling.
  InputTrace InpTrace;
  // Function virtual addresses discovered from the input trace.
  FuncAddrs FAddrs;

  // A stack of BasicBlocks. Each time we enter a new call frame, we push the
  // first basic block to the stack. Following a branch to another basic block
  // updates the most recently pushed block. This is required for selecting the
  // correct incoming value when tracing a PHI node.
  vector<BasicBlock *> LastCompletedBlocks;

  // Maps field indices in the live variables struct to the value stored prior
  // to calling the control point.
  std::map<uint64_t, Value *> LiveIndexMap;

  // The LLVM type for a C `int` on the current machine.
  Type *IntTy;

  // A pointer-sized integer (i.e. a Rust `usize`) for the current machine.
  IntegerType *PointerSizedIntTy;

  // A pointer to the Value representing the trace input struct.
  Value *TraceInputs;

  // A pointer to the instruction that calls the patched control point.
  CallInst *ControlPointCallInst;

  // The function inside which we build the IR for the trace.
  Function *JITFunc;

  // Map JIT instruction to basic block index, instruction index, function
  // name, and stackframe index of the corresponding AOT instruction.
  std::map<Value *, std::tuple<size_t, size_t, Instruction *, size_t>> AOTMap;

  // The last seen call to llvm.exerpimental.stackmap. We use this to keep
  // track of the AOT live values on a per-frame basis. Matching those values
  // to their corresponding values in the JIT tells us which values need to be
  // deoptimised.
  CallBase *LastSMCall;

  Value *getMappedValue(Value *V) {
    if (VMap.find(V) != VMap.end()) {
      return VMap[V];
    }
    assert(isa<Constant>(V));
    return V;
  }

  void insertAOTMap(Instruction *AOT, Value *JIT, size_t BBIdx,
                    size_t InstrIdx) {
    AOTMap[JIT] = {BBIdx, InstrIdx, AOT, CallDepth};
  }

  // Returns true if the given function exists on the call stack, which means
  // this is a recursive call.
  bool isRecursiveCall(Function *F) {
    for (auto Tup : InlinedCalls) {
      CallInst *CInst = get<1>(Tup);
      if (CInst->getCalledFunction() == F) {
        return true;
      }
    }
    return false;
  }

  // Determines whether we are currently outlining code, i.e. we are processing
  // the IR of a function we do not wish to inline.
  bool isOutlining() { return OutlineCallDepth > 0; }

  // Add an external declaration for the given function to JITMod.
  void declareFunction(Function *F) {
    assert(JITMod->getFunction(F->getName()) == nullptr);
    auto DeclFunc = llvm::Function::Create(F->getFunctionType(),
                                           GlobalValue::ExternalLinkage,
                                           F->getName(), JITMod);
    VMap[F] = DeclFunc;
  }

  // Find the machine code corresponding to the given AOT IR function and
  // ensure there's a mapping from its name to that machine code.
  void addGlobalMappingForFunction(Function *CF) {
    StringRef CFName = CF->getName();
    void *FAddr = FAddrs[CFName.data()];
    assert(FAddr != nullptr);
    GlobalMappings.insert({CF, FAddr});
  }

  // Generate LLVM IR to create a struct and store the given values.
  AllocaInst *createAndFillStruct(IRBuilder<> &Builder,
                                  std::vector<Value *> Vec) {
    LLVMContext &Context = Builder.getContext();
    IntegerType *Int32Ty = Type::getInt32Ty(Context);
    std::vector<Type *> Types;
    for (Value *V : Vec) {
      Types.push_back(V->getType());
    }
    StructType *StructTy = StructType::get(Context, Types);
    AllocaInst *Struct =
        Builder.CreateAlloca(StructTy, ConstantInt::get(PointerSizedIntTy, 1));
    for (size_t I = 0; I < Vec.size(); I++) {
      auto GEP = Builder.CreateGEP(StructTy, Struct,
                                   {ConstantInt::get(PointerSizedIntTy, 0),
                                    ConstantInt::get(Int32Ty, I)});
      Builder.CreateStore(Vec[I], GEP);
    }
    return Struct;
  }

  void handleCallInst(CallInst *CI, Function *CF, size_t &CurBBIdx,
                      size_t &CurInstrIdx) {
    if (CF == nullptr || CF->isDeclaration()) {
      // The definition of the callee is external to AOTMod. We still
      // need to declare it locally if we have not done so yet.
      if (CF != nullptr && VMap.find(CF) == VMap.end()) {
        declareFunction(CF);
      }
      if (!isOutlining()) {
        copyInstruction(&Builder, (Instruction *)&*CI, CurBBIdx, CurInstrIdx);
      }
      // We should expect an "unmappable hole" in the trace. This is
      // where the trace followed a call into external code for which we
      // have no IR, and thus we cannot map blocks for.
      ExpectUnmappable = true;
      ResumeAfter = make_tuple(CurInstrIdx, CI);
    } else {
      LastCompletedBlocks.push_back(nullptr);
      if (isOutlining()) {
        // When outlining a recursive function, we need to count all other
        // function calls so we know when we left the recusion.
        OutlineCallDepth += 1;
        InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
      }
      // If this is a recursive call that has been inlined, or if the callee
      // has the "yk_outline" annotation, remove the inlined code and turn it
      // into a normal (outlined) call.
      else if (CF->hasFnAttribute(YK_OUTLINE_FNATTR) || CF->isVarArg() ||
               isRecursiveCall(CF)) {
        if (VMap.find(CF) == VMap.end()) {
          declareFunction(CF);
          addGlobalMappingForFunction(CF);
        }
        copyInstruction(&Builder, CI, CurBBIdx, CurInstrIdx);
        InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
        OutlineCallDepth = 1;
      } else {
        // Otherwise keep the call inlined.
        InlinedCalls.push_back(make_tuple(CurInstrIdx, CI));
        // Remap function arguments to the variables passed in by the caller.
        for (unsigned int i = 0; i < CI->arg_size(); i++) {
          Value *Var = CI->getArgOperand(i);
          Value *Arg = CF->getArg(i);
          // Check the operand for things we need to remap, e.g. globals.
          handleOperand(Var);
          // If the operand has already been cloned into JITMod then we
          // need to use the cloned value in the VMap.
          VMap[Arg] = getMappedValue(Var);
        }
      }
      LastSMCall = cast<CallBase>(CI->getNextNonDebugInstruction());
      ActiveFrames.push_back(
          {CurBBIdx, CurInstrIdx, CI->getFunction()->getName(), LastSMCall});
      CallDepth += 1;
    }
  }

  // Emits a guard for a LLVM `br` instruction, returning a pointer to the
  // guard success block, or null if no guard was required.
  BasicBlock *handleBranchInst(Function *JITFunc, BasicBlock *NextBlock,
                               Instruction *I, size_t CurBBIdx,
                               size_t CurInstrIdx) {
    assert(isa<BranchInst>(I));
    BranchInst *BI = cast<BranchInst>(I);
    LLVMContext &Context = JITMod->getContext();

    if (BI->isUnconditional())
      return nullptr; // Control-flow can not diverge. No guard required.

    if (BI->getCondition() == NewControlPointCall) {
      // This is the branch that checks if we need to return after executing the
      // control point. We don't need to emit a guard for it, as the condition
      // is guaranteed to be false here (it can only be true after having just
      // executed a trace).
      return nullptr;
    }

    // A conditional branch should have two successors and one of them should
    // be the block we observed in the trace.
    assert(BI->getNumSuccessors() == 2);
    assert((BI->getSuccessor(0) == NextBlock) ||
           (BI->getSuccessor(1) == NextBlock));

    // Get/create the guard failure and success blocks.
    //
    // Note that we don't have to worry about the block names being unique, as
    // LLVM will make it so by appending a number to the block's name.
    BasicBlock *FailBB = getGuardFailureBlock(
        JITFunc,
        {CurBBIdx, CurInstrIdx, I->getFunction()->getName(), LastSMCall});
    BasicBlock *SuccBB =
        BasicBlock::Create(Context, GUARD_SUCCESS_BLOCK_NAME, JITFunc);

    // Insert the guard, using the original AOT branch condition for now.
    //
    // OPT: Could add branch weights to `CreateCondBr` to hint to LLVM that we
    // expect the guard to rarely fail?
    BranchInst *Guard = Builder.CreateCondBr(getMappedValue(BI->getCondition()),
                                             SuccBB, FailBB);

    // If the trace took the false arm of the AOT branch, then we have to
    // invert the condition of the guard we just inserted into the trace.
    if (BI->getSuccessor(0) != NextBlock)
      Guard->swapSuccessors();

    return SuccBB;
  }

  // Emits a guard for a LLVM `switch` instruction, returning a pointer to the
  // guard success block.
  BasicBlock *handleSwitchInst(Function *JITFunc, BasicBlock *NextBlock,
                               Instruction *I, size_t CurBBIdx,
                               size_t CurInstrIdx) {
    assert(isa<SwitchInst>(I));
    SwitchInst *SI = cast<SwitchInst>(I);

    // Get/create the guard failure and success blocks.
    LLVMContext &Context = JITMod->getContext();
    BasicBlock *FailBB = getGuardFailureBlock(
        JITFunc,
        {CurBBIdx, CurInstrIdx, I->getFunction()->getName(), LastSMCall});
    BasicBlock *SuccBB =
        BasicBlock::Create(Context, GUARD_SUCCESS_BLOCK_NAME, JITFunc);

    // Determine which switch case the trace took.
    ConstantInt *MatchedValue = SI->findCaseDest(NextBlock);
    if (MatchedValue != nullptr) {
      // A non-default case was taken.
      Value *Cmp = Builder.CreateICmpEQ(getMappedValue(SI->getCondition()),
                                        MatchedValue);
      Builder.CreateCondBr(Cmp, SuccBB, FailBB);
    } else {
      // The default case was taken.
      SwitchInst *NewSI = Builder.CreateSwitch(
          getMappedValue(SI->getCondition()), SuccBB, SI->getNumCases());
      for (SwitchInst::CaseHandle Case : SI->cases())
        NewSI->addCase(Case.getCaseValue(), FailBB);
    }

    return SuccBB;
  }

  void handleReturnInst(Instruction *I, size_t CurBBIdx, size_t CurInstrIdx) {
    ActiveFrames.pop_back();
    CallDepth -= 1;
    ResumeAfter = InlinedCalls.back();
    InlinedCalls.pop_back();
    LastCompletedBlocks.pop_back();
    if (isOutlining()) {
      OutlineCallDepth -= 1;
      return;
    }
    // Replace the return variable of the call with its return value.
    // Since the return value will have already been copied over to the
    // JITModule, make sure we look up the copy.
    auto OldRetVal = ((ReturnInst *)&*I)->getReturnValue();
    if (OldRetVal != nullptr) {
      assert(ResumeAfter.hasValue());
      // Update the AOTMap accordingly.
      Instruction *AOT = get<1>(ResumeAfter.getValue());
      Value *JIT = getMappedValue(OldRetVal);
      VMap[AOT] = getMappedValue(OldRetVal);
      insertAOTMap(AOT, JIT, CurBBIdx, CurInstrIdx);
    }
  }

  void handlePHINode(Instruction *I, BasicBlock *BB, size_t CurBBIdx,
                     size_t CurInstrIdx) {
    Value *V = ((PHINode *)&*I)->getIncomingValueForBlock(BB);
    // From the point of view of the compiled trace it's enough to just update
    // the `VMap` with the incoming value `V`. However, this means there's no
    // corresponding instruction in the JITMod and thus there's nothing mapping
    // back to it during a guard failure. To solve this we need to assign the
    // incoming value to a new SSA variable to make sure there's a mapping from
    // a JITMod value back to an AOT value and that the latter is initialised
    // in the stopgap interpreter. Unfortunately for us, LLVM doesn't have a
    // simple assignment instruction so we have to emulate one using a select
    // instruction.
    if (isa<GlobalVariable>(V)) {
      // If the operand is a global variable, remember to clone it, or else the
      // select instruction will reference the one in the old module.
      V = cloneGlobalVariable(V);
    }
    Instruction *NewInst = SelectInst::Create(
        ConstantInt::get(Type::getInt1Ty(JITMod->getContext()), 0), V, V);
    Builder.Insert(NewInst);
    llvm::RemapInstruction(NewInst, VMap, RF_NoModuleLevelChanges);
    VMap[&*I] = NewInst;
    insertAOTMap(I, NewInst, CurBBIdx, CurInstrIdx);
  }

  Function *createJITFunc(Value *TraceInputs, Type *FrameAddr) {
    // Compute a name for the trace.
    uint64_t TraceIdx = getNewTraceIdx();
    TraceName = string(TRACE_FUNC_PREFIX) + to_string(TraceIdx);

    // Create the function.
    std::vector<Type *> InputTypes;
    InputTypes.push_back(TraceInputs->getType());

    // Add arguments for stackmap pointer and size.
    InputTypes.push_back(PointerSizedIntTy->getPointerTo());
    InputTypes.push_back(PointerSizedIntTy);

    // Add argument in which to store the value of an interpreted return.
    InputTypes.push_back(FrameAddr);

    // Add argument for memory block holding live AOT values.
    InputTypes.push_back(PointerSizedIntTy->getPointerTo());

    llvm::FunctionType *FType = llvm::FunctionType::get(
        PointerType::get(JITMod->getContext(), 0), InputTypes, false);
    llvm::Function *JITFunc = llvm::Function::Create(
        FType, Function::ExternalLinkage, TraceName, JITMod);
    JITFunc->setCallingConv(CallingConv::C);

    return JITFunc;
  }

  // Delete the dead value `V` from its parent, also deleting any dependencies
  // of `V` (i.e. operands) which then become dead.
  void deleteDeadTransitive(Value *V) {
    assert(V->user_empty()); // The value must be dead.
    vector<Value *> Work;
    Work.push_back(V);
    while (!Work.empty()) {
      Value *V = Work.back();
      Work.pop_back();
      // Remove `V` (an instruction or a global variable) from its parent
      // container. If any of the operands of `V` have a sole use, then they
      // will become dead and can also be deleted too.
      if (isa<Instruction>(V)) {
        Instruction *I = cast<Instruction>(V);
        for (auto &Op : I->operands()) {
          if (Op->hasOneUser()) {
            Work.push_back(&*Op);
          }
        }
        I->eraseFromParent();
      } else if (isa<GlobalVariable>(V)) {
        GlobalVariable *G = cast<GlobalVariable>(V);
        for (auto &Op : G->operands()) {
          if (Op->hasOneUser()) {
            Work.push_back(&*Op);
          }
        }
        // Be sure to remove this global variable from `cloned_globals` too, so
        // that we don't try to add an initialiser later in `finalise()`.
        erase_if(cloned_globals, [G, this](GlobalVariable *CG) {
          assert(VMap.find(CG) != VMap.end());
          return G == VMap[CG];
        });
        G->eraseFromParent();
      } else {
        dumpValueAndExit("Unexpected Value", V);
      }
    }
  }

  // Given an `IRBlock`, find and return the LLVM data structures for the basic
  // block and its parent function.
  std::pair<Function *, BasicBlock *> getLLVMAOTFuncAndBlock(IRBlock *IB) {
    Function *F = AOTMod->getFunction(IB->FuncName);
    assert(F != nullptr);

    // Skip to the correct block.
    auto It = F->begin();
    std::advance(It, IB->BBIdx);
    BasicBlock *BB = &*It;

    return {F, BB};
  }

  // Returns a pointer to the guard failure block, creating it if necessary.
  BasicBlock *getGuardFailureBlock(Function *JITFunc, FrameInfo FInfo) {
    // If `JITFunc` contains no blocks already, then the guard failure block
    // becomes the entry block. This would lead to a trace that
    // unconditionally and immediately fails a guard.
    assert(JITFunc->getBasicBlockList().size() != 0);

    // Declare `errx(3)`.
    LLVMContext &Context = JITFunc->getContext();

    // Create the block.
    // FIXME: Cache guard blocks where live variables and frames are the same,
    // e.g. this can happen when a loop is unrolled and the same condition
    // produces a guard in each unrolled iteration.
    BasicBlock *GuardFailBB = BasicBlock::Create(Context, "guardfail", JITFunc);
    IRBuilder<> FailBuilder(GuardFailBB);

    // Add the control point struct to the live variables we pass into the
    // `deoptimize` call so the stopgap interpreter can access it.
    Value *YKCPArg = JITFunc->getArg(JITFUNC_ARG_INPUTS_STRUCT_IDX);
    std::tuple<size_t, size_t, Instruction *, size_t> YkCPAlloca =
        getYkCPAlloca();
    AOTMap[YKCPArg] = YkCPAlloca;

    IntegerType *Int32Ty = Type::getInt32Ty(Context);
    PointerType *Int8PtrTy = Type::getInt8PtrTy(Context);

    // Create a vector of active stackframes (i.e. basic block index,
    // instruction index, function name). This will be needed later to point
    // the stopgap interpeter at the correct location from where to start
    // interpretation and to setup its stackframes.
    // FIXME: Use function index instead of string name.
    ActiveFrames.push_back(FInfo); // Add current frame.
    StructType *ActiveFrameSTy = StructType::get(
        Context, {PointerSizedIntTy, PointerSizedIntTy, Int8PtrTy});
    AllocaInst *ActiveFrameVec = FailBuilder.CreateAlloca(
        ActiveFrameSTy,
        ConstantInt::get(PointerSizedIntTy, ActiveFrames.size()));

    std::vector<Value *> LiveValues;
    for (size_t I = 0; I < ActiveFrames.size(); I++) {
      FrameInfo FI = ActiveFrames[I];

      // Read live AOTMod values from stackmap calls and find their
      // corresponding values in JITMod. These are exactly the values that are
      // live at each guard failure and need to be deoptimised.
      CallBase *SMC = FI.SMCall;
      for (size_t Idx = YK_STACKMAP_SKIP_ARGS; Idx < SMC->arg_size(); Idx++) {
        Value *Arg = SMC->getArgOperand(Idx);
        if (Arg == NewControlPointCall) {
          // There's no corresponding JIT value for the return value of the
          // control point call, so just skip it.
          continue;
        }
        if (VMap.find(Arg) != VMap.end()) {
          Value *JITArg = VMap[Arg];
          LiveValues.push_back(JITArg);
        }
      }

      // Create GEP instructions to get pointers into the fields of the
      // individual frames inside the ActiveFrameVec vector. The first index
      // is for the element in the vector at position `I` (and is thus
      // pointer sized). The second index is for the field inside that
      // element (since each element has only 3 fields a Int8 would suffice,
      // but for convenience we just use the Int32Ty we already have defined
      // above).
      auto GEP = FailBuilder.CreateGEP(ActiveFrameSTy, ActiveFrameVec,
                                       {ConstantInt::get(PointerSizedIntTy, I),
                                        ConstantInt::get(Int32Ty, 0)});
      FailBuilder.CreateStore(ConstantInt::get(PointerSizedIntTy, FI.BBIdx),
                              GEP);
      GEP = FailBuilder.CreateGEP(ActiveFrameSTy, ActiveFrameVec,
                                  {ConstantInt::get(PointerSizedIntTy, I),
                                   ConstantInt::get(Int32Ty, 1)});
      FailBuilder.CreateStore(ConstantInt::get(PointerSizedIntTy, FI.InstrIdx),
                              GEP);
      Value *CurFunc = FailBuilder.CreateGlobalStringPtr(FI.Fname);
      GEP = FailBuilder.CreateGEP(ActiveFrameSTy, ActiveFrameVec,
                                  {ConstantInt::get(PointerSizedIntTy, I),
                                   ConstantInt::get(Int32Ty, 2)});
      FailBuilder.CreateStore(CurFunc, GEP);
    }

    // Store the active frames vector and its length in a separate struct to
    // save arguments.
    AllocaInst *ActiveFramesStruct = createAndFillStruct(
        FailBuilder, {ActiveFrameVec, ConstantInt::get(PointerSizedIntTy,
                                                       ActiveFrames.size())});

    // Make more space to store the locations of the corresponding live AOT
    // values for this guard failure.
    size_t CurPos = LiveAOTNum;
    LiveAOTNum += LiveValues.size();
    LiveAOTArray = static_cast<AOTInfo *>(
        reallocarray(LiveAOTArray, LiveAOTNum, sizeof(AOTInfo)));
    assert(LiveAOTArray != NULL);
    // Get a pointer to this guard failure's region in the memory block.
    AOTInfo *CurrentRegion = &LiveAOTArray[CurPos];

    for (size_t I = 0; I < LiveValues.size(); I++) {
      Value *Live = LiveValues[I];
      std::tuple<size_t, size_t, Instruction *, size_t> Entry = AOTMap[Live];
      size_t BBIdx = std::get<0>(Entry);
      size_t InstrIdx = std::get<1>(Entry);
      Instruction *AOTVar = std::get<2>(Entry);
      size_t StackFrameIdx = std::get<3>(Entry);
      const char *FName = AOTVar->getFunction()->getName().data();
      CurrentRegion[I] = {BBIdx, InstrIdx, FName, StackFrameIdx};
    }

    // Store the live variable vector and its length in a separate struct to
    // save arguments.
    AllocaInst *AOTLocs = createAndFillStruct(
        FailBuilder,
        {JITFunc->getArg(JITFUNC_ARG_LIVEAOTVALS_PTR_IDX),
         ConstantInt::get(PointerSizedIntTy, CurPos * sizeof(AOTInfo)),
         ConstantInt::get(PointerSizedIntTy, LiveValues.size())});

    // Store the stackmap address and length in a separate struct to save
    // arguments.
    AllocaInst *StackMapStruct = createAndFillStruct(
        FailBuilder, {JITFunc->getArg(JITFUNC_ARG_STACKMAP_ADDR_IDX),
                      JITFunc->getArg(JITFUNC_ARG_STACKMAP_LEN_IDX)});

    // Create the deoptimization call.
    Type *retty = PointerType::get(Context, 0);
    Function *DeoptInt = Intrinsic::getDeclaration(
        JITFunc->getParent(), Intrinsic::experimental_deoptimize, {retty});
    OperandBundleDef ob =
        OperandBundleDef("deopt", (ArrayRef<Value *>)LiveValues);
    // We already passed the stackmap address and size into the trace
    // function so pass them on to the __llvm_deoptimize call.
    CallInst *Ret =
        CallInst::Create(DeoptInt,
                         {StackMapStruct, AOTLocs, ActiveFramesStruct,
                          JITFunc->getArg(JITFUNC_ARG_FRAMEADDR_IDX)},
                         {ob}, "", GuardFailBB);

    // We always need to return after the deoptimisation call.
    ReturnInst::Create(Context, Ret, GuardFailBB);

    // Now that we've finished creating the guard, pop the current frame, so
    // that future guards don't include it in their list of active frames.
    ActiveFrames.pop_back();
    return GuardFailBB;
  }

  std::tuple<size_t, size_t, Instruction *, size_t> getYkCPAlloca() {
    Function *F = ((Instruction *)TraceInputs)->getFunction();
    BasicBlock &BB = F->getEntryBlock();
    size_t Idx = 0;
    for (auto I = BB.begin(); I != BB.end(); I++) {
      if (cast<Instruction>(I) == TraceInputs) {
        break;
      }
      Idx++;
    }
    assert(Idx < BB.size() &&
           "Could not find control point struct alloca in entry block.");
    return {0, Idx, cast<Instruction>(TraceInputs), 0};
  }

  void handleBranchingControlFlow(Instruction *I, size_t TraceIdx,
                                  Function *JITFunc, size_t CurBBIdx,
                                  size_t CurInstrIdx) {
    // First, peek ahead in the trace and retrieve the next block. We need this
    // so that we can insert an appropriate guard into the trace. A block must
    // exist at `InpTrace[TraceIdx + 1]` because the branch instruction must
    // transfer to a successor block, and branching cannot turn off tracing.
    assert(InpTrace[TraceIdx + 1].hasValue()); // Should be a mappable block.
    IRBlock NextIB = InpTrace[TraceIdx + 1].getValue();
    BasicBlock *NextBB;
    Function *NextFunc;
    std::tie(NextFunc, NextBB) = getLLVMAOTFuncAndBlock(&NextIB);

    // The branching instructions we are handling here are all transfer to a
    // block in the same function.
    assert(NextFunc == I->getFunction());

    BasicBlock *SuccBB = nullptr;
    if (isa<BranchInst>(I)) {
      SuccBB = handleBranchInst(JITFunc, NextBB, &*I, CurBBIdx, CurInstrIdx);
    } else if (isa<SwitchInst>(I)) {
      SuccBB = handleSwitchInst(JITFunc, NextBB, &*I, CurBBIdx, CurInstrIdx);
    } else {
      assert(isa<IndirectBrInst>(I));
      // It isn't necessary to copy the indirect branch into the `JITMod`
      // as the successor block is known from the trace. However, naively
      // not copying the branch would lead to dangling references in the
      // IR because the `address` operand typically (indirectly)
      // references AOT block addresses not present in the `JITMod`.
      // Therefore we also remove the IR instruction which defines the
      // `address` operand and anything which also becomes dead as a
      // result (recursively).
      Value *FirstOp = I->getOperand(0);
      assert(VMap.find(FirstOp) != VMap.end());
      DeleteDeadOnFinalise.push_back(VMap[FirstOp]);
      // FIXME: guards for indirect branches are not yet implemented.
      // https://github.com/ykjit/yk/issues/438
      abort();
    }

    // If a guard was emitted, then the block we had been building the trace
    // into will have been terminated (to check the guard condition) and we
    // should resume building the trace into the new guard success block.
    if (SuccBB != nullptr)
      Builder.SetInsertPoint(SuccBB);
  }

  void handleOperand(Value *Op) {
    if (VMap.find(Op) == VMap.end()) {
      // The operand is undefined in JITMod.
      Type *OpTy = Op->getType();

      // Variables allocated outside of the traced section must be passed into
      // the trace and thus must already have a mapping.
      assert(!isa<llvm::AllocaInst>(Op));

      if (isa<ConstantExpr>(Op)) {
        // A `ConstantExpr` may contain operands that require remapping, e.g.
        // global variables. Iterate over all operands and recursively call
        // `handleOperand` on them, then generate a new `ConstantExpr` with
        // the remapped operands.
        ConstantExpr *CExpr = cast<ConstantExpr>(Op);
        std::vector<Constant *> NewCEOps;
        for (unsigned CEOpIdx = 0; CEOpIdx < CExpr->getNumOperands();
             CEOpIdx++) {
          Value *CEOp = CExpr->getOperand(CEOpIdx);
          handleOperand(CEOp);
          NewCEOps.push_back(cast<Constant>(getMappedValue(CEOp)));
        }
        Constant *NewCExpr = CExpr->getWithOperands(NewCEOps);
        VMap[CExpr] = NewCExpr;
      } else if (isa<GlobalVariable>(Op)) {
        cloneGlobalVariable(Op);
      } else if ((isa<Constant>(Op)) || (isa<InlineAsm>(Op))) {
        if (isa<Function>(Op)) {
          // We are storing a function pointer in a variable, so we need to
          // redeclare the function in the JITModule in case it gets called.
          declareFunction(cast<Function>(Op));
        }
        // Constants and inline asm don't need to be mapped.
      } else if (Op == NewControlPointCall) {
        // The value generated by NewControlPointCall is the thread tracer.
        // At some optimisation levels, this gets stored in an alloca'd
        // stack space. Since we've stripped the instruction that
        // generates that value (from the JIT module), we have to make a
        // dummy stack slot to keep LLVM happy.
        Value *NullVal = Constant::getNullValue(OpTy);
        VMap[Op] = NullVal;
      } else {
        dumpValueAndExit("don't know how to handle operand", Op);
      }
    }
  }

  GlobalVariable *cloneGlobalVariable(Value *V) {
    GlobalVariable *OldGV = cast<GlobalVariable>(V);
    // We don't need to check if this global already exists, since
    // we're skipping any operand that's already been cloned into
    // the VMap.
    GlobalVariable *GV = new GlobalVariable(
        *JITMod, OldGV->getValueType(), OldGV->isConstant(),
        OldGV->getLinkage(), (Constant *)nullptr, OldGV->getName(),
        (GlobalVariable *)nullptr, OldGV->getThreadLocalMode(),
        OldGV->getType()->getAddressSpace());
    VMap[OldGV] = GV;
    if (OldGV->isConstant()) {
      GV->copyAttributesFrom(&*OldGV);
      cloned_globals.push_back(OldGV);
    }
    return GV;
  }

  void copyInstruction(IRBuilder<> *Builder, Instruction *I, size_t CurBBIdx,
                       size_t CurInstrIdx) {
    // Before copying an instruction, we have to scan the instruction's
    // operands checking that each is defined in JITMod.
    for (unsigned OpIdx = 0; OpIdx < I->getNumOperands(); OpIdx++) {
      Value *Op = I->getOperand(OpIdx);
      handleOperand(Op);
    }

    // Shortly we will copy the instruction into the JIT module. We start by
    // cloning the instruction.
    auto NewInst = &*I->clone();

    // Since the instruction operands still reference values from the AOT
    // module, we must remap them to point to new values in the JIT module.
    llvm::RemapInstruction(NewInst, VMap, RF_NoModuleLevelChanges);
    VMap[&*I] = NewInst;
    insertAOTMap(I, NewInst, CurBBIdx, CurInstrIdx);

    // Copy over any debugging metadata required by the instruction.
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> metadataList;
    I->getAllMetadata(metadataList);
    for (auto MD : metadataList) {
      NewInst->setMetadata(
          MD.first,
          MapMetadata(MD.second, VMap, llvm::RF_ReuseAndMutateDistinctMDs));
    }

    // And finally insert the new instruction into the JIT module.
    Builder->Insert(NewInst);
  }

  // Finalise the JITModule by adding a return instruction and initialising
  // global variables.
  void finalise(Module *AOTMod, IRBuilder<> *Builder) {
    // Now that we've seen all possible uses of values in the JITMod, we can
    // delete the values we've marked dead (and possibly their dependencies if
    // they too turn out to be dead).
    for (auto &V : DeleteDeadOnFinalise)
      deleteDeadTransitive(V);

    // Fix initialisers/referrers for copied global variables.
    // FIXME Do we also need to copy Linkage, MetaData, Comdat?
    for (GlobalVariable *G : cloned_globals) {
      GlobalVariable *NewGV = cast<GlobalVariable>(VMap[G]);
      if (G->isDeclaration())
        continue;

      if (G->hasInitializer())
        NewGV->setInitializer(MapValue(G->getInitializer(), VMap));
    }

    // Ensure that the JITModule has a `!llvm.dbg.cu`.
    // This code is borrowed from LLVM's `cloneFunction()` implementation.
    // OPT: Is there a faster way than scanning the whole module?
    DebugInfoFinder DIFinder;
    DIFinder.processModule(*AOTMod);
    if (DIFinder.compile_unit_count()) {
      auto *NMD = JITMod->getOrInsertNamedMetadata("llvm.dbg.cu");
      SmallPtrSet<const void *, 8> Visited;
      for (auto *Operand : NMD->operands())
        Visited.insert(Operand);
      for (auto *Unit : DIFinder.compile_units())
        if (Visited.insert(Unit).second)
          NMD->addOperand(Unit);
    }
  }

  // Determines if the LLVM values `V1` and `V2` are instructions defined
  // within the same LLVM `BasicBlock`. `V1` and `V2` must both be an instance
  // of `Instruction`.
  bool areInstrsDefinedInSameBlock(Value *V1, Value *V2) {
    assert(isa<Instruction>(V1) && isa<Instruction>(V2));
    return cast<Instruction>(V1)->getParent() ==
           cast<Instruction>(V2)->getParent();
  }

  // When executing the interpreter loop AOT code, the code before the control
  // point is executed, then the control point is called, then the code after
  // the control point is executed.
  //
  // But when we collect a trace, the first code we see is the code *after* the
  // call to the control point, then (assuming the interpreter loop doesn't
  // exit) we branch back to the start of the loop and only then see the code
  // before the call to the control point.
  //
  // In other words, there is a disparity between the order of the code in the
  // AOT module and in collected traces and this has implications for the trace
  // compiler. Without extra logic, alloca'd variables become undefined (as
  // they are defined outside of the trace) and thus need to be remapped to the
  // input of the compiled trace. SSA values (from the same block as the
  // control point) remain correct as phi nodes at the beginning of the trace
  // automatically select the appropriate input value.
  //
  // For example, once patched, a typical interpreter loop will look like this:
  //
  // clang-format off
  //
  // ```
  // bb0:
  //   %a = alloca  // Stack variable
  //   store 0, %a
  //   %b = 1       // Register variable
  //   %s = alloca YkCtrlPointVars
  //   br %bb1
  //
  // bb1:
  //   %b1 = phi [%b, %bb0], [%binc, %bb1]
  //   %aptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 0
  //   store %aptr, %a
  //   %bptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 1
  //   store %bptr, %b
  //   // traces end here
  //   call yk_new_control_point(%s)
  //   // traces start here
  //   %aptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 0
  //   %anew = load %aptr
  //   %bptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 1
  //   %bnew = load %bptr
  //
  //   %aload = load %anew
  //   %ainc = add 1, %aload
  //   store %ainc, %a
  //   %binc = add 1, %bnew
  //   br %bb1
  // ```
  //
  // clang-format on
  //
  // There are two live variables stored into the `YKCtrlPointVars` struct
  // before the call to the control point (`%a` and `%b`), and those variables
  // are loaded back out after the call to the control point (into `%anew` and
  // `%bnew`). `%a` and `%anew` correspond to the same high-level variable, and
  // so do `%b1` and `%bnew`. When assembling a trace from the above IR, it
  // would look like this:
  //
  // clang-format off
  //
  // ```
  // void compiled_trace(%YkCtrlPointVars* %s) {
  //   %aptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 0
  //   %anew = load %aptr
  //   %bptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 1
  //   %bnew = load %bptr
  //
  //   %aload = load %anew
  //   %ainc = add 1, %aload
  //   store %ainc, %a                // %a is undefined
  //   %binc = add 1, %bnew
  //   %b1 = %bbinc                   // RHS selected from PHI.
  //
  //   %aptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 0
  //   store %aptr, %a
  //   %bptr = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %s, i32 0, i32 1
  //   store %bptr, %b1
  //   ...
  // }
  // ```
  //
  // clang-format on
  //
  // Here `%a` is undefined because we didn't trace its allocation. Instead we
  // need to use the definition extracted from the `YkCtrlPointVars`, which
  // means we need to replace `%a` with `%anew` in the store instruction. The
  // other value `%b` doesn't have this problem, since the PHI node in the
  // control point block already makes sure it selects the correct SSA value
  // `%binc`.
  void createLiveIndexMap(Instruction *CPCI, Type *YkCtrlPointVarsPtrTy) {
    BasicBlock *CPCIBB = CPCI->getParent();

    // Scan for `getelementpointer`/`store` pairs leading up the control point.
    // For each pair we add an entry to `LiveIndexMap`.
    //
    // For example, this instruction pair:
    //
    // ```
    // %19 = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %3, i32 0, i32 2
    // store i32* %6, i32** %19, align 8
    // ```
    //
    // Adds an entry mapping the index `2` to `%6`.
    for (BasicBlock::iterator CI = CPCIBB->begin(); &*CI != CPCI; CI++) {
      assert(CI != CPCIBB->end());
      if (!isa<GetElementPtrInst>(CI))
        continue;

      GetElementPtrInst *GI = cast<GetElementPtrInst>(CI);
      if (GI->getPointerOperand() != TraceInputs)
        continue;

      // We have seen a lookup into the live variables struct, the succeeding
      // store instruction tells us which value is written into that field.
      Instruction *NextInst = &*std::next(CI);
      assert(isa<StoreInst>(NextInst));
      StoreInst *SI = cast<StoreInst>(NextInst);
      Value *StoredVal = SI->getValueOperand();
      Value *StoredAtIdxVal = *(std::next(GI->idx_begin()));
      assert(isa<ConstantInt>(StoredAtIdxVal));
      uint64_t StoredAtIdx = cast<ConstantInt>(StoredAtIdxVal)->getZExtValue();

      // We need an entry in this map for any live variable that isn't defined
      // by a PHI node at the top of the block cotaining the call to the
      // control point.
      if (!(isa<PHINode>(StoredVal) &&
            areInstrsDefinedInSameBlock(StoredVal, SI))) {
        LiveIndexMap[StoredAtIdx] = StoredVal;
      }
    }
  }

  // Find the call site to the (patched) control point and the type of the
  // struct used to pass in the live LLVM variables.
  //
  // FIXME: this assumes that there is a single call-site of the control point.
  // https://github.com/ykjit/yk/issues/479
  static tuple<CallInst *, Value *> GetControlPointInfo(Module *AOTMod) {
    Function *F = AOTMod->getFunction(YK_NEW_CONTROL_POINT);
    assert(F->arg_size() == YK_CONTROL_POINT_NUM_ARGS);

    User *CallSite = F->user_back();
    CallInst *CPCI = cast<CallInst>(CallSite);
    assert(CPCI->arg_size() == YK_CONTROL_POINT_NUM_ARGS);

    Value *Inputs = CPCI->getArgOperand(YK_CONTROL_POINT_ARG_VARS_IDX);
#ifndef NDEBUG
    Type *InputsTy = Inputs->getType();
    assert(InputsTy->isPointerTy());
#endif

    return {CPCI, Inputs};
  }

  // OPT: https://github.com/ykjit/yk/issues/419
  JITModBuilder(Module *AOTMod, char *FuncNames[], size_t BBs[],
                size_t TraceLen, char *FAddrKeys[], void *FAddrVals[],
                size_t FAddrLen, CallInst *CPCI, Value *TraceInputs)
      : AOTMod(AOTMod), Builder(AOTMod->getContext()),
        InpTrace(FuncNames, BBs, TraceLen),
        FAddrs(FAddrKeys, FAddrVals, FAddrLen), TraceInputs(TraceInputs),
        ControlPointCallInst(CPCI) {
    LLVMContext &Context = AOTMod->getContext();
    JITMod = new Module("", Context);

    // Cache common types.
    IntTy = Type::getIntNTy(Context, sizeof(int) * CHAR_BIT);
    DataLayout DL(JITMod);
    PointerSizedIntTy = DL.getIntPtrType(Context);

    // Create a function inside of which we construct the IR for our trace.
    JITFunc = createJITFunc(TraceInputs, PointerType::get(Context, 0));
    auto DstBB = BasicBlock::Create(JITMod->getContext(), "entry", JITFunc);
    Builder.SetInsertPoint(DstBB);

    createLiveIndexMap(ControlPointCallInst, TraceInputs->getType());

    // Map the live variables struct used inside the trace to the corresponding
    // argument of the compiled trace function.
    VMap[TraceInputs] = JITFunc->getArg(JITFUNC_ARG_INPUTS_STRUCT_IDX);

    LastCompletedBlocks.push_back(nullptr);
  }

public:
  // Store virtual addresses for called functions.
  std::map<GlobalValue *, void *> GlobalMappings;
  // The function name of this trace.
  string TraceName;
  // Mapping from AOT instructions to JIT instructions.
  ValueToValueMapTy VMap;
  // Heap allocated memory storing the corresponding AOT values for currently
  // live JIT values.
  AOTInfo *LiveAOTArray = nullptr;
  size_t LiveAOTNum = 0;

  JITModBuilder(JITModBuilder &&);

  static JITModBuilder Create(Module *AOTMod, char *FuncNames[], size_t BBs[],
                              size_t TraceLen, char *FAddrKeys[],
                              void *FAddrVals[], size_t FAddrLen) {
    CallInst *CPCI;
    Value *TI;
    std::tie(CPCI, TI) = GetControlPointInfo(AOTMod);
    return JITModBuilder(AOTMod, FuncNames, BBs, TraceLen, FAddrKeys, FAddrVals,
                         FAddrLen, CPCI, TI);
  }

#ifdef YK_TESTING
  static JITModBuilder CreateMocked(Module *AOTMod, char *FuncNames[],
                                    size_t BBs[], size_t TraceLen,
                                    char *FAddrKeys[], void *FAddrVals[],
                                    size_t FAddrLen) {
    LLVMContext &Context = AOTMod->getContext();

    // The trace compiler expects to be given a) a call to a control point, and
    // b) a struct containing live variables.
    //
    // For the trace compiler tests, we don't want the user to have to worry
    // about that stuff, so we fobb off the trace compiler with dummy versions
    // of those things.
    //
    // First, in order for a) and b) to exist, they need a parent function to
    // live in. We inject a never-called dummy function.
    llvm::FunctionType *FuncType =
        llvm::FunctionType::get(Type::getVoidTy(Context), {}, false);
    llvm::Function *Func = llvm::Function::Create(
        FuncType, Function::InternalLinkage, "__yk_tc_tests_dummy", AOTMod);
    BasicBlock *BB = BasicBlock::Create(Context, "", Func);
    IRBuilder<> Builder(BB);

    // Now we make a struct that we pretend contains the values of the live
    // variables at the time of the control point. It must have at least one
    // field, or LLVM chokes because it is unsized.
    Type *TraceInputsTy =
        StructType::create(AOTMod->getContext(), Type::getInt8Ty(Context));
    Value *TraceInputs = Builder.CreateAlloca(TraceInputsTy, 0, "");

    // Now we make a call instruction, which tell the trace compiler is a call
    // to the control point. It's actually a recursive call to the dummy
    // function.
    CallInst *CPCI = Builder.CreateCall(Func, {});
    Builder.CreateUnreachable();

    // Populate the function address map with dummy entries for all of the
    // functions in the AOT module, so that the trace compiler can outline
    // calls to them if neccessary.
    //
    // The actual addresses inserted don't matter, as the trace compiler suite
    // only compiles traces (without executing them).
    std::vector<char *> NewFAddrKeys;
    std::vector<void *> NewFAddrVals;
    for (Function &F : AOTMod->functions()) {
      NewFAddrKeys.push_back(const_cast<char *>(F.getName().data()));
      NewFAddrVals.push_back((void *)YK_INVALID_ALIGNED_VADDR);
    }

    JITModBuilder JB(AOTMod, FuncNames, BBs, TraceLen, &NewFAddrKeys[0],
                     &NewFAddrVals[0], NewFAddrKeys.size(), CPCI, TraceInputs);

    // Trick the trace compiler into thinking that it has already seen the call
    // to the control point, so that it starts copying instructions into JITMod
    // straight away.
    JB.NewControlPointCall = CPCI;
    JB.ExpectUnmappable = true;

    return JB;
  }
#endif

  // Generate the JIT module by "glueing together" blocks that the trace
  // executed in the AOT module.
  Module *createModule() {
    // Initialise the memory block holding live AOT values for guard failures.
    BasicBlock *NextCompletedBlock = nullptr;
    for (size_t Idx = 0; Idx < InpTrace.Length(); Idx++) {
      Optional<IRBlock> MaybeIB = InpTrace[Idx];
      if (ExpectUnmappable && !MaybeIB.hasValue()) {
        ExpectUnmappable = false;
        continue;
      }
      assert(MaybeIB.hasValue());
      IRBlock IB = MaybeIB.getValue();
      size_t CurBBIdx = IB.BBIdx;

      Function *F;
      BasicBlock *BB;
      std::tie(F, BB) = getLLVMAOTFuncAndBlock(&IB);

      assert(LastCompletedBlocks.size() >= 1);
      LastCompletedBlocks.back() = NextCompletedBlock;
      NextCompletedBlock = BB;

      // Iterate over all instructions within this block and copy them over
      // to our new module.
      for (size_t CurInstrIdx = 0; CurInstrIdx < BB->size(); CurInstrIdx++) {
        // If we've returned from a call, skip ahead to the instruction where
        // we left off.
        if (ResumeAfter.hasValue() != 0) {
          // If we find ourselves resuming in a block other than the one we
          // expected, then the compiler has changed the block structure. For
          // now we are disabling fallthrough optimisations in ykllvm to
          // prevent this from happening.
          assert(std::get<1>(ResumeAfter.getValue())->getParent() == BB);
          CurInstrIdx = std::get<0>(ResumeAfter.getValue()) + 1;
          ResumeAfter.reset();
        }
        auto I = BB->begin();
        std::advance(I, CurInstrIdx);
        assert(I != BB->end());

#ifdef YK_TESTING
        // In trace compiler tests, blocks may be terminated with an
        // `unreachable` terminator.
        if (isa<UnreachableInst>(I))
          break;
#endif

        // Skip calls to debug intrinsics (e.g. @llvm.dbg.value). We don't
        // currently handle debug info and these "pseudo-calls" cause our blocks
        // to be prematurely terminated.
        if (isa<DbgInfoIntrinsic>(I))
          continue;

        if (isa<CallInst>(I)) {
          if (isa<IntrinsicInst>(I)) {
            Intrinsic::ID IID = cast<CallBase>(I)->getIntrinsicID();

            // `llvm.lifetime.start.*` and `llvm.lifetime.end.*` are
            // annotations added by some compiler front-ends to allow backends
            // to perform stack slot optimisations.
            //
            // Removing these annotations makes generated code slightly less
            // efficient, but does not affect correctness, so we remove them to
            // make our lives easier.
            //
            // OPT: Consider leaving the annotations in, or generating our own
            // annotations, so that our compiled traces can take advantage of
            // stack slot optimisations.
            if ((IID == Intrinsic::lifetime_start) ||
                (IID == Intrinsic::lifetime_end))
              continue;

            // This intrinsic is used in AOTMod to pass the current frame
            // address into the control point which is required for stack
            // reconstruction. There's no use for this inside JITMod, so just
            // ignore it.
            if (IID == Intrinsic::frameaddress)
              continue;

            // Calls to `llvm.experimental.stackmap` are not really calls and
            // they generate no code anyway. We can skip them.
            if (IID == Intrinsic::experimental_stackmap) {
              LastSMCall = cast<CallBase>(I);
              continue;
            }

            // Whitelist intrinsics that appear to be always inlined.
            if (IID == Intrinsic::vastart || IID == Intrinsic::vaend ||
                IID == Intrinsic::smax ||
                IID == Intrinsic::usub_with_overflow) {
              if (NewControlPointCall != nullptr && !isOutlining())
                copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                CurInstrIdx);
              continue;
            }

            // Any intrinsic call which may generate machine code must have
            // metadata attached that specifies whether it has been inlined or
            // not.
            MDNode *IMD = I->getMetadata("yk.intrinsic.inlined");
            if (IMD == nullptr) {
              dumpValueAndExit(
                  "instrinsic is missing `yk.intrinsic.inlined` metadata", &*I);
            }
            ConstantAsMetadata *CAM =
                cast<ConstantAsMetadata>(IMD->getOperand(0));
            if (CAM->getValue()->isOneValue()) {
              // The intrinsic was inlined so we don't need to expect an
              // unmappable block and thus can just copy the call instruction
              // and continue processing the current block.
              if (NewControlPointCall != nullptr && !isOutlining())
                copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                CurInstrIdx);
              continue;
            }
            // The intrinsic wasn't inlined so we let the following code handle
            // it which already knows how to deal with such cases.
          }

          CallInst *CI = cast<CallInst>(I);
          Function *CF = CI->getCalledFunction();
          if (CF == nullptr) {
            if (NewControlPointCall == nullptr) {
              continue;
            }
            // The target isn't statically known, so we can't inline the
            // callee.
            if (!isa<InlineAsm>(CI->getCalledOperand())) {
              // Look ahead in the trace to find the callee so we can
              // map the arguments if we are inlining the call.
              Optional<IRBlock> MaybeNextIB = InpTrace[Idx + 1];
              if (MaybeNextIB.hasValue()) {
                CF = AOTMod->getFunction(MaybeNextIB.getValue().FuncName);
              } else {
                CF = nullptr;
              }
              // FIXME Don't inline indirect calls unless promoted.
              handleCallInst(CI, CF, CurBBIdx, CurInstrIdx);
              break;
            }
          } else if (CF->getName() == YK_NEW_CONTROL_POINT) {
            ExpectUnmappable = true; // control point is always opaque.
            if (NewControlPointCall == nullptr) {
              NewControlPointCall = &*CI;
            } else {
              assert(CF->arg_size() == YK_CONTROL_POINT_NUM_ARGS);
              VMap[CI] = ConstantPointerNull::get(
                  Type::getInt8PtrTy(JITMod->getContext()));
              ResumeAfter = make_tuple(CurInstrIdx, CI);
              break;
            }
            continue;
          } else if (NewControlPointCall != nullptr) {
            handleCallInst(CI, CF, CurBBIdx, CurInstrIdx);
            break;
          }
        }

        // We don't start copying instructions into the JIT module until we've
        // seen the call to YK_NEW_CONTROL_POINT.
        if (NewControlPointCall == nullptr)
          continue;

        if (isa<ReturnInst>(I)) {
          handleReturnInst(&*I, CurBBIdx, CurInstrIdx);
          break;
        }

        if (isOutlining()) {
          // We are currently ignoring an inlined function.
          continue;
        }

        if ((isa<BranchInst>(I)) || (isa<IndirectBrInst>(I)) ||
            (isa<SwitchInst>(I))) {
          handleBranchingControlFlow(&*I, Idx, JITFunc, CurBBIdx, CurInstrIdx);
          break;
        }

        if (isa<PHINode>(I)) {
          assert(LastCompletedBlocks.size() >= 1);
          handlePHINode(&*I, LastCompletedBlocks.back(), CurBBIdx, CurInstrIdx);
          continue;
        }

        // If execution reaches here, then the instruction I is to be copied
        // into JITMod.
        copyInstruction(&Builder, (Instruction *)&*I, CurBBIdx, CurInstrIdx);

        // If we see a `getelementpointer`/`load` pair that is loading from the
        // `YkCtrlPointVars` pointer, then we have to update the `VMap` using
        // the information we previously computed in `LiveIndexMap`. See
        // comments above about `LiveIndexMap`.
        if (isa<LoadInst>(I)) {
          LoadInst *LI = cast<LoadInst>(I);
          Value *LoadOper = LI->getPointerOperand();
          if (isa<GetElementPtrInst>(LoadOper)) {
            GetElementPtrInst *GI = cast<GetElementPtrInst>(LoadOper);
            if (GI->getPointerOperand() == TraceInputs) {
              Value *LoadedFromIdxVal = *(std::next(GI->idx_begin()));
              assert(isa<ConstantInt>(LoadedFromIdxVal));
              uint64_t LoadedFromIdx =
                  cast<ConstantInt>(LoadedFromIdxVal)->getZExtValue();
              Value *NewMapVal = LiveIndexMap[LoadedFromIdx];
              VMap[NewMapVal] = getMappedValue(LI);
            }
          }
        }
      }
    }

    // Recursive calls must have completed by the time we finish constructing
    // the trace.
    assert(!isOutlining());

    // If the trace succeeded return a null pointer instead of a reconstructed
    // frame address.
    Builder.CreateRet(
        ConstantPointerNull::get(PointerType::get(JITMod->getContext(), 0)));
    finalise(AOTMod, &Builder);
    return JITMod;
  }
};

tuple<Module *, string, std::map<GlobalValue *, void *>, void *>
createModule(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t TraceLen,
             char *FAddrKeys[], void *FAddrVals[], size_t FAddrLen) {
  JITModBuilder JB = JITModBuilder::Create(AOTMod, FuncNames, BBs, TraceLen,
                                           FAddrKeys, FAddrVals, FAddrLen);
  auto JITMod = JB.createModule();
  return make_tuple(JITMod, std::move(JB.TraceName),
                    std::move(JB.GlobalMappings), JB.LiveAOTArray);
}

#ifdef YK_TESTING
tuple<Module *, string, std::map<GlobalValue *, void *>, void *>
createModuleForTraceCompilerTests(Module *AOTMod, char *FuncNames[],
                                  size_t BBs[], size_t TraceLen,
                                  char *FAddrKeys[], void *FAddrVals[],
                                  size_t FAddrLen) {
  JITModBuilder JB = JITModBuilder::CreateMocked(
      AOTMod, FuncNames, BBs, TraceLen, FAddrKeys, FAddrVals, FAddrLen);

  auto JITMod = JB.createModule();

  // When the trace compiler encounters a non-const global in a trace, it
  // inserts an LLVM `global external` variable referencing the variable in the
  // interpreter's address space. In the trace compiler tests, such global
  // variables don't actually exist so we will get symbol resolution errors
  // when generating code for a trace.
  //
  // To avoid this, we insert a dummy address for all global variables in the
  // JITMod which are external references (have no initialiser), and which
  // don't already have a known address in the `GlobalMappings` map. Since we
  // don't actually plan to execute the trace, their address is
  // inconsequential. We just need it to compile.
  for (GlobalVariable &G : JITMod->globals()) {
    if ((!G.hasInitializer()) &&
        (JB.GlobalMappings.find(&G) == JB.GlobalMappings.end())) {
      JB.GlobalMappings.insert({&G, (void *)YK_INVALID_ALIGNED_VADDR});
    }
  }

  // Provide a dummy implementation of `__llvm_optimize()`.
  //
  // Without this, traces will sometimes fail to compile.
  LLVMContext &Context = JITMod->getContext();
  llvm::FunctionType *DOFuncType =
      llvm::FunctionType::get(Type::getVoidTy(Context), {}, false);
  llvm::Function *DOFunc = llvm::Function::Create(
      DOFuncType, Function::ExternalLinkage, "__llvm_deoptimize", JITMod);
  BasicBlock *DOBB = BasicBlock::Create(Context, "", DOFunc);
  IRBuilder<> DOBuilder(DOBB);
  DOBuilder.CreateUnreachable();

  return make_tuple(JITMod, std::move(JB.TraceName),
                    std::move(JB.GlobalMappings), nullptr);
}
#endif
