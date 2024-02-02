// Classes and functions for constructing a new LLVM module from a trace.

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "jitmodbuilder.h"

#include <atomic>
#include <bit>
#include <err.h>
#include <sys/types.h>
#include <variant>

using namespace llvm;
using namespace std;

extern "C" size_t __yk_lookup_promote_usize();

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
#define JITFUNC_ARG_COMPILEDTRACE_IDX 1
#define JITFUNC_ARG_FRAMEADDR_IDX 2

const char *PromoteRecFnName = "__yk_promote";

#define YK_OUTLINE_FNATTR "yk_outline"

// The first two arguments of a stackmap call are it's id and shadow bytes and
// need to be skipped when scanning the operands for live values.
#define YK_STACKMAP_SKIP_ARGS 2

// Return value telling the caller of the compiled trace that no guard failure
// occurred and it's safe to continue running the fake trace stitching loop.
#define TRACE_RETURN_SUCCESS 0

// The name prefix used for blocks that are branched to when a guard succeeds.
#define GUARD_SUCCESS_BLOCK_NAME "guardsuccess"

const std::array<Intrinsic::ID, 5> AlwaysInlinedIntrinsics = {
    Intrinsic::ctpop, Intrinsic::smax, Intrinsic::usub_with_overflow,
    Intrinsic::vaend, Intrinsic::vastart};

// Dump an error message and an LLVM value to stderr and exit with failure.
void dumpValueAndExit(const char *Msg, Value *V) {
  errs() << Msg << ": ";
  V->print(errs());
  exit(EXIT_FAILURE);
}

// Searches the call stack for a specific function in order to detect recursion
// inside the trace.
bool callStackContainsFunction(std::vector<CallInst *> &CS, CallInst *CI) {
  Function *F = CI->getCalledFunction();
  if (CI->getFunction() == F) {
    // The called function and the caller are the same.
    return true;
  }
  for (Instruction *I : CS) {
    if (I->getFunction() == F) {
      return true;
    }
  }
  return false;
}

// A function name and basic block index pair that identifies a mappable block
// in the AOT LLVM IR.
struct IRBlock {
  // A non-null pointer to the function name.
  char *FuncName;
  // The index of the block in the parent LLVM function.
  size_t BBIdx;
};

// An unmappable region of code spanning one or more machine blocks.
struct UnmappableRegion {};

class TraceLoc {
  std::variant<IRBlock, UnmappableRegion> Loc;

public:
  TraceLoc(std::variant<IRBlock, UnmappableRegion> Loc) : Loc(Loc) {}

  UnmappableRegion *getUnmappableRegion() {
    return std::get_if<UnmappableRegion>(&Loc);
  }

  IRBlock *getMappedBlock() { return std::get_if<IRBlock>(&Loc); }

  void dump() {
    if (IRBlock *IRB = std::get_if<IRBlock>(&Loc)) {
      errs() << "IRBlock(Func=" << IRB->FuncName << ", BBIdx=" << IRB->BBIdx
             << ")\n";
    } else {
#ifndef NDEBUG
      UnmappableRegion *U = std::get_if<UnmappableRegion>(&Loc);
      assert(U);
#endif
      errs() << "UnmappableRegion()\n";
    }
  }
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
  TraceLoc operator[](size_t Idx) {
    assert(Idx < Len);
    char *FuncName = FuncNames[Idx];
    if (FuncName == nullptr) {
      return TraceLoc(variant<IRBlock, UnmappableRegion>{UnmappableRegion{}});
    } else {
      return TraceLoc(
          variant<IRBlock, UnmappableRegion>{IRBlock{FuncName, BBs[Idx]}});
    }
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

// Struct to store a live AOT value.
struct AOTInfo {
  LLVMValueRef Value;
  size_t FrameIdx;
};

class JITModBuilder {
  // Global variables/functions that were copied over and need to be
  // initialised.
  vector<GlobalVariable *> cloned_globals;
  // The module being traced.
  Module *AOTMod;
  // The new module that is being build.
  Module *JITMod;
  // When true, the compiler is outlining.
  bool Outlining = false;
  // The call depth at which we started outlining. Once we reached the same
  // depth after returning from a call, we can stop outlining.
  size_t OutlineBase = 0;
  // Determines whether the last block in the trace was mappable or not.
  // Required to make assumptions about the control flow of the trace.
  bool LastBlockMappable = true;
  // The last basic block that was processed. Required to handle PHI nodes, and
  // for sanity checking control flow.
  BasicBlock *LastBB = nullptr;
  // The last instruction that was processed. Required to determine the control
  // flow of the trace, i.e. whether we are returning from call or not.
  Instruction *LastInst = nullptr;
  // The JITMod's builder.
  llvm::IRBuilder<> Builder;
  // Dead values to recursively delete upon finalisation of the JITMod. This is
  // required because it's not safe to recursively delete values in the middle
  // of creating the JIT module. We don't know if any of those values might be
  // required later in the trace.
  vector<Value *> DeleteDeadOnFinalise;

  // Information about the trace we are compiling.
  InputTrace InpTrace;

  // The LLVM type for a C `int` on the current machine.
  Type *IntTy;

  // A pointer-sized integer (i.e. a Rust `usize`) for the current machine.
  IntegerType *PointerSizedIntTy;

  // A pointer to the Value representing the trace input struct.
  Value *TraceInputs;

  // A pointer to the instruction that calls the patched control point.
  CallInst *ControlPointCallInst;

  // The entry block for trace looping.
  BasicBlock *LoopEntryBB = nullptr;

  // The function inside which we build the IR for the trace.
  Function *JITFunc;

  // Tracks stackmap call instructions of currently active frames (excluding
  // the current frame).
  std::vector<CallInst *> CallStack;

  // Set to true for a side-trace or false for a normal trace.
  bool IsSideTrace = false;

  Value *getMappedValue(Value *V) {
    if (VMap.find(V) != VMap.end()) {
      return VMap[V];
    }
    assert(isa<Constant>(V));
    return V;
  }

  // Add an external declaration for the given function to JITMod.
  void declareFunction(Function *F) {
    assert(JITMod->getFunction(F->getName()) == nullptr);
    auto DeclFunc = llvm::Function::Create(F->getFunctionType(),
                                           GlobalValue::ExternalLinkage,
                                           F->getName(), JITMod);
    VMap[F] = DeclFunc;
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
      // The definition of the callee is external to AOTMod. It will be
      // outlined, but we still need to declare it locally if we have not
      // done so yet.
      if (CF != nullptr && VMap.find(CF) == VMap.end()) {
        declareFunction(CF);
      }
      if (!Outlining) {
        copyInstruction(&Builder, (Instruction *)&*CI, CurBBIdx, CurInstrIdx);
        OutlineBase = CallStack.size();
        Outlining = true;
      }
    } else {
      // Calling to a non-foreign function.
      if (!Outlining) {
        // We are not outlining, but should this call start us outlining?
        if (CF->hasFnAttribute(YK_OUTLINE_FNATTR) || CF->isVarArg() ||
            callStackContainsFunction(CallStack, CI)) {
          // We will outline this call.
          //
          // If this is a recursive call that has been inlined, or if the callee
          // has the "yk_outline" annotation, remove the inlined code and turn
          // it into a normal (outlined) call.
          if (VMap.find(CF) == VMap.end()) {
            declareFunction(CF);
          }
          copyInstruction(&Builder, CI, CurBBIdx, CurInstrIdx);
          OutlineBase = CallStack.size();
          Outlining = true;
        } else {
          // Otherwise keep the call inlined.
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
      }
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
        BI->getParent(), CurBBIdx, I->getPrevNode(), CurInstrIdx, GuardCount);
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
    BasicBlock *FailBB =
        getGuardFailureBlock(SI->getParent(), CurBBIdx, I->getPrevNode(),
                             CurInstrIdx, GuardCount, 1);
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
    // Check if we have arrived back at the frame where outlining started.
    if (Outlining) {
      return;
    }

    // Replace the return variable of the call with its return value.
    // Since the return value will have already been copied over to the
    // JITModule, make sure we look up the copy.
    auto OldRetVal = ((ReturnInst *)&*I)->getReturnValue();
    if (OldRetVal != nullptr) {
      Instruction *PrevCall = CallStack.back()->getPrevNode();
      assert(isa<CallInst>(PrevCall));
      Value *JIT = getMappedValue(OldRetVal);
      VMap[PrevCall] = JIT;
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
    // during deoptimisation. Unfortunately for us, LLVM doesn't have a
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
  }

  Function *createJITFunc(Value *TraceInputs, Type *FrameAddr) {
    // Compute a name for the trace.
    uint64_t TraceIdx = getNewTraceIdx();
    TraceName = string(TRACE_FUNC_PREFIX) + to_string(TraceIdx);

    // Create the function.
    std::vector<Type *> InputTypes;

    // Add YkCtrlPointVars argument.
    InputTypes.push_back(TraceInputs->getType());

    // Add *CompiledTrace argument.
    InputTypes.push_back(PointerSizedIntTy->getPointerTo());

    // Add argument for pointer to the frame containing the control point.
    InputTypes.push_back(FrameAddr);

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
  BasicBlock *getGuardFailureBlock(BasicBlock *CurBB, size_t CurBBIdx,
                                   Instruction *SMCall, size_t CurInstrIdx,
                                   size_t GuardId, size_t IsSwitch = 0) {
    // If `JITFunc` contains no blocks already, then the guard failure block
    // becomes the entry block. This would lead to a trace that
    // unconditionally and immediately fails a guard.
    assert(JITFunc->size() != 0);
    // Keep a count of he current number of guards. This number is used as an
    // index into `CompiledTrace.guards` to handle side-traces inside guard
    // failures.
    GuardCount += 1;

    LLVMContext &Context = JITFunc->getContext();

    // Create the block.
    // FIXME: Cache guard blocks where live variables and frames are the same,
    // e.g. this can happen when a loop is unrolled and the same condition
    // produces a guard in each unrolled iteration.
    BasicBlock *GuardFailBB = BasicBlock::Create(Context, "guardfail", JITFunc);
    IRBuilder<> FailBuilder(GuardFailBB);

    // Clone the CallStack, store it on the heap, and hardcode its pointer into
    // the deoptimize call. We'll need this call stack for side-tracing to
    // initialise JITModBuilder in a way that allows us to start tracing from
    // locations not immediately after the control point.
    std::vector<CallInst *> *NewCallStack = new std::vector(CallStack);
    // Add the stackmap belonging to the branch we are guarding here.
    NewCallStack->push_back(cast<CallInst>(SMCall));

    // Create a vector of active stackframes (i.e. basic block index,
    // instruction index, function name). This will be needed later for
    // reconstructing the stack after deoptimisation.
    AllocaInst *ActiveFrameVec = FailBuilder.CreateAlloca(
        PointerSizedIntTy,
        ConstantInt::get(PointerSizedIntTy, NewCallStack->size()));

    std::vector<std::tuple<Value *, Value *, size_t>> LiveValues;
    for (size_t I = 0; I < NewCallStack->size(); I++) {
      // Read live AOTMod values from stackmap calls and find their
      // corresponding values in JITMod. These are exactly the values that are
      // live at each guard failure and need to be deoptimised.
      CallInst *SMC = (*NewCallStack)[I];
      assert(SMC && isa<CallInst>(SMC));
      assert(cast<CallBase>(SMC)->getIntrinsicID() ==
             Intrinsic::experimental_stackmap);
      for (size_t Idx = YK_STACKMAP_SKIP_ARGS; Idx < SMC->arg_size(); Idx++) {
        Value *Arg = SMC->getArgOperand(Idx);
        if (Arg == ControlPointCallInst) {
          // There's no corresponding JIT value for the return value of the
          // control point call, so just skip it.
          continue;
        }
        if (VMap.find(Arg) != VMap.end()) {
          Value *JITArg = VMap[Arg];
          if (!isa<Constant>(JITArg)) {
            LiveValues.push_back({JITArg, Arg, I});
          }
        }
      }

      // Store the current program counter (instruction) for each active frame.
      auto GEP =
          FailBuilder.CreateGEP(PointerSizedIntTy, ActiveFrameVec,
                                {
                                    ConstantInt::get(PointerSizedIntTy, I),
                                });
      uintptr_t InstrPtr = reinterpret_cast<uintptr_t>(llvm::wrap(SMC));
      FailBuilder.CreateStore(ConstantInt::get(PointerSizedIntTy, InstrPtr),
                              GEP);
    }

    // Store the active frames vector and its length in a separate struct to
    // save arguments.
    AllocaInst *ActiveFramesStruct = createAndFillStruct(
        FailBuilder, {ActiveFrameVec, ConstantInt::get(PointerSizedIntTy,
                                                       NewCallStack->size())});

    // Make more space to store the locations of the corresponding live AOT
    // values for this guard failure.
    size_t CurPos = LiveAOTNum;
    LiveAOTNum += LiveValues.size();
    LiveAOTArray = static_cast<AOTInfo *>(
        reallocarray(LiveAOTArray, LiveAOTNum, sizeof(AOTInfo)));
    assert(LiveAOTArray != NULL);
    // Get a pointer to this guard failure's region in the memory block.
    AOTInfo *CurrentRegion = &LiveAOTArray[CurPos];

    std::vector<Value *> DeoptLives;
    for (size_t I = 0; I < LiveValues.size(); I++) {
      auto [Live, AOTVar, StackFrameIdx] = LiveValues[I];
      CurrentRegion[I] = {llvm::wrap(AOTVar), StackFrameIdx};
      DeoptLives.push_back(Live);
    }

    // Store the offset and length of the live AOT variables.
    AllocaInst *AOTLocs = createAndFillStruct(
        FailBuilder,
        {ConstantInt::get(PointerSizedIntTy, CurPos * sizeof(AOTInfo)),
         ConstantInt::get(PointerSizedIntTy, DeoptLives.size())});

    // Create the deoptimization call.
    Type *retty = PointerType::get(Context, 0);
    Function *DeoptInt = Intrinsic::getDeclaration(
        JITFunc->getParent(), Intrinsic::experimental_deoptimize, {retty});
    OperandBundleDef ob =
        OperandBundleDef("deopt", (ArrayRef<Value *>)DeoptLives);
    // We already passed the stackmap address and size into the trace
    // function so pass them on to the __llvm_deoptimize call.
    CallInst *Ret = CallInst::Create(
        DeoptInt,
        {JITFunc->getArg(JITFUNC_ARG_COMPILEDTRACE_IDX),
         JITFunc->getArg(JITFUNC_ARG_FRAMEADDR_IDX), AOTLocs,
         ActiveFramesStruct, ConstantInt::get(PointerSizedIntTy, GuardId),
         ConstantInt::get(PointerSizedIntTy, (size_t)NewCallStack),
         ConstantInt::get(PointerSizedIntTy, IsSwitch)},
        {ob}, "", GuardFailBB);

    // We always need to return after the deoptimisation call.
    ReturnInst::Create(Context, Ret, GuardFailBB);
    return GuardFailBB;
  }

  void handleBranchingControlFlow(Instruction *I, size_t TraceIdx,
                                  Function *JITFunc, size_t CurBBIdx,
                                  size_t CurInstrIdx) {
    // First, peek ahead in the trace and retrieve the next block. We need this
    // so that we can insert an appropriate guard into the trace. A block must
    // exist at `InpTrace[TraceIdx + 1]` because the branch instruction must
    // transfer to a successor block, and branching cannot turn off tracing.
    assert(TraceIdx + 1 < InpTrace.Length());
    IRBlock *NextIB = InpTrace[TraceIdx + 1].getMappedBlock();
    assert(NextIB);
    BasicBlock *NextBB;
    Function *NextFunc;
    std::tie(NextFunc, NextBB) = getLLVMAOTFuncAndBlock(NextIB);

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
      // If this is a vector containing other constants, we need to clone
      // those as well.
      Type *Ty = OldGV->getValueType();
      Constant *C = OldGV->getInitializer();
      if (Ty->isArrayTy()) {
        if ((cast<ArrayType>(Ty))->getElementType()->isPointerTy()) {
          ConstantArray *GVA = dyn_cast<ConstantArray>(C);
          for (size_t I = 0; I < Ty->getArrayNumElements(); I++) {
            Constant *Elem = GVA->getAggregateElement(I);
            if (isa<GlobalVariable>(Elem) && VMap.count(Elem) == 0) {
              cloneGlobalVariable(Elem);
            }
          }
        }
      }
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

  // At the beginning of the trace we need to extract the live values from
  // YkCtrlPointVars and use those inplace of their original AOT values. We can
  // do this by looking up the value that was stored inside YkCtrlPointVars at
  // AOT time, and then emit the neccessary loads to replace them with.
  //
  // For example the following AOT code:
  //
  // clang-format off
  // ```
  // $1 = 2
  // store $1, YkCtrlPointVars, 0, 0
  // control_point(YkCtrlPointVars)
  // add $1, 1
  // ```
  //
  // would result in the following JIT code:
  //
  // ```
  // void compiled_trace(YkCtrlPointVars $1) {
  //   $2 = load $1, 0, 0
  //   add $2, 1
  // }
  // ```
  // clang-format on
  void createTraceHeader(Instruction *CPCI, Type *YkCtrlPointVarsPtrTy) {
    BasicBlock *CPCIBB = CPCI->getParent();

    // Find the store instructions related to YkCtrlPointVars and generate and
    // insert a load for each stored value.
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

      // Now generate the load and update the VMap to reference this load
      // instead of the original AOT value.
      if (!(isa<PHINode>(StoredVal) &&
            areInstrsDefinedInSameBlock(StoredVal, SI))) {
        Instruction *GEP = Builder.Insert(GI->clone());
        llvm::RemapInstruction(GEP, VMap, RF_NoModuleLevelChanges);
        Instruction *Load = Builder.CreateLoad(StoredVal->getType(), GEP);
        VMap[StoredVal] = Load;
      }
    }

    // Now that we've generated the YkCtrlPointVars loads at the beginning of
    // the trace, create new block, so we don't run these loads on every loop
    // iteration.
    LoopEntryBB =
        BasicBlock::Create(Builder.getContext(), "loopentry", JITFunc);
    Builder.CreateBr(LoopEntryBB);
    Builder.SetInsertPoint(LoopEntryBB);
  }

  // Generate loads for the live variables passed into the side trace call.
  // Unlike the normal trace we can't loop back to the top of the side-trace
  // (the side-trace ends at the control point and this is where we need to
  // return to).
  void createSideTraceHeader(Instruction *CPCI, Type *YkCtrlPointVarsPtrTy,
                             void *AOTValsPtr, size_t AOTValsLen) {
    IntegerType *Int32Ty = Type::getInt32Ty(CPCI->getContext());
    AOTInfo *LiveVals = reinterpret_cast<AOTInfo *>(AOTValsPtr);
    for (size_t i = 0; i < AOTValsLen; i++) {
      AOTInfo Info = LiveVals[i];
      Value *V = llvm::unwrap(Info.Value);
      Value *GEP = Builder.CreateGEP(YkCtrlPointVarsPtrTy, JITFunc->getArg(0),
                                     {ConstantInt::get(Int32Ty, i)});
      Value *Load = Builder.CreateLoad(V->getType(), GEP);
      VMap[V] = Load;
    }
  }

  // Find the call site to the (patched) control point, the index of that call
  // site in the parent block, and the type of the struct used to pass in the
  // live LLVM variables.
  //
  // FIXME: this assumes that there is a single call-site of the control point.
  // https://github.com/ykjit/yk/issues/479
  static tuple<CallInst *, size_t, Value *>
  GetControlPointInfo(Module *AOTMod) {
    Function *F = AOTMod->getFunction(YK_NEW_CONTROL_POINT);
    assert(F->arg_size() == YK_CONTROL_POINT_NUM_ARGS);
    assert(F->getReturnType()->isVoidTy());

    User *CallSite = F->user_back();
    CallInst *CPCI = cast<CallInst>(CallSite);
    assert(CPCI->arg_size() == YK_CONTROL_POINT_NUM_ARGS);

    // Get the instruction index of CPCI in its parent block.
    size_t CPCIIdx = 0;
    assert(CPCI->getParent() != NULL);
    for (Instruction &I : *CPCI->getParent()) {
      if (&I == cast<Instruction>(CPCI)) {
        break;
      }
      CPCIIdx++;
    }

    Value *Inputs = CPCI->getArgOperand(YK_CONTROL_POINT_ARG_VARS_IDX);
    assert(Inputs->getType()->isPointerTy());

    return {CPCI, CPCIIdx, Inputs};
  }

  // OPT: https://github.com/ykjit/yk/issues/419
  JITModBuilder(Module *AOTMod, char *FuncNames[], size_t BBs[],
                size_t TraceLen, CallInst *CPCI,
                std::optional<std::tuple<size_t, CallInst *>> InitialResume,
                Value *TraceInputs, void *CallStackPtr, void *AOTValsPtr,
                size_t AOTValsLen)
      : AOTMod(AOTMod), Builder(AOTMod->getContext()),
        InpTrace(FuncNames, BBs, TraceLen), TraceInputs(TraceInputs),
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

    // Map the live variables struct used inside the trace to the corresponding
    // argument of the compiled trace function.
    VMap[TraceInputs] = JITFunc->getArg(JITFUNC_ARG_INPUTS_STRUCT_IDX);

    // If this is a side-trace (i.e. CallStackPtr is not null), then recast
    // pointer to heap allocated CallStack and clone it into this
    // JITModBuilder's CallStack.
    if (CallStackPtr) {
      CallStack = *reinterpret_cast<std::vector<CallInst *> *>(CallStackPtr);
      // The call stack we passed into deopt contains the current frame, as we
      // require that frame's stackmap for deoptimisation. However, when
      // building a side trace, that frame is equal to the frame we start
      // collection in, and is thus not needed on the stack, and can be
      // removed.
      LastBB = CallStack.back()->getParent();
      CallStack.pop_back();
    }

    // If aotvalsptr is not null then read the values out and generate loads for
    // each at the beginning of the function
    if (AOTValsPtr) {
      IsSideTrace = true;
      createSideTraceHeader(ControlPointCallInst, TraceInputs->getType(),
                            AOTValsPtr, AOTValsLen);
    } else {
      createTraceHeader(ControlPointCallInst, TraceInputs->getType());
    }

    // In debug builds, sanity check our assumptions about the input trace.
#ifndef NDEBUG
    size_t IL = InpTrace.Length();
    // The trace is not empty.
    assert(IL >= 1);
    // The trace never starts with unmappable code (because it gets stripped by
    // the maper).
    assert(InpTrace[0].getMappedBlock());
    // There should never be two unmappable blocks in a row in the trace
    // (because the mapper collapses them to save memory).
    for (size_t I = 0; I < IL - 1; I++) {
      assert(I + 1 < InpTrace.Length());
      assert(InpTrace[I].getMappedBlock() || InpTrace[I + 1].getMappedBlock());
    }
#endif
  }

public:
  // The function name of this trace.
  string TraceName;
  // Mapping from AOT instructions to JIT instructions.
  ValueToValueMapTy VMap;
  // Heap allocated memory storing the corresponding AOT values for currently
  // live JIT values.
  AOTInfo *LiveAOTArray = nullptr;
  size_t LiveAOTNum = 0;
  size_t GuardCount = 0;

  JITModBuilder(JITModBuilder &&);

  static JITModBuilder Create(Module *AOTMod, char *FuncNames[], size_t BBs[],
                              size_t TraceLen, void *CallStack,
                              void *AOTValsPtr, size_t AOTValsLen) {
    CallInst *CPCI;
    Value *TI;
    size_t CPCIIdx;
    std::tie(CPCI, CPCIIdx, TI) = GetControlPointInfo(AOTMod);
    return JITModBuilder(AOTMod, FuncNames, BBs, TraceLen, CPCI,
                         make_tuple(CPCIIdx, CPCI), TI, CallStack, AOTValsPtr,
                         AOTValsLen);
  }

  // Generate the JIT module by "glueing together" blocks that the trace
  // executed in the AOT module.
  Module *createModule() {
    size_t CurBBIdx;
    size_t CurInstrIdx;
    for (size_t Idx = 0; Idx < InpTrace.Length(); Idx++) {
      // Update the previously executed BB in the most-recent frame (if it's
      // mappable).
      TraceLoc Loc = InpTrace[Idx];

      if (UnmappableRegion *UR = Loc.getUnmappableRegion()) {
        LastBlockMappable = false;
        LastInst = nullptr;
        LastBB = nullptr;
        continue;
      }

      IRBlock *IB = Loc.getMappedBlock();
      assert(IB);
      CurBBIdx = IB->BBIdx;

      auto [F, BB] = getLLVMAOTFuncAndBlock(IB);

      // For outlining to function, we need to reliably detect recursive calls
      // and callbacks from unmappable blocks (i.e. external functions). Thanks
      // to two extra LLVM passes (one ensuring no calls can appear in entry
      // blocks, and another splitting blocks after calls) we can infer the
      // control flow from the type of block we see and the last instruction
      // that was processed. For example, if the last block was unmappable and
      // the current block is mappable and an entry block, we know that this
      // must be a callback from an external function. While the CallStack
      // isn't strictly needed to implement this, it is required for
      // deoptimisation as it collects the stackmap calls of inlined functions.
      // We thus use it here as a simple counter to keep track of the call
      // depths.
      if (BB->isEntryBlock()) {
        LastBB = nullptr;
        if (!LastBlockMappable) {
          // Unmappable code called back into mappable code.
          LastBlockMappable = true;
        } else {
          // A normal call from a mappable block into another mappable block.
          // Find the stackmap of the previous frame and add it to the
          // callstack.
          assert(LastInst && isa<CallInst>(LastInst));
          CallInst *SMC = cast<CallInst>(LastInst->getNextNode());
          CallStack.push_back(SMC);
        }
      } else {
        // If the last block was unmappable or the last instruction was a
        // return, then we are returning from a call. Since we've already
        // processed all instructions in this block, we can just skip it.
        if (!LastBlockMappable) {
          LastBlockMappable = true;
          LastBB = BB;
          if (CallStack.size() == OutlineBase) {
            Outlining = false;
            OutlineBase = 0;
          }
          continue;
        } else if (LastInst && isa<ReturnInst>(LastInst)) {
          LastInst = nullptr;
          assert(CallStack.back()->getParent() == BB);
          LastBB = BB;
          CallStack.pop_back();
          if (CallStack.size() == OutlineBase) {
            Outlining = false;
            OutlineBase = 0;
          }
          continue;
        }
      }

#ifndef NDEBUG
      // `BB` should be a successor of the last block executed in this frame.
      if (LastBB) {
        bool PredFound = false;
        for (BasicBlock *PBB : predecessors(BB)) {
          if (PBB == LastBB) {
            PredFound = true;
            break;
          }
        }
        assert(PredFound);
      }
#endif

      // The index of the instruction in `BB` that we are currently processing.
      CurInstrIdx = 0;

      // Iterate over all instructions within this block and copy them over
      // to our new module.
      for (; CurInstrIdx < BB->size(); CurInstrIdx++) {
        auto I = BB->begin();
        std::advance(I, CurInstrIdx);
        assert(I != BB->end());

        // Skip calls to debug intrinsics (e.g. @llvm.dbg.value). We don't
        // currently handle debug info and these "pseudo-calls" cause our blocks
        // to be prematurely terminated.
        if (isa<DbgInfoIntrinsic>(I))
          continue;

        LastInst = &*I;

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
              continue;
            }

            // Whitelist intrinsics that appear to be always inlined.
            if (std::find(AlwaysInlinedIntrinsics.begin(),
                          AlwaysInlinedIntrinsics.end(),
                          IID) != AlwaysInlinedIntrinsics.end()) {
              if (!Outlining) {
                copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                CurInstrIdx);
              }
              continue;
            }

            // Since blocks can only contains a single call, we can tell
            // whether this intrinsic was inlined at compile time by peeking at
            // the next block in the trace. If it is unmappable, this intrinsic
            // wasn't inlined and we need to copy the call into the JIT module.
            // FIXME: What to do about intrinsics that call back to mappable
            // code. As far as I can tell, there's only one intrinsics that
            // allows this: llvm.init.trampoline. Investigate if this is a
            // problem.
            if (Idx + 1 < InpTrace.Length()) {
              if (UnmappableRegion *UR =
                      InpTrace[Idx + 1].getUnmappableRegion()) {
                // The next block is unmappable, which means this intrinsic
                // wasn't inlined and we need to copy the call instruction.
                if (!Outlining) {
                  copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                  CurInstrIdx);
                }
                break;
              }
            }
          }

          CallInst *CI = cast<CallInst>(I);
          Function *CF = CI->getCalledFunction();
          if (CF == nullptr) {
            // The target isn't statically known, so we can't inline the
            // callee.
            if (!isa<InlineAsm>(CI->getCalledOperand())) {
              // Look ahead in the trace to find the callee so we can
              // map the arguments if we are inlining the call.
              assert(Idx + 1 < InpTrace.Length());
              TraceLoc MaybeNextIB = InpTrace[Idx + 1];
              if (const IRBlock *NextIB = MaybeNextIB.getMappedBlock()) {
                CF = AOTMod->getFunction(NextIB->FuncName);
              } else {
                CF = nullptr;
              }
              // FIXME Don't inline indirect calls unless promoted.
              handleCallInst(CI, CF, CurBBIdx, CurInstrIdx);
              break;
            } else {
              // This is an inlineasm instruction so just copy it. We don't
              // currently support any inline asm that isn't the empty string,
              // since calls within the asm messes up the blockmap.
              // FIXME: To fix this, we could parse the asm and count the
              // calls. However, parsing asm is complicated and even LLVM's
              // parser can bail. Alternatively, we could annotate blocks when
              // they contain inline asm and change the PT decoder so that it
              // disassembles those blocks to figure out how many calls the asm
              // contains.
              if (!cast<InlineAsm>(CI->getCalledOperand())
                       ->getAsmString()
                       .empty()) {
                errs() << "InlineAsm is currently not supported.";
                exit(EXIT_FAILURE);
              }
              copyInstruction(&Builder, (Instruction *)&*CI, CurBBIdx,
                              CurInstrIdx);
              break;
            }
          } else if (CF->getName() == PromoteRecFnName) {
            // A value is being promoted to a constant.
            handlePromote(CI, BB, CurBBIdx, CurInstrIdx);
            break;
          } else {
            StringRef S = CF->getName();
            if (S == "_setjmp" || S == "_longjmp") {
              // FIXME: We currently can't deal with traces containing
              // setjmp/longjmp, so for now simply abort this trace.
              // See: https://github.com/ykjit/yk/issues/610
              return nullptr;
            }
            handleCallInst(CI, CF, CurBBIdx, CurInstrIdx);
            break;
          }
        }

        if (isa<ReturnInst>(I)) {
          handleReturnInst(&*I, CurBBIdx, CurInstrIdx);
          break;
        }

        if (Outlining) {
          // We are currently ignoring an inlined function.
          continue;
        }

        if ((isa<BranchInst>(I)) || (isa<IndirectBrInst>(I)) ||
            (isa<SwitchInst>(I))) {
          handleBranchingControlFlow(&*I, Idx, JITFunc, CurBBIdx, CurInstrIdx);
          break;
        }

        if (isa<PHINode>(I)) {
          handlePHINode(&*I, LastBB, CurBBIdx, CurInstrIdx);
          continue;
        }

        if (Idx > 0 && Idx < InpTrace.Length() - 1) {
          // Stores into YkCtrlPointVars only need to be copied if they appear
          // at the beginning or end of the trace. Any YkCtrlPointVars stores
          // inbetween come from tracing over the control point and aren't
          // needed. We remove them here to reduce the size of traces.
          if (isa<GetElementPtrInst>(I)) {
            GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
            if (GEP->getPointerOperand() == TraceInputs) {
              // Collect stores
              std::vector<Value *> Stores;
              while (isa<GetElementPtrInst>(I)) {
                assert(cast<GetElementPtrInst>(I)->getPointerOperand() ==
                       TraceInputs);
                I++;
                CurInstrIdx++;
                assert(isa<StoreInst>(I));
                Stores.insert(Stores.begin(),
                              cast<StoreInst>(I)->getValueOperand());
                I++;
                CurInstrIdx++;
              }
              assert(InpTrace.Length() > 2);
              if (Idx == InpTrace.Length() - 2) {
                // Once we reached the YkCtrlPointVars stores at the end of the
                // trace, we're done. We don't need to copy those instructions
                // over, since all YkCtrlPointVars are stored on the shadow
                // stack.
                // FIXME: Once we allow more optimisations in AOT, some of the
                // YkCtrlPointVars won't be stored on the shadow stack anymore,
                // and we might have to insert PHI nodes into the loop entry
                // block.
                I++;
                CurInstrIdx++;
                assert(isa<CallInst>(I)); // control point call
                I++;
                CurInstrIdx++;
                assert(isa<CallInst>(I)); // stackmap call
                if (IsSideTrace) {
                  // We can't currently loop back to the parent trace when the
                  // side trace reaches its end (this requires patching the
                  // parent trace). Instead we simply guard fail back to the
                  // main interpreter.
                  BasicBlock *FailBB = getGuardFailureBlock(
                      BB, CurBBIdx, &*I, CurInstrIdx, GuardCount);
                  Builder.CreateBr(FailBB);
                }
                break;
              }
              // Skip frameaddress, control point, and stackmap call
              // instructions.
              assert(isa<CallInst>(I)); // frameaddr call
              I++;
              CurInstrIdx++;
              assert(isa<CallInst>(I)); // control point call
              I++;
              CurInstrIdx++;
              assert(isa<CallInst>(I)); // stackmap call
              LastInst = &*I;
              I++;
              CurInstrIdx++;

              // We've seen the control point so the next block will be
              // unmappable.
              if (!Outlining) {
                assert(OutlineBase == 0);
                Outlining = true;
              }
              break;
            }
          }
        }

        // If execution reaches here, then the instruction I is to be copied
        // into JITMod.
        copyInstruction(&Builder, (Instruction *)&*I, CurBBIdx, CurInstrIdx);
      }
      LastBB = BB;
    }

    // If the trace succeeded, loop back to the top. The only way to leave the
    // trace is via a guard failure.
    if (LoopEntryBB) {
      Builder.CreateBr(LoopEntryBB);
    }
    finalise(AOTMod, &Builder);
    return JITMod;
  }

  /// Handle promotions.
  void handlePromote(CallInst *CI, BasicBlock *BB, size_t CurBBIdx,
                     size_t CurInstrIdx) {
    // First lookup the constant value the trace is going to use.
    uint64_t PConst = __yk_lookup_promote_usize();
    Value *PConstLL =
        Builder.getIntN(PointerSizedIntTy->getIntegerBitWidth(), PConst);
    Value *PromoteVar = CI->getArgOperand(0);
    Value *JITPromoteVar = getMappedValue(PromoteVar);

    // By promoting a value to a constant, we specialised the remainder of the
    // trace to that constant value. We must emit a guard that ensures that we
    // doptimise if the promoted value deviates from the "baked-in" constant.

    BasicBlock *FailBB = getGuardFailureBlock(BB, CurBBIdx, CI->getNextNode(),
                                              CurInstrIdx, GuardCount);
    BasicBlock *SuccBB = BasicBlock::Create(JITMod->getContext(),
                                            GUARD_SUCCESS_BLOCK_NAME, JITFunc);
    Value *Deopt = Builder.CreateICmp(CmpInst::Predicate::ICMP_NE,
                                      JITPromoteVar, PConstLL);
    Builder.CreateCondBr(Deopt, FailBB, SuccBB);

    // Carry on constructing the trace module in the success block.
    Builder.SetInsertPoint(SuccBB);

    // You might think that we need to do: `VMap[CI] = PConstLL` so as to use
    // the constant in place of the promoted value from here onward.
    //
    // As it happens, the guard we just inserted already gives LLVM everything
    // it needs to promote the value for us.
    //
    // If we trace a `yk_promote(x)` when `x==5`, then the trace compiler
    // makes a guard that says "if x!=5 then goto the guard failure block,
    // otherwise goto the guard success block". It is then likely that LLVM
    // will infer that `x==5` in the guard success block and propagate the
    // constant for us automatically.

    // We will have traced the constant recorder. We don't want that in
    // the trace so a) we don't copy the call, and b) we enter
    // outlining mode.
    Outlining = true;
    OutlineBase = CallStack.size();
  }
};

tuple<Module *, string, void *, size_t>
createModule(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t TraceLen,
             void *CallStack, void *AOTValsPtr, size_t AOTValsLen) {
  JITModBuilder JB = JITModBuilder::Create(AOTMod, FuncNames, BBs, TraceLen,
                                           CallStack, AOTValsPtr, AOTValsLen);
  auto JITMod = JB.createModule();
  return make_tuple(JITMod, std::move(JB.TraceName), JB.LiveAOTArray,
                    JB.GuardCount);
}
