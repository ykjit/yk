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

const char *PromoteRecFnName = "__ykllvm_recognised_promote";

#define YK_OUTLINE_FNATTR "yk_outline"

// The first two arguments of a stackmap call are it's id and shadow bytes and
// need to be skipped when scanning the operands for live values.
#define YK_STACKMAP_SKIP_ARGS 2

// Return value telling the caller of the compiled trace that no guard failure
// occurred and it's safe to continue running the fake trace stitching loop.
#define TRACE_RETURN_SUCCESS 0

// The name prefix used for blocks that are branched to when a guard succeeds.
#define GUARD_SUCCESS_BLOCK_NAME "guardsuccess"

// Describes a "resume point" in an LLVM IR basic block.
//
// The trace compiler has to keep track of its progress through the AOT control
// flow of the program. Since the LLVM `call` instruction doesn't terminate a
// block, it isn't enough to keep track of progress at the basic block level.
//
// This data structure is used when a call is encountered to allow the
// compiler to resume from the middle of a block, after a call.
struct BlockResumePoint {
  // The index of the basic block (in the parent function) containing the
  // instruction at which to resume.
  size_t ResumeBBIdx;
  // The instruction *after* which to resume.
  //
  // To be clear, if we have two instructions:
  // ```
  //   call @f()
  //   %x = load(...)
  // ```
  //
  // If we want to resume at the `load`, then we store the `call`.
  Instruction *ResumeAfterInstr;
  // The index of the last call instruction. This is the instruction immediately
  // *before* the instruction we want to resume from after returning from a
  // call.
  //
  // This could be computed by looping over `ResumeInstr`'s parent block, but
  // since it's needed fairly frequently, we cache it.
  size_t ResumeAfterInstrIdx;

  // Dump the resume point to stderr for debugging.
  void dump() {
    errs() << "  BlockResumePoint {\n";
    errs() << "    ResumeBBIdx: " << ResumeBBIdx << "\n";
    errs() << "    ResumeAfterInstr: ";
    ResumeAfterInstr->dump();
    errs() << "    ResumeAfterInstrIdx: " << ResumeAfterInstrIdx << "\n";
    errs() << "  }\n";
  }
};

// A frame for an IR function for which we have LLVM IR.
struct MappableFrame {
public:
  // The function for this frame.
  Function *Func;
  // The basic block that the trace compiler previously processed (or null if
  // this is the first block being handled for this frame).
  BasicBlock *PrevBB;
  // The stackmap call describing the stack as-per the trace compiler's
  // progress through the frame's function (or null if no stackmap call has
  // been encountered in this frame yet).
  CallInst *LastSMCall;
  // The resume point for this frame (if any).
  std::optional<BlockResumePoint> Resume;
  // When true, this frame initiated outlining. When this frame is the
  // most-recent frame again, outlining will cease.
  bool OutlineBase = false;

  // Dump the frame to stderr for debugging.
  void dump() {
    errs() << "MappableFrame {\n";
    errs() << "  Function: " << Func->getName() << "\n";
    if (PrevBB) {
      size_t FoundIdx = 0;
      for (BasicBlock &BB : *PrevBB->getParent()) {
        if (&BB == PrevBB) {
          break;
        }
        FoundIdx++;
      }
      errs() << "  PrevBB: " << PrevBB << " (BBIdx=" << FoundIdx << ")\n";
    } else {
      errs() << "  PrevBB: null\n";
    }
    if (!Resume.has_value()) {
      errs() << "  Resume: N/A\n";
    } else {
      Resume.value().dump();
    }
    errs() << "  LastSMCall: ";
    if (LastSMCall) {
      LastSMCall->dump();
    } else {
      errs() << "null\n";
    }
    errs() << "  OutlineBase: " << OutlineBase << "\n";
    errs() << "}\n";
  }

  // Set the frame's resume point.
  void setResume(size_t BBIdx, Instruction *Instr, size_t InstrIdx) {
    assert(!Resume.has_value());
#ifndef NDEBUG
    // Check `BBIdx` agrees with `Instr`.
    size_t FoundIdx = 0;
    bool Found;
    BasicBlock *BB = Instr->getParent();
    for (BasicBlock &TBB : *BB->getParent()) {
      if (&TBB == BB) {
        Found = true;
        break;
      }
      FoundIdx++;
    }
#endif
    assert(Found);
    assert(FoundIdx == BBIdx);
    Resume = {BBIdx, Instr, InstrIdx};
  }

  // Delete the frame's resume point.
  void clearResume() {
    assert(Resume.has_value());
    Resume.reset();
  }

  // Get the index of the instruction to resume at (if a resume point is set).
  optional<size_t> getResumeInstrIdx() {
    if (Resume.has_value()) {
      return Resume.value().ResumeAfterInstrIdx + 1;
    } else {
      return std::nullopt;
    }
  }

  // Return the last encountered call in the current basic block. This is the
  // call instructions from where we need to resume after processing the
  // callee.
  //
  // Throws if no call has yet been encountered.
  Instruction *getLastCallInstruction() {
    return Resume.value().ResumeAfterInstr;
  }
};

// An entry in the callstack that represents one foreign frame.
struct ForeignFrame {
  void dump() { errs() << "ForeignFrame {}\n"; }
};

// An entry in the call stack.
class StackFrame {
  // Either a frame we have IR for, or foreign frames that we don't.
  std::variant<MappableFrame, ForeignFrame> Inner;

  StackFrame(std::variant<MappableFrame, ForeignFrame> Inner) : Inner(Inner){};

public:
  // Create a frame entry for which we have IR.
  static StackFrame CreateMappableFrame(
      Function *F, CallInst *LastSMCall,
      std::optional<BlockResumePoint> ResumePoint = std::nullopt) {
    return StackFrame(std::variant<MappableFrame, ForeignFrame>(
        MappableFrame{F, nullptr, LastSMCall, ResumePoint}));
  }

  // Create a frame entry for foreign code for which we have no IR.
  static StackFrame CreateForeignFrame() {
    return StackFrame(
        std::variant<MappableFrame, ForeignFrame>(ForeignFrame{}));
  }

  // If this frame is mappable, return a pointer to the mappable frame,
  // otherwise return null.
  MappableFrame *getMappableFrame() {
    return std::get_if<MappableFrame>(&Inner);
  }

  // Dump the frame to stderr for debugging.
  void dump() {
    if (MappableFrame *MF = std::get_if<MappableFrame>(&Inner)) {
      MF->dump();
    } else {
      std::get<ForeignFrame>(Inner).dump();
    }
  }
};

// A dtaa structure tracking the AOT LLVM IR stack during trace compilation.
class CallStack {
  // Stack frames, older frames at lower indices.
  std::vector<StackFrame> Stack;

public:
  // Add a new frame to the stack.
  void pushFrame(StackFrame F) { Stack.push_back(F); }

  // Pop and return the most-recent frame from the stack.
  StackFrame popFrame() {
    assert(!Stack.empty());
    StackFrame F = Stack.back();
    Stack.pop_back();
    return F;
  }

  // Peek at the most-current frame.
  StackFrame &curFrame() {
    assert(!Stack.empty());
    return Stack.back();
  }

  // Peek at the most recent frame, returning a pointer to a mappable frame if
  // it is mappable, otherwise null.
  MappableFrame *curMappableFrame() { return curFrame().getMappableFrame(); }

  // Peek at the frame at index `Idx`. Index zero is the oldest frame.
  StackFrame &getFrame(size_t Idx) { return Stack.at(Idx); }

  // Return the depth of the stack.
  size_t size() { return Stack.size(); }

  // Returns true if the stack contains a frame for the specified function.
  //
  // Used to know if a (mappable) call is recursive.
  bool hasFrameForFunction(Function *F) {
    for (StackFrame &SF : Stack) {
      MappableFrame *MF = SF.getMappableFrame();
      if (MF && (MF->Func == F)) {
        return true;
      }
    }
    return false;
  }

  // Dump the entire call stack to stderr for debugging.
  void dump() {
    errs() << "<CallStack (size=" << Stack.size() << ")>\n";
    for (StackFrame &F : Stack) {
      F.dump();
    }
    errs() << "</CallStack>\n";
  }
};

// Dump an error message and an LLVM value to stderr and exit with failure.
void dumpValueAndExit(const char *Msg, Value *V) {
  errs() << Msg << ": ";
  V->print(errs());
  exit(EXIT_FAILURE);
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
struct UnmappableRegion {
  // The effect executing the unmappable block had on the stack depth. This is
  // the number of frames (not the size of the stack in bytes).
  //
  // Since each UnmappableRegion in a trace is followed by a mappable IRBlock,
  // this value can be used to determine if the unmappable code:
  //
  //  - returned back to mappable code (StackAdjust < 0).
  //  - called deeper into mappable code (StackAdjust > 0).
  //  - "fell through" into mappable code (StackAdjust ==  0).
  ssize_t StackAdjust;
};

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
      UnmappableRegion *U = std::get_if<UnmappableRegion>(&Loc);
      assert(U);
      errs() << "UnmappableRegion(StackAdjust=" << U->StackAdjust << ")\n";
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
      // Subtle bitcast from `size_t` to `ssize_t`. When the trace was encoded
      // into an FFI friendly format, unmappable blocks use the `Idx` field
      // (a `size_t`) to encode the stack adjustment value (a `ssize_t`). The
      // cast below reverses that.
      return TraceLoc(variant<IRBlock, UnmappableRegion>{
          UnmappableRegion{llvm::bit_cast<ssize_t, size_t>(BBs[Idx])}});
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
  size_t BBIdx;
  size_t InstrIdx;
  const char *FName;
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

  // Map JIT instruction to basic block index, instruction index, function
  // name, and stackframe index of the corresponding AOT instruction.
  std::map<Value *, std::tuple<size_t, size_t, Instruction *, size_t>> AOTMap;

  // The trace-compiler's view of the AOT IR call stack during trace
  // compilation.
  CallStack CallStack;

  Value *getMappedValue(Value *V) {
    if (VMap.find(V) != VMap.end()) {
      return VMap[V];
    }
    assert(isa<Constant>(V));
    return V;
  }

  void insertAOTMap(Instruction *AOT, Value *JIT, size_t BBIdx,
                    size_t InstrIdx) {
    AOTMap[JIT] = {BBIdx, InstrIdx, AOT, CallStack.size() - 1};
  }

  // Start outlining.
  //
  // This flips the "are we outlining" flag and marks the most-recent frame as
  // the "outline base" (the frame where outlining started).
  void startOutlining() {
    assert(!Outlining);
    MappableFrame *MPF = CallStack.curMappableFrame();
    assert(MPF);
    MPF->OutlineBase = true;
    Outlining = true;
  }

  // Try to stop outlining.
  //
  // Outlining is stopped if the most-recent frame is the "outline base".
  void tryStopOutlining() {
    assert(Outlining);
    MappableFrame *MPF = CallStack.curMappableFrame();
    if (MPF && MPF->OutlineBase) {
      // We got back to the frame where outlining started. Stop outlining.
      MPF->OutlineBase = false;
      Outlining = false;
    }
  }

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
    // Update the most-recent frame's progress so that we return to the right
    // place when we return from this call.
    MappableFrame *CurFrame = CallStack.curFrame().getMappableFrame();
    assert(CurFrame);
    CurFrame->setResume(CurBBIdx, CI, CurInstrIdx);

    if (CF == nullptr || CF->isDeclaration()) {
      // The definition of the callee is external to AOTMod. It will be
      // outlined, but we still need to declare it locally if we have not
      // done so yet.
      if (CF != nullptr && VMap.find(CF) == VMap.end()) {
        declareFunction(CF);
      }
      if (!Outlining) {
        copyInstruction(&Builder, (Instruction *)&*CI, CurBBIdx, CurInstrIdx);
        startOutlining();
      }
      CallStack.pushFrame(StackFrame::CreateForeignFrame());
    } else {
      // Calling to a non-foreign function.
      if (!Outlining) {
        // We are not outlining, but should this call start us outlining?
        if (CF->hasFnAttribute(YK_OUTLINE_FNATTR) || CF->isVarArg() ||
            CallStack.hasFrameForFunction(CF)) {
          // We will outline this call.
          //
          // If this is a recursive call that has been inlined, or if the callee
          // has the "yk_outline" annotation, remove the inlined code and turn
          // it into a normal (outlined) call.
          if (VMap.find(CF) == VMap.end()) {
            declareFunction(CF);
            addGlobalMappingForFunction(CF);
          }
          copyInstruction(&Builder, CI, CurBBIdx, CurInstrIdx);
          startOutlining();
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
      CallInst *LastSMCall = cast<CallInst>(CI->getNextNonDebugInstruction());
      CurFrame->LastSMCall = LastSMCall;
      CallStack.pushFrame(StackFrame::CreateMappableFrame(CF, nullptr));
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
    BasicBlock *FailBB =
        getGuardFailureBlock(BI->getParent(), CurBBIdx, I, CurInstrIdx);
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
        getGuardFailureBlock(SI->getParent(), CurBBIdx, I, CurInstrIdx);
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
    CallStack.popFrame();

    // Check if we have arrived back at the frame where outlining started.
    if (Outlining) {
      tryStopOutlining();
      return;
    }

    // Replace the return variable of the call with its return value.
    // Since the return value will have already been copied over to the
    // JITModule, make sure we look up the copy.
    auto OldRetVal = ((ReturnInst *)&*I)->getReturnValue();
    if (OldRetVal != nullptr) {
      MappableFrame *MPF = CallStack.curMappableFrame();
      assert(MPF);
      Instruction *AOT = MPF->getLastCallInstruction();
      assert(isa<CallInst>(AOT));
      Value *JIT = getMappedValue(OldRetVal);
      VMap[AOT] = JIT;

      CurBBIdx = MPF->Resume.value().ResumeBBIdx;
      CurInstrIdx = MPF->Resume.value().ResumeAfterInstrIdx;
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
    insertAOTMap(I, NewInst, CurBBIdx, CurInstrIdx);
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
                                   Instruction *Instr, size_t CurInstrIdx) {
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

    // Add the control point struct to the live variables we pass into the
    // `deoptimize` call so that it can be accessed.
    Value *YKCPArg = JITFunc->getArg(JITFUNC_ARG_INPUTS_STRUCT_IDX);
    std::tuple<size_t, size_t, Instruction *, size_t> YkCPAlloca =
        getYkCPAlloca();
    AOTMap[YKCPArg] = YkCPAlloca;

    IntegerType *Int32Ty = Type::getInt32Ty(Context);
    PointerType *Int8PtrTy = Type::getInt8PtrTy(Context);

    // Create a vector of active stackframes (i.e. basic block index,
    // instruction index, function name). This will be needed later for
    // reconstructing the stack after deoptimisation.
    // FIXME: Use function index instead of string name.
    StructType *ActiveFrameSTy = StructType::get(
        Context, {PointerSizedIntTy, PointerSizedIntTy, Int8PtrTy});
    AllocaInst *ActiveFrameVec = FailBuilder.CreateAlloca(
        ActiveFrameSTy, ConstantInt::get(PointerSizedIntTy, CallStack.size()));

    // To describe the callstack at a guard failure, the following code uses
    // the basic block resume points for all of the frames on the stack. The
    // most-recent frame won't have a resume point set (it's usually only set
    // by calls), so we will add one temporarily.
    MappableFrame *CurFrame = CallStack.curMappableFrame();
    assert(CurFrame);
    CurFrame->setResume(CurBBIdx, Instr, CurInstrIdx);

    std::vector<Value *> LiveValues;
    for (size_t I = 0; I < CallStack.size(); I++) {
      StackFrame &SF = CallStack.getFrame(I);
      MappableFrame *MF = SF.getMappableFrame();
      assert(MF); // No unmappable frame can occur here.

      // Read live AOTMod values from stackmap calls and find their
      // corresponding values in JITMod. These are exactly the values that are
      // live at each guard failure and need to be deoptimised.
      CallInst *SMC = MF->LastSMCall;
      assert(SMC);
      for (size_t Idx = YK_STACKMAP_SKIP_ARGS; Idx < SMC->arg_size(); Idx++) {
        Value *Arg = SMC->getArgOperand(Idx);
        if (Arg == ControlPointCallInst) {
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
      assert(MF->Resume.has_value());
      BlockResumePoint &RP = MF->Resume.value();
      FailBuilder.CreateStore(
          ConstantInt::get(PointerSizedIntTy, RP.ResumeBBIdx), GEP);
      GEP = FailBuilder.CreateGEP(ActiveFrameSTy, ActiveFrameVec,
                                  {ConstantInt::get(PointerSizedIntTy, I),
                                   ConstantInt::get(Int32Ty, 1)});
      FailBuilder.CreateStore(
          ConstantInt::get(PointerSizedIntTy, RP.ResumeAfterInstrIdx), GEP);
      Value *CurFunc = FailBuilder.CreateGlobalStringPtr(MF->Func->getName());
      GEP = FailBuilder.CreateGEP(ActiveFrameSTy, ActiveFrameVec,
                                  {ConstantInt::get(PointerSizedIntTy, I),
                                   ConstantInt::get(Int32Ty, 2)});
      FailBuilder.CreateStore(CurFunc, GEP);
    }

    // Store the active frames vector and its length in a separate struct to
    // save arguments.
    AllocaInst *ActiveFramesStruct = createAndFillStruct(
        FailBuilder, {ActiveFrameVec,
                      ConstantInt::get(PointerSizedIntTy, CallStack.size())});

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

    // Store the offset and length of the live AOT variables.
    AllocaInst *AOTLocs = createAndFillStruct(
        FailBuilder,
        {ConstantInt::get(PointerSizedIntTy, CurPos * sizeof(AOTInfo)),
         ConstantInt::get(PointerSizedIntTy, LiveValues.size())});

    // Create the deoptimization call.
    Type *retty = PointerType::get(Context, 0);
    Function *DeoptInt = Intrinsic::getDeclaration(
        JITFunc->getParent(), Intrinsic::experimental_deoptimize, {retty});
    OperandBundleDef ob =
        OperandBundleDef("deopt", (ArrayRef<Value *>)LiveValues);
    // We already passed the stackmap address and size into the trace
    // function so pass them on to the __llvm_deoptimize call.
    CallInst *Ret = CallInst::Create(
        DeoptInt,
        {JITFunc->getArg(JITFUNC_ARG_COMPILEDTRACE_IDX),
         JITFunc->getArg(JITFUNC_ARG_FRAMEADDR_IDX), AOTLocs,
         ActiveFramesStruct, ConstantInt::get(PointerSizedIntTy, GuardCount)},
        {ob}, "", GuardFailBB);

    // We always need to return after the deoptimisation call.
    ReturnInst::Create(Context, Ret, GuardFailBB);

    // Delete the temporary resume point for the most-recent frame.
    CurFrame->clearResume();

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
    Function *F = CPCIBB->getParent();

    size_t CurInstrIdx = 0;
    size_t CurBBIdx = 0;
    for (auto BB = F->begin(); &*BB != CPCIBB; BB++) {
      CurBBIdx += 1;
    }

    // Find the store instructions related to YkCtrlPointVars and generate and
    // insert a load for each stored value.
    for (BasicBlock::iterator CI = CPCIBB->begin(); &*CI != CPCI; CI++) {
      assert(CI != CPCIBB->end());
      CurInstrIdx++;
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
        // For deoptimisation we need to create a mapping from the original AOT
        // value to the corresponding JIT value. This mapping is encoded using
        // the basic block index and instruction index. These are costly to
        // calculate, so instead we create mapping to the YkCtrlPointVars store
        // instruction instead. During optimisation we can then easily fish out
        // the original AOT value via this instructions operand.
        insertAOTMap(cast<Instruction>(StoredVal), Load, CurBBIdx, CurInstrIdx);
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

    User *CallSite = F->user_back();
    CallInst *CPCI = cast<CallInst>(CallSite);
    assert(CPCI->arg_size() == YK_CONTROL_POINT_NUM_ARGS);

    // Get the instruction index of CPCI in its parent block.
    size_t CPCIIdx = 0;
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
                size_t TraceLen, char *FAddrKeys[], void *FAddrVals[],
                size_t FAddrLen, CallInst *CPCI,
                std::optional<std::tuple<size_t, CallInst *>> InitialResume,
                Value *TraceInputs)
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

    // Map the live variables struct used inside the trace to the corresponding
    // argument of the compiled trace function.
    VMap[TraceInputs] = JITFunc->getArg(JITFUNC_ARG_INPUTS_STRUCT_IDX);

    // Push the initial frame.
    IRBlock *StartIRB = InpTrace[0].getMappedBlock();
    assert(StartIRB);
    Function *StartFunc = AOTMod->getFunction(StartIRB->FuncName);
    std::optional<BlockResumePoint> RP;
    if (InitialResume.has_value()) {
      auto [StartInstrIdx, StartCPCall] = *InitialResume;
      assert(StartCPCall->getFunction() == StartFunc);
      RP = BlockResumePoint{StartIRB->BBIdx, StartCPCall, StartInstrIdx};
    }
    StackFrame InitFrame =
        StackFrame::CreateMappableFrame(StartFunc, nullptr, RP);
    CallStack.pushFrame(InitFrame);

    createTraceHeader(ControlPointCallInst, TraceInputs->getType());

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
      assert(InpTrace[I].getMappedBlock() || InpTrace[I + 1].getMappedBlock());
    }
#endif
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
  size_t GuardCount = 0;

  JITModBuilder(JITModBuilder &&);

  static JITModBuilder Create(Module *AOTMod, char *FuncNames[], size_t BBs[],
                              size_t TraceLen, char *FAddrKeys[],
                              void *FAddrVals[], size_t FAddrLen) {
    CallInst *CPCI;
    Value *TI;
    size_t CPCIIdx;
    std::tie(CPCI, CPCIIdx, TI) = GetControlPointInfo(AOTMod);
    return JITModBuilder(AOTMod, FuncNames, BBs, TraceLen, FAddrKeys, FAddrVals,
                         FAddrLen, CPCI, make_tuple(CPCIIdx, CPCI), TI);
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

    // Actually make the trace compiler now.
    //
    // Note the use of the empty optional `{}` for the initial value of the
    // first frame's `BlockResumePoint`. This means that the compiler will
    // start copying instructions from the beginning of the first block in the
    // trace, instead of after the return from the control point.
    JITModBuilder JB(AOTMod, FuncNames, BBs, TraceLen, &NewFAddrKeys[0],
                     &NewFAddrVals[0], NewFAddrKeys.size(), CPCI, {},
                     TraceInputs);

    return JB;
  }
#endif

  // Generate the JIT module by "glueing together" blocks that the trace
  // executed in the AOT module.
  Module *createModule() {
    // Iterate over the blocks of the trace.
    for (size_t Idx = 0; Idx < InpTrace.Length(); Idx++) {
      // Update the previously executed BB in the most-recent frame (if it's
      // mappable).
      TraceLoc Loc = InpTrace[Idx];

      if (UnmappableRegion *UR = Loc.getUnmappableRegion()) {
        // The trace entered a region of unmappable foreign code.
        //
        // As noted in the mapper and asserted in the JITModBuilder constructor,
        // there are never two unmappable regions in a row, so the next block
        // (if there is one) is guaranteed to be mappable. The question is: will
        // the foreign code be *returning* into mappable code, or will it be
        // *calling back* deeper into mappable code? The trace decoder keeps
        // track of the stack effects of foreign code (assuming its control flow
        // isn't bonkers), so we are able to use that information to decide.

        if (InpTrace.Length() == Idx + 1) {
          // This unmappable region is the end of the trace, so there's no
          // need to maintain the stack any more.
          continue;
        }

        assert(!CallStack.curMappableFrame());

        // FIXME: think about what do do in the face of setjmp/longjmp. How
        // would we find the stack adjustment value for such code?
        if (UR->StackAdjust < 0) {
          // The stack got smaller as a result of executing the foreign code,
          // so we must be returning to mappable code.
          while (UR->StackAdjust < 0) {
            // We don't allow foreign code to pop non-foreign frames. That
            // seems like a sure indiciator of bonkers control flow.
            assert(!CallStack.curMappableFrame());
            CallStack.popFrame();
            UR->StackAdjust++;
          }
          assert(CallStack.curMappableFrame());
          tryStopOutlining();
        } else {
          // If the stack size hasn't changed, either we've got something wrong,
          // or the unmappable code has transitioned back to mappable code in a
          // way we don't currently support.
          assert(UR->StackAdjust > 0);

          // The stack got bigger as a result of executing the foreign code. It
          // must have called deeper and eventually into mappable code.
          //
          // If the stack adjustment value is N, then there must be N-1 foreign
          // frames and the last remaining frame is the new mappable frame.
          while (UR->StackAdjust > 1) {
            CallStack.pushFrame(StackFrame::CreateForeignFrame());
            UR->StackAdjust--;
          }
          IRBlock *NextIB = InpTrace[Idx + 1].getMappedBlock();
          assert(NextIB);
          assert(NextIB->BBIdx == 0);
          auto [NextFunc, BB] = getLLVMAOTFuncAndBlock(NextIB);
          CallStack.pushFrame(
              StackFrame::CreateMappableFrame(NextFunc, nullptr));
        }
        continue;
      }

      // If we get here then we must have a mappable block and the most-recent
      // frame should also be mappable.
      MappableFrame *MPF = CallStack.curMappableFrame();
      assert(MPF);

      IRBlock *IB = Loc.getMappedBlock();
      assert(IB);
      size_t CurBBIdx = IB->BBIdx;

      auto [F, BB] = getLLVMAOTFuncAndBlock(IB);
      assert(MPF->Func == F);

#ifndef NDEBUG
      // `BB` should be a successor of the last block executed in this frame.
      if (MPF->PrevBB) {
        bool PredFound = false;
        for (BasicBlock *PBB : predecessors(BB)) {
          if (PBB == MPF->PrevBB) {
            PredFound = true;
            break;
          }
        }
        assert(PredFound);
      }
#endif

      // Used to decide if we should update `PrevBB` of the most-recent frame
      // after this block finishes.
      size_t StackDepthBefore = CallStack.size();

      // The index of the instruction in `BB` that we are currently processing.
      size_t CurInstrIdx = 0;

      // If we've returned from a call, skip ahead to the instruction where
      // we left off.
      std::optional<size_t> ResumeIdx = MPF->getResumeInstrIdx();
      if (ResumeIdx.has_value()) {
        CurInstrIdx = *ResumeIdx;
        MPF->clearResume();
      }

      // Iterate over all instructions within this block and copy them over
      // to our new module.
      for (; CurInstrIdx < BB->size(); CurInstrIdx++) {
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
              MPF->LastSMCall = cast<CallInst>(I);
              continue;
            }

            // Whitelist intrinsics that appear to be always inlined.
            if (IID == Intrinsic::vastart || IID == Intrinsic::vaend ||
                IID == Intrinsic::smax ||
                IID == Intrinsic::usub_with_overflow) {
              if (!Outlining) {
                copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                CurInstrIdx);
              }
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
              if (!Outlining) {
                copyInstruction(&Builder, cast<CallInst>(I), CurBBIdx,
                                CurInstrIdx);
              }
              continue;
            }
            // The intrinsic wasn't inlined so we let the following code handle
            // it which already knows how to deal with such cases.
          }

          CallInst *CI = cast<CallInst>(I);
          Function *CF = CI->getCalledFunction();
          if (CF == nullptr) {
            // The target isn't statically known, so we can't inline the
            // callee.
            if (!isa<InlineAsm>(CI->getCalledOperand())) {
              // Look ahead in the trace to find the callee so we can
              // map the arguments if we are inlining the call.
              TraceLoc MaybeNextIB = InpTrace[Idx + 1];
              if (const IRBlock *NextIB = MaybeNextIB.getMappedBlock()) {
                CF = AOTMod->getFunction(NextIB->FuncName);
              } else {
                CF = nullptr;
              }
              // FIXME Don't inline indirect calls unless promoted.
              handleCallInst(CI, CF, CurBBIdx, CurInstrIdx);
              break;
            }
          } else if (CF->getName() == PromoteRecFnName) {
            // A value is being promoted to a constant.
            handlePromote(CI, MPF, BB, CurBBIdx, CurInstrIdx);
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
          assert(MPF->PrevBB);
          handlePHINode(&*I, MPF->PrevBB, CurBBIdx, CurInstrIdx);
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
              if (Idx == InpTrace.Length() - 2) {
                // Once we reached the YkCtrlPointVars stores at the end of the
                // trace, we're done. We don't need to copy those instructions
                // over, since all YkCtrlPointVars are stored on the shadow
                // stack.
                // FIXME: Once we allow more optimisations in AOT, some of the
                // YkCtrlPointVars won't be stored on the shadow stack anymore,
                // and we might have to insert PHI nodes into the loop entry
                // block.
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
              MPF->LastSMCall = cast<CallInst>(I);
              I++;
              CurInstrIdx++;

              // We've seen the control point so the next block will be
              // unmappable. Set a resume point so we can continue collecting
              // instructions afterwards.
              MPF->setResume(CurBBIdx, &*I, CurInstrIdx);
              if (!Outlining) {
                startOutlining();
              }
              CallStack.pushFrame(StackFrame::CreateForeignFrame());
              break;
            }
          }
        }

        // If execution reaches here, then the instruction I is to be copied
        // into JITMod.
        copyInstruction(&Builder, (Instruction *)&*I, CurBBIdx, CurInstrIdx);
      }

      // Block complete. If we are still in the same frame, then update the
      // previous basic block.
      if (CallStack.size() == StackDepthBefore) {
        MPF->PrevBB = BB;
      }
    }

    // If the trace succeeded, loop back to the top. The only way to leave the
    // trace is via a guard failure.
    if (LoopEntryBB) {
      Builder.CreateBr(LoopEntryBB);
    } else {
      // This is here only because some of our `.ll` tests don't contain a
      // control point, so the loop-entry block is never created.
      Builder.CreateRet(
          ConstantPointerNull::get(PointerType::get(JITMod->getContext(), 0)));
    }
    finalise(AOTMod, &Builder);
    return JITMod;
  }

  /// Handle promotions.
  void handlePromote(CallInst *CI, MappableFrame *MPF, BasicBlock *BB,
                     size_t CurBBIdx, size_t CurInstrIdx) {
    // First lookup the constant value the trace is going to use.
    uint64_t PConst = __yk_lookup_promote_usize();
    Value *PConstLL =
        Builder.getIntN(PointerSizedIntTy->getIntegerBitWidth(), PConst);
    Value *PromoteVar = CI->getArgOperand(0);
    Value *JITPromoteVar = getMappedValue(PromoteVar);

    // By promoting a value to a constant, we specialised the remainder of the
    // trace to that constant value. We must emit a guard that ensures that we
    // doptimise if the promoted value deviates from the "baked-in" constant.
    MPF->LastSMCall = cast<CallInst>(CI->getNextNonDebugInstruction());

    // When we encounter `%y = yk_promote(%x)` in the AOT module, we don't copy
    // this instruction into the JIT module, only update future references to
    // `%y` with a constant. However, in AOT `%y` still exists and needs to be
    // re-materialised when we deoptimise, so we have to point some JIT variable
    // back to `%y` in our AOTMAP.
    //
    // Since in AOT, `%y` and `%x` are identical, really we'd like to point
    // `AOTMap[JITPromoteVar]` to `CI`, but that would overwrite the existing
    // (required!) mapping for `JITPromoteVar`.
    //
    // What we therefore do is make an instruction to the effect `%y = %x` in
    // the JIT module, and map that instruction back to `%y` in the AOT module.
    Instruction *NewInst = SelectInst::Create(
        ConstantInt::get(Type::getInt1Ty(JITMod->getContext()), 0),
        JITPromoteVar, JITPromoteVar);
    Builder.Insert(NewInst);

    VMap[CI] = NewInst;
    insertAOTMap(CI, NewInst, CurBBIdx, CurInstrIdx);

    BasicBlock *FailBB = getGuardFailureBlock(BB, CurBBIdx, CI, CurInstrIdx);
    BasicBlock *SuccBB = BasicBlock::Create(JITMod->getContext(),
                                            GUARD_SUCCESS_BLOCK_NAME, JITFunc);
    Value *Deopt = Builder.CreateICmp(CmpInst::Predicate::ICMP_NE,
                                      JITPromoteVar, PConstLL);
    Builder.CreateCondBr(Deopt, FailBB, SuccBB);

    // Carry on constructing the trace module in the success block.
    Builder.SetInsertPoint(SuccBB);

    // Update the VMap so that the promoted value is now a constant.
    VMap[CI] = PConstLL;

    // We will have traced the constant recorder. We don't want that in
    // the trace so a) we don't copy the call, and b) we enter
    // outlining mode.
    MPF->setResume(CurBBIdx, &*CI, CurInstrIdx);
    startOutlining();
    CallStack.pushFrame(StackFrame::CreateForeignFrame());
  }
};

tuple<Module *, string, std::map<GlobalValue *, void *>, void *, size_t>
createModule(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t TraceLen,
             char *FAddrKeys[], void *FAddrVals[], size_t FAddrLen) {
  JITModBuilder JB = JITModBuilder::Create(AOTMod, FuncNames, BBs, TraceLen,
                                           FAddrKeys, FAddrVals, FAddrLen);
  auto JITMod = JB.createModule();
  return make_tuple(JITMod, std::move(JB.TraceName),
                    std::move(JB.GlobalMappings), JB.LiveAOTArray,
                    JB.GuardCount);
}

#ifdef YK_TESTING
tuple<Module *, string, std::map<GlobalValue *, void *>, void *, size_t>
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
                    std::move(JB.GlobalMappings), nullptr, 0);
}
#endif
