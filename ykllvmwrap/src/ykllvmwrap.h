#ifndef _YKLLVMWRAP_H
#define _YKLLVMWRAP_H

#include <atomic>

#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

using namespace llvm;

// A function name and basic block index pair that identifies a block in the
// AOT LLVM IR.
struct IRBlock {
  // A non-null pointer to the function name.
  char *FuncName;
  // The index of the block in the parent LLVM function.
  size_t BBIdx;
};

// Function virtual addresses observed in the input trace.
// Maps a function symbol name to a virtual address.
class FuncAddrs {
  std::map<std::string, void *> Map;

public:
  FuncAddrs(char **FuncNames, void **VAddrs, size_t Len);

  // Lookup the address of the specified function name or return nullptr on
  // failure.
  void *operator[](const char *FuncName);
};

// Describes the software or hardware trace to be compiled using LLVM.
class InputTrace {
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
  InputTrace(char **FuncNames, size_t *BBs, size_t Len);
  size_t Length();
  const Optional<IRBlock> operator[](size_t Idx);
  const IRBlock getUnchecked(size_t Idx);
};

class JITModBuilder {
  // Global variables/functions that were copied over and need to be
  // initialised.
  std::vector<GlobalVariable *> cloned_globals;
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
  std::vector<std::tuple<size_t, CallInst *>> InlinedCalls;
  // Instruction at which to continue after an a call.
  Optional<std::tuple<size_t, CallInst *>> ResumeAfter;
  // Depth of nested calls when outlining a recursive function.
  size_t RecCallDepth = 0;
  // Signifies a hole (for which we have no IR) in the trace.
  bool ExpectUnmappable = false;
  // The JITMod's builder.
  llvm::IRBuilder<> Builder;
  // Dead values to recursively delete upon finalisation of the JITMod. This is
  // required because it's not safe to recursively delete values in the middle
  // of creating the JIT module. We don't know if any of those values might be
  // required later in the trace.
  std::vector<Value *> DeleteDeadOnFinalise;

  // Information about the trace we are compiling.
  InputTrace InpTrace;
  // Function virtual addresses discovered from the input trace.
  FuncAddrs FAddrs;

  // A stack of BasicBlocks. Each time we enter a new call frame, we push the
  // first basic block to the stack. Following a branch to another basic block
  // updates the most recently pushed block. This is required for selecting the
  // correct incoming value when tracing a PHI node.
  std::vector<BasicBlock *> LastCompletedBlocks;

  // Since a trace starts tracing after the control point but ends before it,
  // we need to map the values inserted into the `YkCtrlPointVars` (appearing
  // before the control point) to the extracted values (appearing after the
  // control point). This map helps to match inserted values to their
  // corresponding extracted values using their index in the struct.
  std::map<uint64_t, Value *> InsertValueMap;

  Value *getMappedValue(Value *V);

  // Returns true if the given function exists on the call stack, which means
  // this is a recursive call.
  bool isRecursiveCall(Function *F);

  // Add an external declaration for the given function to JITMod.
  void declareFunction(Function *F);

  // Find the machine code corresponding to the given AOT IR function and
  // ensure there's a mapping from its name to that machine code.
  void addGlobalMappingForFunction(Function *CF);

  void handleCallInst(CallInst *CI, Function *CF, size_t &CurInstrIdx);
  void handleReturnInst(Instruction *I);
  void handlePHINode(Instruction *I, BasicBlock *BB);
  void handleOperand(Value *Op);

  void copyInstruction(IRBuilder<> *Builder, Instruction *I);
  Function *createJITFunc(Value *TraceInputs, Type *RetTy);

  // Delete the dead value `V` from its parent, also deleting any dependencies
  // of `V` (i.e. operands) which then become dead.
  void deleteDeadTransitive(Value *V);

  // Finalise the JITModule by adding a return instruction and initialising
  // global variables.
  void finalise(Module *AOTMod, IRBuilder<> *Builder);

public:
  // Store virtual addresses for called functions.
  std::map<GlobalValue *, void *> globalMappings;
  // The function name of this trace.
  std::string TraceName;
  // Mapping from AOT instructions to JIT instructions.
  ValueToValueMapTy VMap;

  JITModBuilder(Module *AOTMod, char *FuncNames[], size_t BBs[],
                size_t TraceLen, char *FAddrKeys[], void *FAddrVals[],
                size_t FAddrLen);

  // Generate the JIT module.
  Module *createModule();
};

struct AllocMem {
  uint8_t *Ptr;
  uintptr_t Size;
};

class MemMan : public RTDyldMemoryManager {
public:
  MemMan();
  ~MemMan() override;

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool isReadOnly) override;

  bool finalizeMemory(std::string *ErrMsg) override;
  void freeMemory();

private:
  std::vector<AllocMem> code;
  std::vector<AllocMem> data;
};

#endif
