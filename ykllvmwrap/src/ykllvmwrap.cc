// LLVM-related C++ code wrapped in the C ABI for calling from Rust.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include <llvm/DebugInfo/Symbolize/Symbolize.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <atomic>
#include <dlfcn.h>
#include <err.h>
#include <limits>
#include <link.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

#include "memman.cc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::symbolize;
using namespace std;

// The bitcode module loaded from the .llvmbc section of the currently-running
// binary. This cannot be shared across threads and used concurrently without
// acquiring a lock, and since we do want to allow parallel compilation, each
// thread takes a copy of this module.
ThreadSafeModule GlobalAOTMod;

// Flag used to ensure that GlobalAOTMod is loaded only once.
once_flag GlobalAOTModLoaded;

// A copy of GlobalAOTMod for use by a single thread.
//
// A thread should never access this directly, but should instead go via
// getThreadAOTMod() which deals with the necessary lazy initialisation.
//
// PERF: Copying GlobalAOTMod is quite expensive (cloneToNewContext()
// serialises and deserializes). When a compilation thread dies, we should
// return its ThreadAOTMod to a pool and transfer ownership to the next thread
// that needs its own copy of GlobalAOTMod.
thread_local ThreadSafeModule ThreadAOTMod;

// A flag indicating whether GlobalAOTMod has been copied into the thread yet.
thread_local bool ThreadAOTModInitialized = false;

#define TRACE_FUNC_PREFIX "__yk_compiled_trace_"
#define YKTRACE_START "__yktrace_start_tracing"
#define YKTRACE_STOP "__yktrace_stop_tracing"

// An atomic counter used to issue compiled traces with unique names.
atomic<uint64_t> NextTraceIdx(0);

extern "C" LLVMSymbolizer *__yk_llvmwrap_symbolizer_new() {
  return new LLVMSymbolizer;
}

extern "C" void __yk_llvmwrap_symbolizer_free(LLVMSymbolizer *Symbolizer) {
  delete Symbolizer;
}

// Finds the name of a code symbol from a virtual address.
// The caller is responsible for freeing the returned (heap-allocated) C string.
extern "C" char *
__yk_llvmwrap_symbolizer_find_code_sym(LLVMSymbolizer *Symbolizer,
                                       const char *Obj, uint64_t Off) {
  object::SectionedAddress Mod{Off, object::SectionedAddress::UndefSection};
  auto LineInfo = Symbolizer->symbolizeCode(Obj, Mod);
  if (auto Err = LineInfo.takeError()) {
    return NULL;
  }

  // PERF: get rid of heap allocation.
  return strdup(LineInfo->FunctionName.c_str());
}

// Load the GlobalAOTMod.
//
// This must only be called from getAOTMod() for correct synchronisation.
void loadAOTMod(char *Ptr, size_t Len) {
  auto Sf = StringRef(Ptr, Len);
  auto Mb = MemoryBufferRef(Sf, "");
  SMDiagnostic Error;
  ThreadSafeContext AOTCtx = std::make_unique<LLVMContext>();
  auto M = parseIR(Mb, Error, *AOTCtx.getContext());
  if (!M)
    errx(EXIT_FAILURE, "Can't load module.");
  GlobalAOTMod = ThreadSafeModule(std::move(M), std::move(AOTCtx));
}

// Get a thread-safe handle on the LLVM module stored in the .llvmbc section of
// the binary. The module is loaded if we haven't yet done so.
//
// When loading the module, the section's raw data must have been loaded
// into memory elsewhere and passed in via the `Ptr` and `Len` arguments.
//
// If the module has already been loaded, then `Ptr` and `Len` are unused.
//
// FIXME The raw section data is repeatedly loaded in Rust code every time
// IRTrace::compile() is called. We should move the raw loading into C++ and
// put it in loadModule(). This would simplify the interface to this function
// (Ptr and Len could go) and would ensure that the section is only read in
// once.
ThreadSafeModule *getThreadAOTMod(char *Ptr, size_t Len) {
  std::call_once(GlobalAOTModLoaded, loadAOTMod, Ptr, Len);
  if (!ThreadAOTModInitialized) {
    ThreadAOTMod = cloneToNewContext(GlobalAOTMod);
    ThreadAOTModInitialized = true;
  }
  return &ThreadAOTMod;
}

std::vector<Value *> get_trace_inputs(Function *F, uintptr_t BBIdx) {
  std::vector<Value *> Vec;
  auto It = F->begin();
  // Skip to the first block in the trace which contains the `start_tracing`
  // call.
  std::advance(It, BBIdx);
  BasicBlock *BB = &*It;
  for (auto I = BB->begin(); I != BB->end(); I++) {
    if (isa<CallInst>(I)) {
      CallInst *CI = cast<CallInst>(&*I);
      if (CI->getCalledFunction()->getName() == YKTRACE_START) {
        // Skip first argument to start_tracing.
        for (auto Arg = CI->arg_begin() + 1; Arg != CI->arg_end(); Arg++) {
          Vec.push_back(Arg->get());
        }
        break;
      }
    }
  }
  return Vec;
}

// Compile a module in-memory and return a pointer to its function.
extern "C" void *compile_module(string TraceName, Module *M) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // FIXME Remember memman or allocated memory pointers so we can free the
  // latter when we're done with the trace.
  auto memman = new MemMan();

  auto MPtr = std::unique_ptr<Module>(M);
  ExecutionEngine *EE =
      EngineBuilder(std::move(MPtr))
          .setMemoryManager(std::unique_ptr<MCJITMemoryManager>(memman))
          .create();
  EE->finalizeObject();

  if (EE->hasError())
    errx(EXIT_FAILURE, "Couldn't compile trace: %s",
         EE->getErrorMessage().c_str());

  return (void *)EE->getFunctionAddress(TraceName);
}

// Compile an IRTrace to executable code in memory.
//
// The trace to compile is passed in as two arrays of length Len. Then each
// (FuncName[I], BBs[I]) pair identifies the LLVM block at position `I` in the
// trace.
//
// Returns a pointer to the compiled function.
extern "C" void *__ykllvmwrap_irtrace_compile(char *FuncNames[], size_t BBs[],
                                              size_t Len, char *SecPtr,
                                              size_t SecSize) {
  ThreadSafeModule *ThreadAOTMod = getThreadAOTMod(SecPtr, SecSize);
  // Getting the module without acquiring the context lock is safe in this
  // instance since ThreadAOTMod is not shared between threads.
  Module *AOTMod = ThreadAOTMod->getModuleUnlocked();
  LLVMContext &JITContext = AOTMod->getContext();
  auto JITMod = new Module("", JITContext);
  uint64_t TraceIdx = NextTraceIdx.fetch_add(1);
  if (TraceIdx == numeric_limits<uint64_t>::max())
    errx(EXIT_FAILURE, "trace index counter overflowed");

  // Get var args from start_tracing call.
  auto Inputs = get_trace_inputs(AOTMod->getFunction(FuncNames[0]), BBs[0]);

  std::vector<Type *> InputTypes;
  for (auto Val : Inputs) {
    InputTypes.push_back(Val->getType());
  }

  // Create function to store compiled trace.
  string TraceName = string(TRACE_FUNC_PREFIX) + to_string(TraceIdx);
  llvm::FunctionType *FType =
      llvm::FunctionType::get(Type::getVoidTy(JITContext), InputTypes, false);
  llvm::Function *DstFunc = llvm::Function::Create(
      FType, Function::InternalLinkage, TraceName, JITMod);
  DstFunc->setCallingConv(CallingConv::C);

  // Create entry block and setup builder.
  auto DstBB = BasicBlock::Create(JITContext, "", DstFunc);
  llvm::IRBuilder<> Builder(JITContext);
  Builder.SetInsertPoint(DstBB);

  llvm::ValueToValueMapTy VMap;
  // Variables that are used (but not defined) inbetween start and stop tracing
  // need to be replaced with function arguments which the user passes into the
  // compiled trace. This loop creates a mapping from those original variables
  // to the function arguments of the compiled trace function.
  for (size_t Idx = 0; Idx != Inputs.size(); Idx++) {
    Value *OldVal = Inputs[Idx];
    Value *NewVal = DstFunc->getArg(Idx);
    assert(NewVal->getType()->isPointerTy());
    VMap[OldVal] = NewVal;
  }

  bool Tracing = false;
  std::vector<CallInst *> inlined_calls;
  CallInst *last_call = nullptr;

  // Iterate over the PT trace and stitch together all traced blocks.
  for (size_t Idx = 0; Idx < Len; Idx++) {
    auto FuncName = FuncNames[Idx];

    // Get a traced function so we can extract blocks from it.
    Function *F = AOTMod->getFunction(FuncName);
    if (!F)
      errx(EXIT_FAILURE, "can't find function %s", FuncName);

    // Skip to the correct block.
    auto It = F->begin();
    std::advance(It, BBs[Idx]);
    BasicBlock *BB = &*It;

    // Iterate over all instructions within this block and copy them over
    // to our new module.
    for (auto I = BB->begin(); I != BB->end(); I++) {
      // If we've returned from a call skip ahead to the instruction where we
      // left off.
      if (last_call != nullptr) {
        if (&*I == last_call) {
          last_call = nullptr;
        }
        continue;
      }
      if (isa<CallInst>(I)) {
        Function *CF = cast<CallInst>(&*I)->getCalledFunction();
        if (CF == nullptr)
          continue;

        // FIXME Strip storage of return value of __yktrace_start_tracing and
        // argument setup of __yktrace_stop_tracing.
        if (CF->getName() == YKTRACE_START) {
          Tracing = true;
          continue;
        } else if (CF->getName() == YKTRACE_STOP) {
          // FIXME Remove argument setup before __yktrace_stop_tracing call.
          Tracing = false;
        } else {
          // Skip remainder of this block and remember where we stopped so we
          // can continue tracing from this position after returning frome the
          // inlined call.
          // FIXME Deal with calls we cannot or don't want to inline.
          if (Tracing) {
            inlined_calls.push_back((CallInst *)&*I);
            break;
          }
        }
      }

      if (!Tracing)
        continue;

      if (llvm::isa<llvm::BranchInst>(I)) {
        // FIXME Replace all branch instruction with guards.
        continue;
      }

      if (isa<ReturnInst>(I)) {
        last_call = inlined_calls.back();
        inlined_calls.pop_back();
        // FIXME write return value to LHS of the call.
        continue;
      }

      // If execution reaches here, then the instruction I is to be copied into
      // JITMod. We now scan the instruction's operands checking that each is
      // defined in JITMod. Any variable not defined means that the
      // corresponding variable in AOTMod was instantiated prior to tracing.
      // Eventually these operands need to become inputs to the trace, but
      // until we have figured out how to do that, we simply allocate dummy
      // storage for them so that the module will verify and compile. Obviously
      // mutations to these dummies are not observed outside the trace code, so
      // this is strictly a FIXME.
      for (unsigned OpIdx = 0; OpIdx < I->getNumOperands(); OpIdx++) {
        Value *Op = I->getOperand(OpIdx);
        if (VMap[Op] == nullptr) {
          // Value is undefined in JITMod.
          Type *OpTy = Op->getType();
          if (isa<llvm::AllocaInst>(Op)) {
            // Value is a stack allocation, so make a dummy stack slot.
            Value *Alloca = Builder.CreateAlloca(
                OpTy->getPointerElementType(), OpTy->getPointerAddressSpace());
            VMap[Op] = Alloca;
          } else {
            if (OpTy->isIntegerTy()) {
              // Value is an integer constant, so leave it as is.
              // FIXME Extend this with other types as needed to get new
              // tests to run. Ultimately, find a better way to do this.
              VMap[Op] = Op;
              continue;
            } else {
              // Value is not a stack allocation or constant, so make a dummy
              // default value.
              // FIXME fails for meta-data types (when you build with -g).
              Value *NullVal = Constant::getNullValue(OpTy);
              VMap[Op] = NullVal;
            }
          }
        }
      }

      // Copy instruction over into the IR trace. Since the instruction
      // operands still reference values in the original bitcode, remap
      // the operands to point to new values within the IR trace.
      auto NewInst = &*I->clone();
      llvm::RemapInstruction(NewInst, VMap, RF_NoModuleLevelChanges);
      VMap[&*I] = NewInst;
      Builder.Insert(NewInst);
    }
  }
  Builder.CreateRetVoid();
#ifndef NDEBUG
  llvm::verifyModule(*JITMod, &llvm::errs());
#endif

  // Compile IR trace and return a pointer to its function.
  return compile_module(TraceName, JITMod);
}
