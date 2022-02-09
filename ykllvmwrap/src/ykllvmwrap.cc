// LLVM-related C++ code wrapped in the C ABI for calling from Rust.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <err.h>
#include <link.h>
#include <stdlib.h>
#include <string.h>

#include "jitmodbuilder.h"
#include "memman.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::symbolize;
using namespace std;

struct BitcodeSection {
  // Pointer to the start of the LLVM bitcode section.
  void *data;
  // The length of the LLVM bitcode section, in bytes.
  size_t len;
};

// An annotator for `Module::print()` which adds debug location lines.
class DebugAnnotationWriter : public AssemblyAnnotationWriter {
  string LastLineInfo;

public:
  void emitInstructionAnnot(const Instruction *I, formatted_raw_ostream &OS) {
    const DebugLoc &DL = I->getDebugLoc();
    string LineInfo;
    raw_string_ostream RSO(LineInfo);
    DL.print(RSO);
    if ((!LineInfo.empty()) && (LineInfo != LastLineInfo)) {
      string FuncName = "<unknown-func>";
      const MDNode *Scope = DL.getInlinedAtScope();
      if (auto *SP = getDISubprogram(Scope))
        FuncName.assign(SP->getName().data());
      OS << "  ; " << FuncName << "() " << LineInfo << "\n";
      LastLineInfo = LineInfo;
    }
  }
};

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

// Flag used to ensure that LLVM is initialised only once.
once_flag LLVMInitialised;

#ifndef NDEBUG
// Left trim (in-place) the character `C` from the string `S`.
void lTrim(string &S, const char C) {
  S.erase(0, std::min(S.find_first_not_of(C), S.size() - 1));
}

// Dumps an LLVM Value to a string and trims leading whitespace.
void dumpValueToString(Value *V, string &S) {
  raw_string_ostream RSO(S);
  V->print(RSO);
  lTrim(S, ' ');
}
#endif

enum DebugIR {
  AOT,
  JITPreOpt,
  JITPostOpt,
};

class DebugIRPrinter {
private:
  bitset<3> toPrint;

  const char *debugIRStr(DebugIR IR) {
    switch (IR) {
    case DebugIR::AOT:
      return "aot";
    case DebugIR::JITPreOpt:
      return "jit-pre-opt";
    case DebugIR::JITPostOpt:
      return "jit-post-opt";
    default:
      errx(EXIT_FAILURE, "unreachable");
    }
  }

public:
  DebugIRPrinter() {
    char *Env = std::getenv("YKD_PRINT_IR");
    char *Val;
    while ((Val = strsep(&Env, ",")) != nullptr) {
      if (strcmp(Val, "aot") == 0)
        toPrint.set(DebugIR::AOT);
      else if (strcmp(Val, "jit-pre-opt") == 0)
        toPrint.set(DebugIR::JITPreOpt);
      else if (strcmp(Val, "jit-post-opt") == 0)
        toPrint.set(DebugIR::JITPostOpt);
      else
        errx(EXIT_FAILURE, "invalid parameter for YKD_PRINT_IR: '%s'", Val);
    }
  }

  void print(enum DebugIR IR, Module *M) {
    if (toPrint[IR]) {
      string PrintMode = debugIRStr(IR);
      errs() << "--- Begin " << PrintMode << " ---\n";
      DebugAnnotationWriter DAW;
      M->print(errs(), &DAW);
      errs() << "--- End " << PrintMode << " ---\n";
    }
  }
};

// Initialise LLVM for JIT compilation. This must be executed exactly once.
void initLLVM(void *Unused) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
}

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
void loadAOTMod(struct BitcodeSection &Bitcode) {
  auto Sf = StringRef((const char *)Bitcode.data, Bitcode.len);
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
ThreadSafeModule *getThreadAOTMod(struct BitcodeSection &Bitcode) {
  std::call_once(GlobalAOTModLoaded, loadAOTMod, Bitcode);
  if (!ThreadAOTModInitialized) {
    ThreadAOTMod = cloneToNewContext(GlobalAOTMod);
    ThreadAOTModInitialized = true;
  }
  return &ThreadAOTMod;
}

// Compile a module in-memory and return a pointer to its function.
extern "C" void *compileModule(string TraceName, Module *M,
                               map<GlobalValue *, void *> GlobalMappings) {
  std::call_once(LLVMInitialised, initLLVM, nullptr);

  // Use our own memory manager to keep track of stackmap address.
  AllocMem SMR;
  MemMan *memman = new MemMan();
  memman->setStackMapStore(&SMR);

  auto MPtr = std::unique_ptr<Module>(M);
  string ErrStr;
  ExecutionEngine *EE =
      EngineBuilder(std::move(MPtr))
          .setEngineKind(EngineKind::JIT)
          .setMemoryManager(std::unique_ptr<MCJITMemoryManager>(memman))
          .setErrorStr(&ErrStr)
          .create();

  if (EE == nullptr)
    errx(EXIT_FAILURE, "Couldn't compile trace: %s", ErrStr.c_str());

  for (auto GM : GlobalMappings) {
    EE->addGlobalMapping(GM.first, GM.second);
  }

  EE->finalizeObject();
  if (EE->hasError())
    errx(EXIT_FAILURE, "Couldn't compile trace: %s",
         EE->getErrorMessage().c_str());

  // Allocate space for compiled trace address, stackmap address, and stackmap
  // size.
  // FIXME This is a temporary hack until the redesigned hot location is up.
  uintptr_t *ptr = (uintptr_t *)malloc(sizeof(uintptr_t) * 3);
  ptr[0] = EE->getFunctionAddress(TraceName);
  ptr[1] = (uintptr_t)SMR.Ptr;
  ptr[2] = SMR.Size;

  return ptr;
}

// Compile an IRTrace to executable code in memory.
//
// The trace to compile is passed in as two arrays of length Len. Then each
// (FuncName[I], BBs[I]) pair identifies the LLVM block at position `I` in the
// trace.
//
// Returns a pointer to the compiled function.
template <typename FN>
void *compileIRTrace(FN Func, char *FuncNames[], size_t BBs[], size_t TraceLen,
                     char *FAddrKeys[], void *FAddrVals[], size_t FAddrLen,
                     void *BitcodeData, size_t BitcodeLen) {
  DebugIRPrinter DIP;

  struct BitcodeSection Bitcode = {BitcodeData, BitcodeLen};
  ThreadSafeModule *ThreadAOTMod = getThreadAOTMod(Bitcode);
  // Getting the module without acquiring the context lock is safe in this
  // instance since ThreadAOTMod is not shared between threads.
  Module *AOTMod = ThreadAOTMod->getModuleUnlocked();

  DIP.print(DebugIR::AOT, AOTMod);

  Module *JITMod;
  std::string TraceName;
  std::map<GlobalValue *, void *> GlobalMappings;
  std::tie(JITMod, TraceName, GlobalMappings) =
      Func(AOTMod, FuncNames, BBs, TraceLen, FAddrKeys, FAddrVals, FAddrLen);

  DIP.print(DebugIR::JITPreOpt, JITMod);
#ifndef NDEBUG
  llvm::verifyModule(*JITMod, &llvm::errs());
#endif

  // The MCJIT code-gen does no optimisations itself, so we must do it
  // ourselves.
  PassManagerBuilder Builder;
  Builder.OptLevel = 2; // FIXME Make this user-tweakable.
  legacy::FunctionPassManager FPM(JITMod);
  Builder.populateFunctionPassManager(FPM);
  for (Function &F : *JITMod)
    FPM.run(F);

  DIP.print(DebugIR::JITPostOpt, JITMod);

  // Compile IR trace and return a pointer to its function.
  return compileModule(TraceName, JITMod, GlobalMappings);
}

extern "C" void *__ykllvmwrap_irtrace_compile(
    char *FuncNames[], size_t BBs[], size_t TraceLen, char *FAddrKeys[],
    void *FAddrVals[], size_t FAddrLen, void *BitcodeData, size_t BitcodeLen) {
  return compileIRTrace(createModule, FuncNames, BBs, TraceLen, FAddrKeys,
                        FAddrVals, FAddrLen, BitcodeData, BitcodeLen);
}

#ifdef YK_TESTING
extern "C" void *__ykllvmwrap_irtrace_compile_for_tc_tests(
    char *FuncNames[], size_t BBs[], size_t TraceLen, char *FAddrKeys[],
    void *FAddrVals[], size_t FAddrLen, void *BitcodeData, size_t BitcodeLen) {
  return compileIRTrace(createModuleForTraceDriver, FuncNames, BBs, TraceLen,
                        FAddrKeys, FAddrVals, FAddrLen, BitcodeData,
                        BitcodeLen);
}
#endif
