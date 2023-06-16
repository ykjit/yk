// LLVM-related C++ code wrapped in the C ABI for calling from Rust.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <dlfcn.h>
#include <err.h>
#include <filesystem>
#include <link.h>
#include <optional>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jitmodbuilder.h"
#include "memman.h"

// When we create a compilation unit for our JIT debug info, LLVM forces us to
// choose a language from one of those "recognised" by the DWARF spec (see
// `Dwarf.def` in the LLVM sources). If you choose a value out of range, then an
// LLVM assertion will fail, so we have to pretend to be another language.
#define YKJIT_DWARF_LANG dwarf::DW_LANG_Cobol74

using namespace llvm;
using namespace llvm::orc;
using namespace std;

struct BitcodeSection {
  // Pointer to the start of the LLVM bitcode section.
  void *data;
  // The length of the LLVM bitcode section, in bytes.
  size_t len;
};

// If possible, return a string describing the location of an instruction in
// the AOT-compiled interpreter source code.
//
// Since it is undesirable to show duplicated debug locations, the callee
// should pass in the last annotation that it saw (in `LastAnnot`), and this
// function will return `None` if the debug location string was the same as
// before.
//
// The string is prefixed with a `;` so as to look like a comment in a `.ll`
// bytecode file.
optional<string> getSourceLevelInstructionAnnotation(const Instruction *I,
                                                     string &LastAnnot) {
  const DebugLoc &DL = I->getDebugLoc();
  string LineInfo;
  raw_string_ostream RSO(LineInfo);
  DL.print(RSO);
  if (LineInfo.empty())
    return optional<string>();

  string FuncName = "<unknown-func>";
  const MDNode *Scope = DL.getInlinedAtScope();
  if (auto *SP = getDISubprogram(Scope))
    FuncName.assign(SP->getName().data());
  string Line = string("; ") + FuncName + "() " + LineInfo;

  // We only want to show an annotation when the location has changed.
  if (Line == LastAnnot)
    return optional<string>();

  LastAnnot = Line;
  return Line;
}

// An annotator for `Module::print()` which adds debug location lines at the
// source-level. In other words, if the interpreter being JITted is written in
// C, then we will be adding lines to the output which allude to where we are
// in the C code.
class DebugAnnotationWriter : public AssemblyAnnotationWriter {
  string LastLineInfo;

public:
  void emitInstructionAnnot(const Instruction *I, formatted_raw_ostream &OS) {
    optional<string> LineInfo =
        getSourceLevelInstructionAnnotation(I, LastLineInfo);
    if (LineInfo.has_value())
      OS << "  " << LineInfo << "\n";
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

// Load the GlobalAOTMod.
//
// This must only be called from getAOTMod() for correct synchronisation.
void loadAOTMod(struct BitcodeSection &Bitcode) {
  auto Sf = StringRef((const char *)Bitcode.data, Bitcode.len);
  auto Mb = MemoryBufferRef(Sf, "");
  SMDiagnostic Error;
  ThreadSafeContext AOTCtx = std::make_unique<LLVMContext>();
  auto M = parseIR(Mb, Error, *AOTCtx.getContext());
  if (!M) {
    Error.print("", errs(), false);
    errx(EXIT_FAILURE, "Can't load module.");
  }
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

// Exposes `getThreadAOTMod` so we can get a thread-safe copy of the
// AOT IR from within Rust.
extern "C" LLVMModuleRef
LLVMGetThreadSafeModule(struct BitcodeSection &Bitcode) {
  ThreadSafeModule *ThreadAOTMod = getThreadAOTMod(Bitcode);
  Module *AOTMod = ThreadAOTMod->getModuleUnlocked();
  return llvm::wrap(AOTMod);
}

// Compile a module in-memory and return a pointer to its function.
extern "C" void *compileModule(string TraceName, Module *M,
                               map<GlobalValue *, void *> GlobalMappings,
                               void *LiveAOTVals, size_t GuardCount) {
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
    // If a value now has no parent, then it was optimised out and LLVM will be
    // unhappy if we try to regster a global mapping for it.
    if (GM.first->getParent() != nullptr)
      EE->addGlobalMapping(GM.first, GM.second);
  }

  EE->finalizeObject();
  if (EE->hasError())
    errx(EXIT_FAILURE, "Couldn't compile trace: %s",
         EE->getErrorMessage().c_str());

  // Allocate space for compiled trace address, stackmap address, and stackmap
  // size.
  // FIXME This is a temporary hack until the redesigned hot location is up.
  uintptr_t *ptr = (uintptr_t *)malloc(sizeof(uintptr_t) * 5);
  ptr[0] = EE->getFunctionAddress(TraceName);
  ptr[1] = reinterpret_cast<uintptr_t>(SMR.Ptr);
  ptr[2] = SMR.Size;
  ptr[3] = reinterpret_cast<uintptr_t>(LiveAOTVals);
  ptr[4] = GuardCount;

  return ptr;
}

/// Write the string `S` in its entirety to the file descriptor `FD`.
void writeString(int FD, string S) {
  const char *Buf = S.c_str();

  size_t Remain = strlen(Buf);

  // Otherwise we can't reliably know how much was written!
  if (Remain > SSIZE_MAX)
    errx(EXIT_FAILURE, "Remain > SSIZE_MAX");

  while (Remain) {
    ssize_t Wrote = write(FD, Buf, Remain);
    if (Wrote == -1) {
      if (errno == EINTR)
        continue;
      else
        err(EXIT_FAILURE, "write");
    }

    assert(Wrote >= 0 && static_cast<size_t>(Wrote) <= Remain);

    Remain -= Wrote;
    Buf += Wrote;
  }
}

/// Add debugging metadata to the module to help with debugging JITted code.
///
/// This works by iterating over the IR instructions of the JITted code and:
///
///  a) writing faked source code lines into the temporary file named by `Path`
///     (whose file descriptor `FD` is open and ready for writing), and...
///
///  b) Add debug locations to the IR instructions that point to the relevant
///     lines in the temporary file.
///
/// This means that debuggers (that conform to gdb's JIT interface:
/// https://sourceware.org/gdb/current/onlinedocs/gdb/JIT-Interface.html) can
/// show locations in the fake source code as you step over the machine code of
/// the trace.
///
/// Note that the temporary file is free-form and doesn't have to be a valid
/// source file in any particular language, so we can add any text we
/// like to the file, if we think that would aid debugging.
void rewriteDebugInfo(Module *M, string TraceName, int FD,
                      const filesystem::path &Path) {
  Function *JITFunc = M->getFunction(TraceName);
  assert(JITFunc);

  // Create a debug subprogram for the `JITFunc`.
  DIBuilder DIB(*M);
  DIFile *DF =
      DIB.createFile(Path.filename().string(), Path.parent_path().string());
  DIB.createCompileUnit(YKJIT_DWARF_LANG, DF, "ykjit", true, "", 0);
  DISubroutineType *ST = DIB.createSubroutineType({});
  DISubprogram *DS =
      DIB.createFunction(DF, TraceName, TraceName, DF, 1, ST, 1,
                         DINode::FlagZero, DISubprogram::SPFlagDefinition);
  JITFunc->setSubprogram(DS);

  // For each instruction in the trace IR, emit a human-readable version of the
  // instruction into the temporary file and update the instruction's debug
  // location to point to this line.
  size_t LineNo = 1;
  string LastSrcAnnot;
  for (BasicBlock &BB : *JITFunc) {
    // Emit a label for the block.
    Twine BBLabel = string("\n") + BB.getName() + ": \n";
    writeString(FD, BBLabel.str());
    LineNo += 2;
    for (Instruction &I : BB) {
      // See if there's an "interpreter-source-level" annotation we can prepend.
      // This makes it easier to see which (approximate) part of the AOT code
      // the trace IR came from.
      optional<string> MaybeSrcAnnot =
          getSourceLevelInstructionAnnotation(&I, LastSrcAnnot);
      if (MaybeSrcAnnot.has_value()) {
        writeString(FD, string("  ") + MaybeSrcAnnot.value() + "\n");
        LineNo++;
      }

      // Appends the stringified instruction.
      string IS;
      raw_string_ostream SS(IS);
      I.print(SS);
      writeString(FD, IS + "\n");

      // Update debug location.
      DILocation *DIL = DILocation::get(DS->getContext(), LineNo, 0, DS);
      I.setDebugLoc(DebugLoc(DIL));
      LineNo++;
    }
  }

  DIB.finalize();
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
                     void *BitcodeData, size_t BitcodeLen, int DebugInfoFD,
                     char *DebugInfoPath) {
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
  void *AOTMappingVec;
  size_t GuardCount;
  std::tie(JITMod, TraceName, GlobalMappings, AOTMappingVec, GuardCount) =
      Func(AOTMod, FuncNames, BBs, TraceLen, FAddrKeys, FAddrVals, FAddrLen);

  // If we failed to build the trace, return null.
  if (JITMod == nullptr) {
    return nullptr;
  }

  DIP.print(DebugIR::JITPreOpt, JITMod);
#ifndef NDEBUG
  llvm::verifyModule(*JITMod, &llvm::errs());
#endif

  // The MCJIT code-gen does no optimisations itself, so we must do it
  // ourselves.
  PassManagerBuilder Builder;
  Builder.OptLevel = 2; // FIXME Make this user-tweakable.
  legacy::FunctionPassManager FPM(JITMod);
  legacy::PassManager MPM;
  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(MPM);
  MPM.run(*JITMod);

  DIP.print(DebugIR::JITPostOpt, JITMod);

  // If `DebugInfoFD` is -1, then trace debuginfo was not requested.
  if (DebugInfoFD != -1)
    rewriteDebugInfo(JITMod, TraceName, DebugInfoFD,
                     filesystem::path(DebugInfoPath));

  // Compile IR trace and return a pointer to its function.
  return compileModule(TraceName, JITMod, GlobalMappings, AOTMappingVec,
                       GuardCount);
}

extern "C" void *__yktracec_irtrace_compile(
    char *FuncNames[], size_t BBs[], size_t TraceLen, char *FAddrKeys[],
    void *FAddrVals[], size_t FAddrLen, void *BitcodeData, uint64_t BitcodeLen,
    int DebugInfoFD, char *DebugInfoPath) {
  return compileIRTrace(createModule, FuncNames, BBs, TraceLen, FAddrKeys,
                        FAddrVals, FAddrLen, BitcodeData, BitcodeLen,
                        DebugInfoFD, DebugInfoPath);
}

#ifdef YK_TESTING
extern "C" void *__yktracec_irtrace_compile_for_tc_tests(
    char *FuncNames[], size_t BBs[], size_t TraceLen, char *FAddrKeys[],
    void *FAddrVals[], size_t FAddrLen, void *BitcodeData, uint64_t BitcodeLen,
    int DebugInfoFD, char *DebugInfoPath) {
  return compileIRTrace(createModuleForTraceCompilerTests, FuncNames, BBs,
                        TraceLen, FAddrKeys, FAddrVals, FAddrLen, BitcodeData,
                        BitcodeLen, DebugInfoFD, DebugInfoPath);
}
#endif
