// LLVM-related C++ code wrapped in the C ABI for calling from Rust.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <llvm/DebugInfo/Symbolize/Symbolize.h>
#include <string.h>
#include <link.h>

using namespace llvm;
using namespace llvm::symbolize;

extern "C" LLVMSymbolizer *__yk_symbolizer_new() {
    return new LLVMSymbolizer;
}

extern "C" void __yk_symbolizer_free(LLVMSymbolizer *Symbolizer) {
    delete Symbolizer;
}

// Finds the name of a code symbol from a virtual address.
// The caller is responsible for freeing the returned (heap-allocated) C string.
extern "C" char *__yk_symbolizer_find_code_sym(LLVMSymbolizer *Symbolizer, const char *Obj, uint64_t Off) {
    object::SectionedAddress Mod{Off, object::SectionedAddress::UndefSection};
    auto LineInfo = Symbolizer->symbolizeCode(Obj, Mod);
    if (auto Err = LineInfo.takeError()) {
        return NULL;
    }

    // OPTIMISE_ME: get rid of heap allocation.
    return strdup(LineInfo->FunctionName.c_str());
}

/// Compiles an IRTrace to executable code in memory.
//
// The trace to compile is passed in as two arrays of length Len. Then each
// (FuncName[I], BBs[I]) pair identifies the LLVM block at position `I` in the
// trace.
extern "C" void __ykllvmwrap_irtrace_compile(char *FuncNames[], size_t BBs[], size_t Len) {
    for (size_t Idx = 0; Idx < Len; Idx++) {
        // FIXME populate.
        //auto FuncName = FuncNames[Idx];
        //auto BB = BBs[Idx];
    }
}
