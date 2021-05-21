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

extern "C" LLVMSymbolizer *__yk_llvmwrap_symbolizer_new() {
    return new LLVMSymbolizer;
}

extern "C" void __yk_llvmwrap_symbolizer_free(LLVMSymbolizer *Symbolizer) {
    delete Symbolizer;
}

// Finds the name of a code symbol from a virtual address.
// The caller is responsible for freeing the returned (heap-allocated) C string.
extern "C" char *__yk_llvmwrap_symbolizer_find_code_sym(
    LLVMSymbolizer *Symbolizer,
    const char *Obj,
    uint64_t Off)
{
    object::SectionedAddress Mod{Off, object::SectionedAddress::UndefSection};
    auto LineInfo = Symbolizer->symbolizeCode(Obj, Mod);
    if (auto Err = LineInfo.takeError()) {
        return NULL;
    }

    // OPTIMISE_ME: get rid of heap allocation.
    return strdup(LineInfo->FunctionName.c_str());
}
