// LLVM-related C++ code wrapped in the C ABI for calling from Rust.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <llvm/DebugInfo/Symbolize/Symbolize.h>
#include <string.h>

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
extern "C" char *__yk_symbolizer_find_code_sym(LLVMSymbolizer *Symbolizer, void *Vaddr) {
    // Find which of the loaded objects holds this virtual address.
    Dl_info Info;
    if (dladdr(Vaddr, &Info) == 0 ) {
        return NULL;
    }

    // Find the corresponding offset of vaddr from the start of the object.
    auto Offs = (uintptr_t) Vaddr - (uintptr_t) Info.dli_fbase;

    // Use the LLVM symbolizer to find the symbol name. Note that we don't
    // simply rely on `Info.dli_sname`, as `dl_addr()` can only find a subset
    // of symbols -- those exported -- and we don't want users to have to link
    // with --export-dynamic.
    auto LineInfo = Symbolizer->symbolizeCode(Info.dli_fname,
        {Offs, object::SectionedAddress::UndefSection});
    if (auto err = LineInfo.takeError()) {
        return NULL;
    }

    // OPTIMISE_ME: get rid of heap allocation.
    return strdup(LineInfo->FunctionName.c_str());
}
