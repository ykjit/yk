#ifndef __JITMODBUILDER_H
#define __JITMODBUILDER_H

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"

// An invalid virtual address for a function to begin at.
#define YK_INVALID_FUNC_VADDR 0x1

using namespace llvm;

std::tuple<Module *, std::string, std::map<GlobalValue *, void *>>
createModule(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t TraceLen,
             char *FAddrKeys[], void *FAddrVals[], size_t FAddrLen);
#ifdef YK_TESTING
std::tuple<Module *, std::string, std::map<GlobalValue *, void *>>
createModuleForTraceCompilerTests(Module *AOTMod, char *FuncNames[],
                                  size_t BBs[], size_t TraceLen,
                                  char *FAddrKeys[], void *FAddrVals[],
                                  size_t FAddrLen);
#endif // YK_TESTING
#endif
