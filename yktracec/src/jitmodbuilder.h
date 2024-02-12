#ifndef __JITMODBUILDER_H
#define __JITMODBUILDER_H

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include <map>

// An unaligned virtual address.
#define YK_INVALID_ALIGNED_VADDR 0x1

using namespace llvm;

std::tuple<Module *, std::string, void *, size_t>
createModule(Module *AOTMod, char *FuncNames[], size_t BBs[], size_t TraceLen,
             void *CallStack, void *AOTValsPtr, size_t AOTValsLen,
             uintptr_t *Promotions, size_t PromotionsLen);
#endif
