#ifndef __STACKMAP_OLL_PLUGIN_H
#define __STACKMAP_OLL_PLUGIN_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

#define SM_SYM_NAME "__yk_stackmap_section"

using namespace llvm;
using namespace llvm::orc;

// Describes the virtual address range of a section.
struct SectionExtent {
  uintptr_t Begin;
  uintptr_t End;
};

class StackmapOLLPlugin : public ObjectLinkingLayer::Plugin {
  // The virtual address range of the stackmap section (once known).
  std::optional<SectionExtent> &SMExtent;

public:
  StackmapOLLPlugin(std::optional<SectionExtent> &SMExtent)
      : SMExtent(SMExtent) {}

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &LG,
                        jitlink::PassConfiguration &Config) override;

  Error notifyFailed(MaterializationResponsibility &MR) override;
  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override;
  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override;
};

#endif
