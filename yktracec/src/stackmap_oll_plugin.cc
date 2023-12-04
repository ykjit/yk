// LLVM ORC Object Linking Layer plugin for preserving and locating stackmaps.

#include "stackmap_oll_plugin.h"

#ifdef __linux__
const char *SMSectionName = ".llvm_stackmaps";
#else
#error unknown stackmap section name for this platform
#endif

using namespace jitlink;

void StackmapOLLPlugin::modifyPassConfig(MaterializationResponsibility &MR,
                                         jitlink::LinkGraph &LG,
                                         jitlink::PassConfiguration &Config) {

  // Prevent the stackmap records (if we find any) from being GC'd.
  //
  // Note that we don't flag failure if we can't find a stackmap section
  // because the `trace_compiler` test suite doesn't use stackmaps and comes
  // through here.
  auto NoGCStackmapsPass = [](jitlink::LinkGraph &G) {
    jitlink::Section *SMS = G.findSectionByName(SMSectionName);
    if (SMS != nullptr) {
      for (Symbol *Sym : SMS->symbols()) {
        if (Sym->hasName() && Sym->getName() == "__LLVM_StackMaps") {
          Sym->setLive(true);
        }
      }
    }
    return static_cast<Error>(Error::success());
  };
  Config.PrePrunePasses.push_back(NoGCStackmapsPass);

  // Extract the address and size of the stackmaps section.
  auto GetStackmapsExtentPass = [this](jitlink::LinkGraph &G) {
    jitlink::Section *SMS = G.findSectionByName(SMSectionName);
    if (SMS != nullptr) {
      SectionRange SR(*SMS);
      assert(!SR.empty());
      uintptr_t Start =
          reinterpret_cast<uintptr_t>(SR.getStart().toPtr<void *>());
      uintptr_t End = reinterpret_cast<uintptr_t>(SR.getEnd().toPtr<void *>());
      SMExtent = {Start, End};
    }
    return static_cast<Error>(Error::success());
  };
  Config.PostFixupPasses.push_back(GetStackmapsExtentPass);
}

Error StackmapOLLPlugin::notifyFailed(MaterializationResponsibility &MR) {
  return Error::success();
}

Error StackmapOLLPlugin::notifyRemovingResources(JITDylib &JD, ResourceKey K) {
  return Error::success();
}

void StackmapOLLPlugin::notifyTransferringResources(JITDylib &JD,
                                                    ResourceKey DstKey,
                                                    ResourceKey SrcKey) {}
