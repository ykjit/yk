#ifndef __MEMMAN_H
#define __MEMMAN_H

#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"

using namespace llvm;

struct AllocMem {
  uint8_t *Ptr;
  uintptr_t Size;
};

class MemMan : public RTDyldMemoryManager {
  std::vector<AllocMem> code;
  std::vector<AllocMem> data;

public:
  MemMan();
  ~MemMan() override;

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override;
  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool isReadOnly) override;
  bool finalizeMemory(std::string *ErrMsg) override;
  void freeMemory();
};

#endif
