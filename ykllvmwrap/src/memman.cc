#include <err.h>
#include <sys/mman.h>

#include "memman.h"

using namespace llvm;

uint8_t *allocateSection(uintptr_t Size, unsigned Alignment,
                         std::vector<AllocMem> *Vec) {
  uintptr_t RequiredSize = Alignment * ((Size + Alignment - 1) / Alignment + 1);
  auto Ptr = (uint8_t *)mmap(0, RequiredSize, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(Ptr != MAP_FAILED);
  Vec->push_back({Ptr, RequiredSize});
  return Ptr;
}

MemMan::MemMan(){};
MemMan::~MemMan(){};

uint8_t *MemMan::allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                     unsigned SectionID,
                                     StringRef SectionName) {
  return allocateSection(Size, Alignment, &code);
}

uint8_t *MemMan::allocateDataSection(uintptr_t Size, unsigned Alignment,
                                     unsigned SectionID, StringRef SectionName,
                                     bool isReadOnly) {
  uint8_t *Ptr = allocateSection(Size, Alignment, &data);
  // FIXME: Linux only. E.g. on MachO this is called "__llvm_stackmaps".
  if (SectionName == ".llvm_stackmaps") {
    SMR->Ptr = Ptr;
    SMR->Size = Size;
  }
  return Ptr;
}

bool MemMan::finalizeMemory(std::string *ErrMsg) {
  for (const AllocMem &Value : code) {
    if (mprotect(Value.Ptr, Value.Size, PROT_READ | PROT_EXEC) == -1) {
      errx(EXIT_FAILURE, "Failed to make code executable.");
    }
  }
  for (const AllocMem &Value : data) {
    if (mprotect(Value.Ptr, Value.Size, PROT_READ) == -1) {
      errx(EXIT_FAILURE, "Failed to make data read-only.");
    }
  }
  return true;
}

void MemMan::freeMemory() {
  for (const AllocMem &Value : code) {
    if (munmap(Value.Ptr, Value.Size) == -1) {
      errx(EXIT_FAILURE, "Failed to unmap memory.");
    }
  }
  for (const AllocMem &Value : data) {
    if (munmap(Value.Ptr, Value.Size) == -1) {
      errx(EXIT_FAILURE, "Failed to unmap memory.");
    }
  }
}

void MemMan::setStackMapStore(AllocMem *Ptr) { SMR = Ptr; }
