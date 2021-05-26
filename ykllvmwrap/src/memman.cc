#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <sys/mman.h>

using namespace llvm;

struct AllocMem {
  uint8_t *Ptr;
  uintptr_t Size;
};

class MemMan : public RTDyldMemoryManager {
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

private:
  std::vector<AllocMem> code;
  std::vector<AllocMem> data;
};

MemMan::MemMan() {}
MemMan::~MemMan() {}

uint8_t *alloc_mem(uintptr_t Size, unsigned Alignment,
                   std::vector<AllocMem> *Vec) {
  uintptr_t RequiredSize = Alignment * ((Size + Alignment - 1) / Alignment + 1);
  auto Ptr = (unsigned char *)mmap(0, RequiredSize, PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(Ptr != MAP_FAILED);
  Vec->push_back({Ptr, RequiredSize});
  return Ptr;
}

uint8_t *MemMan::allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                     unsigned SectionID,
                                     StringRef SectionName) {
  return alloc_mem(Size, Alignment, &code);
}

uint8_t *MemMan::allocateDataSection(uintptr_t Size, unsigned Alignment,
                                     unsigned SectionID, StringRef SectionName,
                                     bool isReadOnly) {
  return alloc_mem(Size, Alignment, &data);
}

bool MemMan::finalizeMemory(std::string *ErrMsg) {
  for (const AllocMem &Value : code) {
    if (mprotect(Value.Ptr, Value.Size, PROT_READ | PROT_EXEC) == -1) {
      errx(EXIT_FAILURE, "Can't make allocated memory executable.");
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
