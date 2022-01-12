#ifndef YK_H
#define YK_H

#include <stdint.h>

// A `Location` stores state that the meta-tracer needs to identify hot loops
// and run associated machine code. This is a C mirror of `ykrt::Location`.
//
// Each position in the end user's program that may be a control point (i.e.
// the possible start of a trace) must have an associated `Location`. The
// `Location` does not need to be at a stable address in memory and can be
// freely moved.
//
// Program positions that can't be control points don't need an associated
// `Location`. For interpreters that can't (or don't want) to be as selective,
// a simple (if moderately wasteful) mechanism is for every bytecode or AST
// node to have its own `Location` (even for bytecodes or nodes that can't be
// control points).
typedef struct {
  uintptr_t state;
} YkLocation;

#if defined(__x86_64)
typedef uint32_t YkHotThreshold;
#else
#error Unable to determine type of HotThreshold
#endif

// Notify yk that an iteration of an interpreter loop is about to start. The
// argument passed uniquely identifies the current location in the user's
// program. A call to this function may cause yk to start/stop tracing, or to
// execute JITted code.
void yk_control_point(YkLocation *);

// Create a new `Location`.
//
// Note that a `Location` created by this call must not simply be discarded:
// if no longer wanted, it must be passed to `yk_location_drop` to allow
// appropriate clean-up.
YkLocation yk_location_new(void);

// Clean-up a `Location` previously created by `yk_new_location`. The
// `Location` must not be further used after this call or undefined behaviour
// will occur.
void yk_location_drop(YkLocation);

void yk_stopgap(void *addr, uintptr_t size, uintptr_t retaddr, void *rsp);

#if defined(__x86_64)
__attribute__((naked)) void __llvm_deoptimize(void *addr, uintptr_t size) {
  // Push all registers to the stack before they can be clobbered, so that we
  // can find their values after parsing in the stackmap. The order in which
  // we push the registers is equivalent to the Sys-V x86_64 ABI, which the
  // stackmap format uses as well. This function has the "naked" attribute to
  // keep the optimiser from generating the function prologue which messes
  // with the RSP value of the previous stack frame (this value is often
  // referenced by the stackmap).

  __asm__ volatile(
      ".intel_syntax\n"
      // Save registers to the stack.
      // FIXME: Add other registers that may be referenced by the stackmap.
      "push rsp\n"
      "push rbp\n"
      "push rdi\n"
      "push rsi\n"
      "push rbx\n"
      "push rcx\n"
      "push rdx\n"
      "push rax\n"
      // Now we need to call yk_stopgap. The arguments need to be in RDI,
      // RSI, RDX, and RCX. The first two arguments (stackmap address and
      // stackmap size) are already where they need to be as we are just
      // forwarding them from the current function's arguments. The remaining
      // arguments (return address and current stack pointer) need to be in
      // RDX and RCX. The return address was at [RSP] before the above
      // pushes, so to find it we need to offset 8 bytes per push.
      "mov rdx, [rsp+64]\n"
      "mov rcx, rsp\n"
      "sub rsp, 8\n"
      "call yk_stopgap\n"
      "add rsp, 64\n"
      "ret");
}
#else
#error Deoptimise function not implemented.
#endif

#endif
