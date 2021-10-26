// ignore: broken during new control point design
// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     ...
//     ...call i32 asm "mov $$5, $0"...
//     ...

// Check that we can handle inline asm properly.

#include <assert.h>
#include <stdlib.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res = 0;
  __yktrace_start_tracing(HW_TRACING, &res);
#ifdef __x86_64__
  // Stores the constant 5 into `res`.
  asm("mov $5, %0"
      : "=r"(res) // outputs.
      :           // inputs.
      :           // clobbers.
  );
#else
#error unknown platform
#endif
  void *tr = __yktrace_stop_tracing();
  assert(res == 5);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *) = (void (*)(int *))ptr;
  int res2 = 0;
  func(&res2);
  assert(res2 == 5);

  return (EXIT_SUCCESS);
}
