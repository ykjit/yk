// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     ...
//     ...store i32 3, i32* %0...
//     ...

// Test indirect calls where we don't have IR for the callee.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int bar(size_t (*func)(const char *)) {
  int a = func("abc");
  return a;
}

int main(int argc, char **argv) {
  int z = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &z);
  z = bar(strlen);
  NOOPT_VAL(z);
  void *tr = __yktrace_stop_tracing(tt);
  assert(z == 3);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  assert(output == 3);

  return (EXIT_SUCCESS);
}
