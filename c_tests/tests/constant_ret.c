// ignore: broken during new control point design
// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     ...
//     define internal void @__yk_compiled_trace_0(i32* %0) {
//       ...
//       store i32 30, i32* %0, align 4...
//       ...

// Check that returning a constant value from a traced function works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline)) int f() { return 30; }

int main(int argc, char **argv) {
  int res = 0;
  __yktrace_start_tracing(HW_TRACING, &res);
  res = f();
  void *tr = __yktrace_stop_tracing();
  assert(res == 30);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *) = (void (*)(int *))ptr;
  int res2 = 0;
  func(&res2);
  assert(res2 == 30);

  return (EXIT_SUCCESS);
}
