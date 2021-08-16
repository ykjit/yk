// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     ...
//     define internal void @__yk_compiled_trace_0(i32* %0) {
//        ...
//        store i32 2, i32* %0, align 4...
//        ret void
//     }
//     ...

// Check that basic trace compilation works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  res = 2;
  void *tr = __yktrace_stop_tracing(tt);
  assert(res == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *) = (void (*)(int *))ptr;
  int res2 = 0;
  func(&res2);
  assert(res2 == 2);

  return (EXIT_SUCCESS);
}
