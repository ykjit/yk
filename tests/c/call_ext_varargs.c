// # static strings: https://github.com/ykjit/yk/issues/382
// ignore-if: true
// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//    ...
//    ...call i32 (i8*, ...) @printf...
//    ...
//    declare i32 @printf(i8* %0, ...)
//    ...
//   stdout:
//     abc123
//     abc101112

// Check that calling an external function works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int x = 1;
  __yktrace_start_tracing(HW_TRACING, &x);
  printf("abc%d%d%d\n", x, x + 1, x + 2);
  void *tr = __yktrace_stop_tracing();

  x = 10;
  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *) = (void (*)(int *))ptr;
  func(&x);

  return (EXIT_SUCCESS);
}
