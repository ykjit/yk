// ignore: broken during new control point design
// Compiler:
// Run-time:

// Ensure that an LLVM switch statement is correctly handled.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int x = 1, res = 0;
  __yktrace_start_tracing(HW_TRACING, &x, &res);
  NOOPT_VAL(x);
  switch (x) {
  case 1:
    res = 5;
    break;
  case 2:
    res = 12;
    break;
  case 3:
    res = 4;
    break;
  default:
    res += 1;
  }
  NOOPT_VAL(res);
  void *tr = __yktrace_stop_tracing();
  assert(res == 5);

  x = 1;
  res = 0;
  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  func(&x, &res);
  printf("%d\n", res);
  assert(res == 5);

  return (EXIT_SUCCESS);
}
