// Compiler:
// Run-time:

// Check that tracing a cascading "if...else if...else" works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int x) {
  if (x == 0)
    return 30;
  else if (x == 1)
    return 47;
  else
    return 52;
}

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res, &argc);
  res = f(argc);
  void *tr = __yktrace_stop_tracing(tt);
  assert(res == 47);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  int res2 = 0;
  func(&res2, &argc);
  assert(res2 == 47);

  return (EXIT_SUCCESS);
}
