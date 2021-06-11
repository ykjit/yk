// Compiler:
// Run-time:

// Check that basic trace compilation works.
// FIXME An optimising compiler can remove all of the code between start/stop
// tracing.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(a) { return a; }

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  int a = f(1);
  int b = f(2);
  res = a + b;
  void *tr = __yktrace_stop_tracing(tt);
  assert(res == 3);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  assert(output == 3);

  return (EXIT_SUCCESS);
}
