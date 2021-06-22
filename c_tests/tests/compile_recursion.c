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

__attribute__((noinline)) int fib(int num) {
  if (num == 0)
    return 0;
  if (num == 1)
    return 1;
  if (num == 2)
    return 1;
  int a = fib(num - 2);
  int b = fib(num - 2);
  int c = a + b;
  return c;
}

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  res = fib(3);
  void *tr = __yktrace_stop_tracing(tt);
  printf("Result: %i\n", res);
  assert(res == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  fprintf(stderr, "Result: %i\n", output);
  assert(output == 2);

  return (EXIT_SUCCESS);
}
