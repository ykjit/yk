// Compiler:
// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int num) {
  if (num == 0)
    return 1;
  if (num == 1)
    return 2;
  if (num == 2)
    return 4;
  return num;
}

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  res = foo(2);
  void *tr = __yktrace_stop_tracing(tt);
  assert(res == 4);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  assert(output == 4);

  return (EXIT_SUCCESS);
}
