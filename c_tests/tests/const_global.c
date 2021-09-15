// Compiler:
// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

const volatile int global_int = 6;

__attribute__((noinline)) int foo() {
  int x = global_int;
  int y = global_int;
  int z = x + y;
  return z;
}

int main(int argc, char **argv) {
  int res = 0;
  __yktrace_start_tracing(HW_TRACING, &res);
  res = foo();
  void *tr = __yktrace_stop_tracing();
  assert(res == 12);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  assert(output == 12);

  return (EXIT_SUCCESS);
}
