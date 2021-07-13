// ignore: Mutable global variables not supported.
// Compiler:
// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int global_int = 12;

__attribute__((noinline)) int foo(int num) {
  global_int = num;
  return global_int;
}

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  res = foo(2);
  void *tr = __yktrace_stop_tracing(tt);
  assert(res == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(void *) = (void (*)(void *))ptr;
  int output = 0;
  func(&output);
  assert(output == 4);

  return (EXIT_SUCCESS);
}
