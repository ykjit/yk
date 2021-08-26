// Compiler:
// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int foo(int a) {
  char pwd[64];
  if (a > 1) {
    memset(pwd, 2, sizeof pwd);
  } else {
    memset(pwd, 1, sizeof pwd);
  }
  return pwd[0];
}

int main(int argc, char **argv) {
  long z = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &z, &argc);
  NOOPT_VAL(argc);
  z = foo(argc);
  void *tr = __yktrace_stop_tracing(tt);
  assert(z == 1);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(long *, int *) = (void (*)(long *, int *))ptr;
  long z2 = 0;
  func(&z2, &argc);
  assert(z2 == 1);

  return (EXIT_SUCCESS);
}
