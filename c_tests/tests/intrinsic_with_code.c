// Compiler:
// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  long z = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &z, &argc);
  NOOPT_VAL(argc);
  __builtin_add_overflow((long)argc, (long)argc, &z);
  void *tr = __yktrace_stop_tracing(tt);
  assert(z == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(long *, int *) = (void (*)(long *, int *))ptr;
  long z2 = 0;
  func(&z2, &argc);
  assert(z2 == 2);

  return (EXIT_SUCCESS);
}
