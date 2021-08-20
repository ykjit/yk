// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=aot
//   stderr:
//     ...
//     indirectbr i8* %...
//     ...

// Check that we can handle indirect branches.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  // Note that LLVM knows that `l1` is dead code because `argc` is always >0.
  void *dispatch[] = {&&l1, &&l2, &&l3};
  int z = 0;

  void *tt = __yktrace_start_tracing(HW_TRACING, &z, &argc);
  NOOPT_VAL(argc);
  goto *dispatch[argc];
l1:
  z = 7;
  goto done;
l2:
  z = 20;
  goto done;
l3:
  z = 33;
done:
  NOOPT_VAL(z);
  void *tr = __yktrace_stop_tracing(tt);
  assert(z == 20);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  int z2 = 0;
  func(&z2, &argc);
  assert(z2 == 20);

  return (EXIT_SUCCESS);
}
