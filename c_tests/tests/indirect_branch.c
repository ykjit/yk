// ignore: broken during new control point design
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
  int z = 0, idx = argc;

  __yktrace_start_tracing(HW_TRACING, &z, &idx);
  NOOPT_VAL(idx);

  // Now jump to l2 and then l3 via computed gotos.
  goto *dispatch[idx];
l1:
  // unreachable.
  exit(EXIT_FAILURE);
l2:
  z += 1;
  idx += 1;
  goto *dispatch[idx];
l3:
  z += 2;

  NOOPT_VAL(z);
  void *tr = __yktrace_stop_tracing();
  printf("ZZZ %d\n", z);
  assert(z == 3);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  int z2 = 0;
  int idx2 = argc;
  func(&z2, &idx2);
  assert(z2 == 3);

  return (EXIT_SUCCESS);
}
