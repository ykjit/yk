// ignore: broken during new control point design
// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=aot,jit-pre-opt,jit-post-opt
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End jit-post-opt ---
//     ...

// Check that our NOOPT_VAL macro blocks optimisations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int x, res;
  __yktrace_start_tracing(HW_TRACING, &res, &x);
  // __yktrace_start_tracing() will already inhibit optimisations on `x` since
  // its address is passed as an argument and the compiler must assume that it
  // may be mutated. Therefore to properly test NOOPT_VAL we need to initialise
  // `x` after the call to __yktrace_start_tracing..
  x = 2;
  NOOPT_VAL(x);
  res = x + 3; // We don't want this operation to be optimised away.
  NOOPT_VAL(res);
  void *tr = __yktrace_stop_tracing();
  assert(res == 5);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  int res2 = 0;
  func(&res2, &x);
  assert(res2 == 5);

  return (EXIT_SUCCESS);
}
