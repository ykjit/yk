// ignore: broken during new control point design
// Compiler:
// Run-time:

// Check that we can call a static function with internal linkage from the same
// compilation unit.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

static int call_me(int x) {
  if (x == 5)
    return x;
  else {
    // The recursion will cause a call to be emitted in the trace.
    return call_me(x + 1);
  }
}

int main(int argc, char **argv) {
  int res = 0;
  __yktrace_start_tracing(HW_TRACING, &res, &argc);
  NOOPT_VAL(argc);
  // At higher optimisation levels LLVM realises that this call can be
  // completely removed. Hence we only structurally test a couple of lower opt
  // levels.
  res = call_me(argc);
  NOOPT_VAL(res);
  void *tr = __yktrace_stop_tracing();
  assert(res == 5);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, int *) = (void (*)(int *, int *))ptr;
  int res2 = 0;
  func(&res2, &argc);
  assert(res2 == 5);

  return (EXIT_SUCCESS);
}
