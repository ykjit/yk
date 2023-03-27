// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_PRINT_JITSTATE=1

// Check that we can reliably deal with "foreign" (not compiled with ykllvm)
// code that calls back into "native code".

// FIXME:  When the trace compiler encounters a call to foreign code, it
// simply emits a `call` into the JIT trace (not inlining it -- how could it?
// We have no IR for the foreign code). However, in order to continue
// building the remainder of the JIT trace after the call, the trace compiler
// must skip over all the parts of the of "input trace" (e.g. the hardware
// trace) that correspond with the outlined call. Currently the compiler
// simply waits until it finds a block with IR again. This is clearly
// incorrect if foreign code calls back to code we have IR for.
//
// The good news is that there is an assertion checking that the place the
// trace compiler tries to pick up compilation again is as expected. This
// ensures that we don't miscompile at least. This test fails that assertion:
//
// Assertion `std::get<1>(ResumeAfter.getValue())->getParent() == BB' failed.
//
// (When fixing this test, add lines to match in the output)

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int call_callback(int (*callback)(int, int), int x, int y);

__attribute((noinline)) int callback(int x, int y) { return (x + y) / 2; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new();
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int x = 0;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(x);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, x=%d\n", i, x);
    call_callback(&callback, i, i);
    i--;
  }
  NOOPT_VAL(x);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
