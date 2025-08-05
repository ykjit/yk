// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     yk-warning: trace-compilation-aborted: encountered call to longjmp
//     ...
//     exit

// Check that we reject longjmps out of a trace.
//
// This is an annoying case to catch. On the face of it you might think you can
// use control flow integrity checking to check that after any outlined call
// to foreign code, you either see a function entry block (i.e. a call-back
// to non-foreign code), or the successor to the call.
//
// In this case however, the longjump is indistinguishable from a callback
// because we longjmp to a foreign parent frame and then call the non-foreign
// function loop(), thus the next block we see is the entry block of loop().

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define ITERS 5

jmp_buf buf;

__attribute__((noinline))
void loop(YkMT *mt, YkLocation *loc, int i) {
  fprintf(stderr, "enter %s(): i=%d\n", __func__, i);
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    fprintf(stderr, "in interp loop: i=%d\n", i);
    yk_mt_control_point(mt, loc);
    // The trace will attempt to jump to the setjmp in the foreign parent frame.
    if (i > 1) {
      longjmp(buf, i);
    }
    i--;
  }
}

// This parent foreign function will set the jump point.
__attribute__((noinline, yk_outline))
void inner_main() {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int r = setjmp(buf);
  fprintf(stderr, "setjmp returned %d\n", r);
  if (r == 0) {
    loop(mt, &loc, ITERS);
  } else {
    loop(mt, &loc, r - 1);
  }

  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
}


int main(int argc, char **argv) {
  inner_main();
  return (EXIT_SUCCESS);
}

