// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     yk-warning: trace-compilation-aborted: Unimplemented: ...
//     ...
//     yk-warning: trace-compilation-aborted: Unimplemented: ...
//     ...

// Check that we can handle inline asm properly (currently this aborts the
// trace until we can deal with calls inside inline asm).

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {

  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 0;
  int i = 4;
  NOOPT_VAL(i);
  NOOPT_VAL(res);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
#ifdef __x86_64__
    // Stores the constant 5 into `res`.
    asm("mov $5, %0"
        : "=r"(res) // outputs.
        :           // inputs.
        :           // clobbers.
    );
    fprintf(stderr, "res=%d\n", res);
#else
#error unknown platform
#endif
    i--;
  }

  assert(res == 5);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
