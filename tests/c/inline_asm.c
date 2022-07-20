// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     ...call i32 asm "mov $$5, $0"...
//     ...
//     --- End jit-pre-opt ---
//     ...
//     jit-state: enter-jit-code
//     res=5
//     ...

// Check that we can handle inline asm properly.

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {

  YkMT *mt = yk_mt_new();
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
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}