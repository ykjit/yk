// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   status: error
//   stderr:
//     ...
//     InlineAsm is currently not supported.

// Check that we bail when we see non-empty inline asm.

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int foo() {
  fprintf(stderr, "foo\n");
  return 11;
}

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
    asm("call foo"
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
