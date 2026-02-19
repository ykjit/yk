// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0 -Xclang -disable-O0-optnone -Xlinker --lto-newpm-passes=instcombine<max-iterations=1;no-verify-fixpoint>
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=0, elem=000
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{10_13}}: ptr = ptr_add %{{0_5}}, 0 + (%{{10_8}} * 64) + (%{{10_10}} * 16) + (%{{%10_12}} * 4)
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{16}}: ptr = dynptradd %{{3}}, %{{_}}, 64
//     %{{17}}: ptr = dynptradd %{{16}}, %{{_}}, {{_}}
//     %{{_}}: ptr = dynptradd %{{17}}, %{{_}}, {{_}}
//     ...
//     --- End jit-pre-opt ---
//     i=1, elem=111
//     yk-execution: enter-jit-code
//     i=2, elem=222
//     i=3, elem=333
//     yk-execution: deoptimise ...

// Check dynamic ptradd instructions work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  // Load up a big multi-dimentsional array that we will dynamically index.
  uint32_t array[4][4][4];
  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 4; y++) {
      for (int z = 0; z < 4; z++) {
        int val = x * 100 + y * 10 + z;
        array[x][y][z] = val;
      }
    }
  }

  int i = 0;
  NOOPT_VAL(loc);
  NOOPT_VAL(array);
  NOOPT_VAL(i);
  while (i < 4) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, elem=%03d\n", i, array[i][i][i]);
    i++;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
