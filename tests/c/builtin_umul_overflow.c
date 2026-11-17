// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5: full=1,4294967294 result=4294967294 flag=1
//     5: wide=1,18446744073709551614
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{umul}}: i65 = umul_overflow %{{_}}, %{{_}}
//     ...
//     --- End hir ---
//     4: full=0,6 result=6 flag=0
//     4: wide=0,8589934590
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: full=1,4294967294 result=4294967294 flag=1
//     3: wide=1,18446744073709551614
//     2: full=0,6 result=6 flag=0
//     2: wide=0,8589934590
//     1: full=1,4294967294 result=4294967294 flag=1
//     1: wide=1,18446744073709551614
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 5;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);

    unsigned int a = i % 2 ? UINT_MAX : 3;
    NOOPT_VAL(a);
    unsigned int full_result, result_only, unused_result;
    bool full_overflow = __builtin_umul_overflow(a, 2, &full_result);
    __builtin_umul_overflow(a, 2, &result_only);
    bool flag_only = __builtin_umul_overflow(a, 2, &unused_result);
    fprintf(stderr, "%d: full=%d,%u result=%u flag=%d\n", i, full_overflow,
            full_result, result_only, flag_only);

    unsigned long long wide_a = i % 2 ? ULLONG_MAX : UINT_MAX;
    NOOPT_VAL(wide_a);
    unsigned long long wide_result;
    bool wide_overflow = __builtin_umulll_overflow(wide_a, 2, &wide_result);
    fprintf(stderr, "%d: wide=%d,%llu\n", i, wide_overflow, wide_result);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return EXIT_SUCCESS;
}
