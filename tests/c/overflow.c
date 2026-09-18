// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5: sadd full=1,-2147483647
//     5: sadd result=-2147483647 overflow=1
//     5: uadd full=1,1
//     5: uadd result=1 overflow=1
//     5: ssub full=1,2147483646
//     5: ssub result=2147483646 overflow=1
//     5: usub full=1,4294967295
//     5: usub result=4294967295 overflow=1
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{sadd}}: {0: i32, 32: i1} = call llvm.sadd.with.overflow.i32(%{{_}}, 2i32)
//     ...
//     %{{sadd_ov}}: i1 = extractvalue %{{sadd}}, [1]
//     %{{sadd_r}}: i32 = extractvalue %{{sadd}}, [0]
//     ...
//     %{{uadd}}: {0: i32, 32: i1} = call llvm.uadd.with.overflow.i32(%{{_}}, 2i32)
//     ...
//     %{{uadd_ov}}: i1 = extractvalue %{{uadd}}, [1]
//     %{{uadd_r}}: i32 = extractvalue %{{uadd}}, [0]
//     ...
//     %{{ssub}}: {0: i32, 32: i1} = call llvm.ssub.with.overflow.i32(%{{_}}, %{{_}})
//     ...
//     %{{ssub_ov}}: i1 = extractvalue %{{ssub}}, [1]
//     %{{ssub_r}}: i32 = extractvalue %{{ssub}}, [0]
//     ...
//     %{{usub}}: {0: i32, 32: i1} = call llvm.usub.with.overflow.i32(%{{_}}, 2i32)
//     ...
//     %{{usub_ov}}: i1 = extractvalue %{{usub}}, [1]
//     %{{usub_r}}: i32 = extractvalue %{{usub}}, [0]
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{h_sadd}}: i64 = sadd_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_uadd}}: i64 = uadd_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_ssub}}: i64 = ssub_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_usub}}: i64 = usub_overflow %{{_}}, %{{_}}
//     ...
//     --- End hir ---
//     4: sadd full=0,2147483647
//     4: sadd result=2147483647 overflow=0
//     4: uadd full=0,4294967295
//     4: uadd result=4294967295 overflow=0
//     4: ssub full=0,-2147483648
//     4: ssub result=-2147483648 overflow=0
//     4: usub full=0,3
//     4: usub result=3 overflow=0
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: sadd full=1,-2147483647
//     3: sadd result=-2147483647 overflow=1
//     3: uadd full=1,1
//     3: uadd result=1 overflow=1
//     3: ssub full=1,2147483646
//     3: ssub result=2147483646 overflow=1
//     3: usub full=1,4294967295
//     3: usub result=4294967295 overflow=1
//     2: sadd full=0,2147483647
//     2: sadd result=2147483647 overflow=0
//     2: uadd full=0,4294967295
//     2: uadd result=4294967295 overflow=0
//     2: ssub full=0,-2147483648
//     2: ssub result=-2147483648 overflow=0
//     2: usub full=0,3
//     2: usub result=3 overflow=0
//     1: sadd full=1,-2147483647
//     1: sadd result=-2147483647 overflow=1
//     1: uadd full=1,1
//     1: uadd result=1 overflow=1
//     1: ssub full=1,2147483646
//     1: ssub result=2147483646 overflow=1
//     1: usub full=1,4294967295
//     1: usub result=4294967295 overflow=1
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that llvm.{sadd,uadd,ssub,usub}.with.overflow are supported by the yk when:
// 1. Only the result is used
// 2. Only the overflow flag is used
// 2. Both the result and the overflow flag are used.
//
// -O0 is required: at higher opt levels LLVM will fold the three calls per op below back into one.

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 5;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);

    int sa = i % 2 ? INT_MAX : INT_MAX - 2;
    NOOPT_VAL(sa);
    int sadd_full_r, sadd_res_r, sadd_ov_r;
    // Three separate calls. Each call is an independent llvm.sadd.with.overflow 
    // call site, so each exercises a different combination of {result, overflow}.
    bool sadd_full_ov = __builtin_sadd_overflow(sa, 2, &sadd_full_r);
    __builtin_sadd_overflow(sa, 2, &sadd_res_r);
    // sadd_ov_r is intentionally never read -- this is the "overflow flag
    // only" case.
    bool sadd_ov_ov = __builtin_sadd_overflow(sa, 2, &sadd_ov_r);
    fprintf(stderr, "%d: sadd full=%d,%d\n", i, sadd_full_ov, sadd_full_r);
    fprintf(stderr, "%d: sadd result=%d overflow=%d\n", i, sadd_res_r,
            sadd_ov_ov);

    unsigned int ua = i % 2 ? UINT_MAX : UINT_MAX - 2;
    NOOPT_VAL(ua);
    unsigned int uadd_full_r, uadd_res_r, uadd_ov_r;
    bool uadd_full_ov = __builtin_uadd_overflow(ua, 2, &uadd_full_r);
    __builtin_uadd_overflow(ua, 2, &uadd_res_r);
    bool uadd_ov_ov = __builtin_uadd_overflow(ua, 2, &uadd_ov_r);
    fprintf(stderr, "%d: uadd full=%d,%u\n", i, uadd_full_ov, uadd_full_r);
    fprintf(stderr, "%d: uadd result=%u overflow=%d\n", i, uadd_res_r,
            uadd_ov_ov);

    int sb = i % 2 ? INT_MIN : INT_MIN + 2;
    int st = 2;
    NOOPT_VAL(sb);
    NOOPT_VAL(st);
    int ssub_full_r, ssub_res_r, ssub_ov_r;
    bool ssub_full_ov = __builtin_ssub_overflow(sb, st, &ssub_full_r);
    __builtin_ssub_overflow(sb, st, &ssub_res_r);
    bool ssub_ov_ov = __builtin_ssub_overflow(sb, st, &ssub_ov_r);
    fprintf(stderr, "%d: ssub full=%d,%d\n", i, ssub_full_ov, ssub_full_r);
    fprintf(stderr, "%d: ssub result=%d overflow=%d\n", i, ssub_res_r,
            ssub_ov_ov);

    unsigned int ub = i % 2 ? 1 : 5;
    NOOPT_VAL(ub);
    unsigned int usub_full_r, usub_res_r, usub_ov_r;
    bool usub_full_ov = __builtin_usub_overflow(ub, 2, &usub_full_r);
    __builtin_usub_overflow(ub, 2, &usub_res_r);
    bool usub_ov_ov = __builtin_usub_overflow(ub, 2, &usub_ov_r);
    fprintf(stderr, "%d: usub full=%d,%u\n", i, usub_full_ov, usub_full_r);
    fprintf(stderr, "%d: usub result=%u overflow=%d\n", i, usub_res_r,
            usub_ov_ov);

    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
