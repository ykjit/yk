// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5: uadd=1,1
//     5: sadd=1,-2147483647
//     5: usub=1,4294967295
//     5: ssub=1,2147483647
//     5: umul=1,4294967293
//     5: smul=1,2147483645
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{uadd}}: {0: i32, 32: i1} = call llvm.uadd.with.overflow.i32(%{{_}}, 2i32)
//     br bb{{uadd_bb}}
//     bb{{uadd_bb}}:
//     %{{uadd_ov}}: i1 = extractvalue %{{uadd}}, [1]
//     %{{uadd_r}}: i32 = extractvalue %{{uadd}}, [0]
//     ...
//     %{{sadd}}: {0: i32, 32: i1} = call llvm.sadd.with.overflow.i32(%{{_}}, 2i32)
//     br bb{{sadd_bb}}
//     bb{{sadd_bb}}:
//     %{{sadd_ov}}: i1 = extractvalue %{{sadd}}, [1]
//     %{{sadd_r}}: i32 = extractvalue %{{sadd}}, [0]
//     ...
//     %{{usub}}: {0: i32, 32: i1} = call llvm.usub.with.overflow.i32(%{{_}}, 1i32)
//     br bb{{usub_bb}}
//     bb{{usub_bb}}:
//     %{{usub_ov}}: i1 = extractvalue %{{usub}}, [1]
//     %{{usub_r}}: i32 = extractvalue %{{usub}}, [0]
//     ...
//     %{{ssub}}: {0: i32, 32: i1} = call llvm.ssub.with.overflow.i32(%{{_}}, %{{_}})
//     br bb{{ssub_bb}}
//     bb{{ssub_bb}}:
//     %{{ssub_ov}}: i1 = extractvalue %{{ssub}}, [1]
//     %{{ssub_r}}: i32 = extractvalue %{{ssub}}, [0]
//     ...
//     %{{umul}}: {0: i32, 32: i1} = call llvm.umul.with.overflow.i32(%{{_}}, 3i32)
//     br bb{{umul_bb}}
//     bb{{umul_bb}}:
//     %{{umul_ov}}: i1 = extractvalue %{{umul}}, [1]
//     %{{umul_r}}: i32 = extractvalue %{{umul}}, [0]
//     ...
//     %{{smul}}: {0: i32, 32: i1} = call llvm.smul.with.overflow.i32(%{{_}}, 3i32)
//     br bb{{smul_bb}}
//     bb{{smul_bb}}:
//     %{{smul_ov}}: i1 = extractvalue %{{smul}}, [1]
//     %{{smul_r}}: i32 = extractvalue %{{smul}}, [0]
//     ...
//     %{{uadd_ovz}}: i32 = zext %{{uadd_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{uadd_ovz}}, %{{uadd_r}})
//     ...
//     %{{sadd_ovz}}: i32 = zext %{{sadd_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{sadd_ovz}}, %{{sadd_r}})
//     ...
//     %{{usub_ovz}}: i32 = zext %{{usub_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{usub_ovz}}, %{{usub_r}})
//     ...
//     %{{ssub_ovz}}: i32 = zext %{{ssub_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{ssub_ovz}}, %{{ssub_r}})
//     ...
//     %{{umul_ovz}}: i32 = zext %{{umul_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{umul_ovz}}, %{{umul_r}})
//     ...
//     %{{smul_ovz}}: i32 = zext %{{smul_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{smul_ovz}}, %{{smul_r}})
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{h_uadd_ov}}: i1 = uadd_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_sadd_ov}}: i1 = sadd_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_usub_ov}}: i1 = usub_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_ssub_ov}}: i1 = ssub_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_umul_ov}}: i1 = umul_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_smul_ov}}: i1 = smul_overflow %{{_}}, %{{_}}
//     ...
//     %{{h_uadd_ovz}}: i32 = zext %{{h_uadd_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_uadd_ovz}}, %{{_}}) ; @fprintf
//     ...
//     %{{h_sadd_ovz}}: i32 = zext %{{h_sadd_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_sadd_ovz}}, %{{_}}) ; @fprintf
//     ...
//     %{{h_usub_ovz}}: i32 = zext %{{h_usub_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_usub_ovz}}, %{{_}}) ; @fprintf
//     ...
//     %{{h_ssub_ovz}}: i32 = zext %{{h_ssub_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_ssub_ovz}}, %{{_}}) ; @fprintf
//     ...
//     %{{h_umul_ovz}}: i32 = zext %{{h_umul_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_umul_ovz}}, %{{_}}) ; @fprintf
//     ...
//     %{{h_smul_ovz}}: i32 = zext %{{h_smul_ov}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{h_smul_ovz}}, %{{_}}) ; @fprintf
//     ...
//     --- End hir ---
//     4: uadd=0,4294967295
//     4: sadd=0,2147483647
//     4: usub=0,1
//     4: ssub=0,-2147483647
//     4: umul=0,6
//     4: smul=0,6
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: uadd=1,1
//     3: sadd=1,-2147483647
//     3: usub=1,4294967295
//     3: ssub=1,2147483647
//     3: umul=1,4294967293
//     3: smul=1,2147483645
//     2: uadd=0,4294967295
//     2: sadd=0,2147483647
//     2: usub=0,1
//     2: ssub=0,-2147483647
//     2: umul=0,6
//     2: smul=0,6
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that llvm.{u,s}{add,sub,mul}.with.overflow are supported by the yk compiler.

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
  while (i > 1) {
    yk_mt_control_point(mt, &loc);

    // Overflow test only on odd `i`.
    int odd = i % 2;
    unsigned int ua = odd ? UINT_MAX : UINT_MAX - 2;
    unsigned int ur;
    NOOPT_VAL(ua);
    bool uadd_ov = __builtin_uadd_overflow(ua, 2, &ur);

    int sa = odd ? INT_MAX : INT_MAX - 2;
    int sr;
    NOOPT_VAL(sa);
    bool sadd_ov = __builtin_sadd_overflow(sa, 2, &sr);

    unsigned int usa = odd ? 0 : 2;
    unsigned int usr;
    NOOPT_VAL(usa);
    bool usub_ov = __builtin_usub_overflow(usa, 1, &usr);

    int ssa = odd ? INT_MIN : INT_MIN + 2;
    int ssb = 1;
    int ssr;
    NOOPT_VAL(ssa);
    NOOPT_VAL(ssb);
    bool ssub_ov = __builtin_ssub_overflow(ssa, ssb, &ssr);

    unsigned int uma = odd ? UINT_MAX : 2;
    unsigned int umr;
    NOOPT_VAL(uma);
    bool umul_ov = __builtin_umul_overflow(uma, 3, &umr);

    int sma = odd ? INT_MAX : 2;
    int smr;
    NOOPT_VAL(sma);
    bool smul_ov = __builtin_smul_overflow(sma, 3, &smr);

    fprintf(stderr, "%d: uadd=%d,%u\n", i, uadd_ov, ur);
    fprintf(stderr, "%d: sadd=%d,%d\n", i, sadd_ov, sr);
    fprintf(stderr, "%d: usub=%d,%u\n", i, usub_ov, usr);
    fprintf(stderr, "%d: ssub=%d,%d\n", i, ssub_ov, ssr);
    fprintf(stderr, "%d: umul=%d,%u\n", i, umul_ov, umr);
    fprintf(stderr, "%d: smul=%d,%d\n", i, smul_ov, smr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
