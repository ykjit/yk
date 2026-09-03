// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5: sadd=1,-2147483647
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{sadd}}: {0: i32, 32: i1} = call llvm.sadd.with.overflow.i32(%{{_}}, 2i32)
//     br bb{{sadd_bb}}
//     bb{{sadd_bb}}:
//     %{{sadd_ov}}: i1 = extractvalue %{{sadd}}, [1]
//     %{{sadd_r}}: i32 = extractvalue %{{sadd}}, [0]
//     ...
//     %{{sadd_ovz}}: i32 = zext %{{sadd_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{sadd_ovz}}, %{{sadd_r}})
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{h_packed}}: i64 = sadd_overflow %{{h_lhs}}, %{{h_rhs}}
//     %{{h_sadd_r}}: i32 = extractval %{{h_packed}} [0]
//     %{{h_sadd_ov}}: i1 = extractval %{{h_packed}} [32]
//     %{{h_stderr}}: ptr = 0x{{_}} ; @stderr
//     %{{h_stream}}: ptr = load %{{h_stderr}}
//     %{{h_i}}: i32 = load %1
//     %{{h_sadd_ovz}}: i32 = zext %{{h_sadd_ov}}
//     %{{h_fmt}}: ptr = 0x{{_}} ; @.str
//     %{{h_fprintf}}: ptr = 0x{{_}} ; @fprintf
//     %{{_}}: i32 = call %{{h_fprintf}}(%{{h_stream}}, %{{h_fmt}}, %{{h_i}}, %{{h_sadd_ovz}}, %{{h_sadd_r}}) ; @fprintf
//     ...
//     --- End hir ---
//     4: sadd=0,2147483647
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: sadd=1,-2147483647
//     2: sadd=0,2147483647
//     1: sadd=1,-2147483647
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that llvm.sadd.with.overflow is supported by the yk compiler.

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
    int sr;
    NOOPT_VAL(sa);
    bool sadd_ov = __builtin_sadd_overflow(sa, 2, &sr);

    fprintf(stderr, "%d: sadd=%d,%d\n", i, sadd_ov, sr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
