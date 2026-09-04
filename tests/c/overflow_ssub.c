// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5: ssub=1,2147483646
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{ssub}}: {0: i32, 32: i1} = call llvm.ssub.with.overflow.i32(%{{_}}, %{{_}})
//     br bb{{ssub_bb}}
//     bb{{ssub_bb}}:
//     %{{ssub_ov}}: i1 = extractvalue %{{ssub}}, [1]
//     %{{ssub_r}}: i32 = extractvalue %{{ssub}}, [0]
//     ...
//     %{{ssub_ovz}}: i32 = zext %{{ssub_ov}}, i32
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{ssub_ovz}}, %{{ssub_r}})
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{h_packed}}: i64 = ssub_overflow %{{h_lhs}}, %{{h_rhs}}
//     %{{h_ssub_r}}: i32 = extractval %{{h_packed}} [0]
//     %{{h_ssub_ov}}: i1 = extractval %{{h_packed}} [32]
//     %{{h_stderr}}: ptr = 0x{{_}} ; @stderr
//     %{{h_stream}}: ptr = load %{{h_stderr}}
//     %{{h_i}}: i32 = load %1
//     %{{h_ssub_ovz}}: i32 = zext %{{h_ssub_ov}}
//     %{{h_fmt}}: ptr = 0x{{_}} ; @.str
//     %{{h_fprintf}}: ptr = 0x{{_}} ; @fprintf
//     %{{_}}: i32 = call %{{h_fprintf}}(%{{h_stream}}, %{{h_fmt}}, %{{h_i}}, %{{h_ssub_ovz}}, %{{h_ssub_r}}) ; @fprintf
//     ...
//     --- End hir ---
//     4: ssub=0,-2147483648
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: ssub=1,2147483646
//     2: ssub=0,-2147483648
//     1: ssub=1,2147483646
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that llvm.ssub.with.overflow is supported by the yk compiler.

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
    int sa = i % 2 ? INT_MIN : INT_MIN + 2;
    int sb = 2;
    int sr;
    NOOPT_VAL(sa);
    NOOPT_VAL(sb);
    bool ssub_ov = __builtin_ssub_overflow(sa, sb, &sr);

    fprintf(stderr, "%d: ssub=%d,%d\n", i, ssub_ov, sr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
