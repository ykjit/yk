// Run-time:
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func add1(%arg0: i32) -> i32;
//     ...
//     func add2(%arg0: i32) -> i32;
//     ...
//     %{{8_4}}: func(i32) -> i32 = select %{{8_2}}, add1, add2
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{10}}: i1 = icmp eq ...
//     %{{12}}: ptr = 0x{{_}} ; @add1
//     %{{13}}: ptr = 0x{{_}} ; @add2
//     %{{14}}: ptr = select %{{10}}, %{{12}}, %{{13}}
//     ...
//     --- End hir ---
//     5
//     yk-execution: enter-jit-code {"trid": "0"}
//     3
//     3
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     exit

// Check that `select` between two function pointers works.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_outline)) int add1(int x) { return x + 1; }

__attribute__((yk_outline)) int add2(int x) { return x + 2; }

void interp() {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int (*fp)(int) = (i % 2 == 0) ? add1 : add2;
    int r = fp(i);
    fprintf(stderr, "%d\n", r);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
}

int main(void) {
  interp();
  return (EXIT_SUCCESS);
}
