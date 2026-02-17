// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0 -Xclang -disable-O0-optnone -Xlinker --lto-newpm-passes=instcombine<max-iterations=1;no-use-loop-info;no-verify-fixpoint>
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stdout:
//     0: 4.200000 8.400000
//     1: 5.200000 9.400000
//     2: 6.200000 10.400000
//   stderr:
//     ...
//     %{{_}}: float = call %{{_}}() ; @float_rtn
//     ...
//     %{{y}}: double = call %{{_}}() ; @double_rtn
//     ...
//     %{{z}}: double = fadd %{{y}}, %{{_}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{z}}) ; @printf
//     ...

#include <stdio.h>
#include <yk.h>

__attribute__((yk_outline))
float float_rtn() {
  return 4.2;
}

__attribute__((yk_outline))
double double_rtn() {
  return float_rtn() * 2;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  float i = 0;
  YkLocation loc = yk_location_new();
  while (i < 3) {
    yk_mt_control_point(mt, &loc);
    printf("%d: %f %f\n", (int) i, float_rtn() + i, double_rtn() + i);
    i += 1.0;
  }
  return 0;
}
