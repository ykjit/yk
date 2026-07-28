// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-execution: enter-jit-code {"trid": "0"}
//     ...

// Build a 2-field struct from an uninitialized local, write its fields elsewhere,
// never read the struct back. This mirrors the mrb_value construction that segfaults 
//in mruby.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct Pair {
  uint64_t a;
  uint32_t b;
};

uint64_t out_a[8];
uint32_t out_b[8];

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 0;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i < 8) {
    yk_mt_control_point(mt, &loc);
    struct Pair p;
    p.a = (uint64_t)i * 10;
    p.b = (uint32_t)i + 1;
    struct Pair q;
    q.a = (uint64_t)i * 20;
    q.b = (uint32_t)i + 2;
    out_a[i] = p.a + q.a;
    out_b[i] = p.b + q.b;
    fprintf(stderr, "%lu %u\n", out_a[i], out_b[i]);
    i++;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
