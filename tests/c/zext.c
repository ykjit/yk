// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     int to long 4
//     long to long long 4
//     uint8_t to uint32_t 3
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     int to long 3
//     long to long long 3
//     uint8_t to uint32_t 3
//     yk-jit-event: enter-jit-code
//     int to long 2
//     long to long long 2
//     uint8_t to uint32_t 3
//     int to long 1
//     long to long long 1
//     uint8_t to uint32_t 3
//     yk-jit-event: deoptimise
//     exit

// Test zero extend.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  unsigned int i = 4;
  uint8_t j = 3;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    long x = i;
    long long y = x;
    uint32_t z = j;
    fprintf(stderr, "int to long %ld\n", x);
    fprintf(stderr, "long to long long %lld\n", y);
    fprintf(stderr, "uint8_t to uint32_t %d\n", z);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
