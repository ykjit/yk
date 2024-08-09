// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     u64 18446744073709551615
//     u32 4294967295
//     u16 65535
//     u8 255
//     yk-jit-event: stop-tracing
//     u64 18446744073709551615
//     u32 4294967295
//     u16 65535
//     u8 255
//     yk-jit-event: enter-jit-code
//     u64 18446744073709551615
//     u32 4294967295
//     u16 65535
//     u8 255
//     yk-jit-event: deoptimise
//     exit

// Test truncation.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3;
  uint64_t x = UINT64_MAX;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(x);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "u64 %lu\n", x);
    fprintf(stderr, "u32 %u\n", (uint32_t)x);
    fprintf(stderr, "u16 %hu\n", (uint16_t)x);
    fprintf(stderr, "u8 %hhu\n", (uint8_t)x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
