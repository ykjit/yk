// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i: 4 == 4.000000
//     u64_123456: 123456 == {{double_u64_123456}}
//     u64_zero: 0 == {{double_zero}}
//     u64_top_bit: 9223372036854775808 == {{double_u64_top_bit}}
//     u64_top_bit_minus_one: 9223372036854775807 == {{double_u64_top_bit}}
//     u64_max: 18446744073709551615 == {{double_u64_max}}
//     yk-tracing: stop-tracing
//     i: 3 == 3.000000
//     u64_123456: 123456 == {{double_u64_123456}}
//     u64_zero: 0 == {{double_zero}}
//     u64_top_bit: 9223372036854775808 == {{double_u64_top_bit}}
//     u64_top_bit_minus_one: 9223372036854775807 == {{double_u64_top_bit}}
//     u64_max: 18446744073709551615 == {{double_u64_max}}
//     yk-execution: enter-jit-code
//     i: 2 == 2.000000
//     u64_123456: 123456 == {{double_u64_123456}}
//     u64_zero: 0 == {{double_zero}}
//     u64_top_bit: 9223372036854775808 == {{double_u64_top_bit}}
//     u64_top_bit_minus_one: 9223372036854775807 == {{double_u64_top_bit}}
//     u64_max: 18446744073709551615 == {{double_u64_max}}
//     i: 1 == 1.000000
//     u64_123456: 123456 == {{double_u64_123456}}
//     u64_zero: 0 == {{double_zero}}
//     u64_top_bit: 9223372036854775808 == {{double_u64_top_bit}}
//     u64_top_bit_minus_one: 9223372036854775807 == {{double_u64_top_bit}}
//     u64_max: 18446744073709551615 == {{double_u64_max}}
//     yk-execution: deoptimise
//     exit

// Check uitofp instructions have correct runtime semantics.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  uint64_t i = 4;
  uint64_t u64_zero = 0;
  uint64_t u64_123456 = 123456;
  uint64_t u64_max = UINT64_MAX;
  // The first numeric value with the top bit set.
  uint64_t u64_top_bit = UINT64_C(1) << 63;
  // note: the following value gets rounded to `u64_top_bit` when casted.
  uint64_t u64_top_bit_minus_one = u64_top_bit - 1;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(u64_123456);
  NOOPT_VAL(u64_zero);
  NOOPT_VAL(u64_top_bit);
  NOOPT_VAL(u64_top_bit_minus_one);
  NOOPT_VAL(u64_max);

#define print_casted(v) fprintf(stderr, #v ": %" PRIu64 " == %f\n", v, (double) v);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    print_casted(i);
    print_casted(u64_123456);
    print_casted(u64_zero);
    print_casted(u64_top_bit);
    print_casted(u64_top_bit_minus_one);
    print_casted(u64_max);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
