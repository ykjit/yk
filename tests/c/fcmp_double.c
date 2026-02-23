// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     1.100000 == 2.200000: 0
//     1.100000 == 1.100000: 1
//     1.100000 != 2.200000: 1
//     1.100000 > 2.200000: 0
//     1.100000 < 2.200000: 1
//     1.100000 >= 2.200000: 0
//     1.100000 <= 2.200000: 1
//     1.100000 >= 1.100000: 1
//     1.100000 <= 1.100000: 1
//     1.100000 <= nan: 0
//     nan == nan: 0
//     nan != nan: 1
//     0.000000 == -0.000000: 1
//     yk-tracing: stop-tracing
//     ...
//     yk-execution: enter-jit-code
//     1.100000 == 2.200000: 0
//     1.100000 == 1.100000: 1
//     1.100000 != 2.200000: 1
//     1.100000 > 2.200000: 0
//     1.100000 < 2.200000: 1
//     1.100000 >= 2.200000: 0
//     1.100000 <= 2.200000: 1
//     1.100000 >= 1.100000: 1
//     1.100000 <= 1.100000: 1
//     1.100000 <= nan: 0
//     nan == nan: 0
//     nan != nan: 1
//     0.000000 == -0.000000: 1
//     yk-execution: deoptimise ...

// Check that double comparisons work.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3;
  double a = 1.1, b = 2.2, nana = NAN;
  double aa = a, nanb = nana;
  double zero = 0.0, negzero = -0.0;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(a);
  NOOPT_VAL(aa);
  NOOPT_VAL(b);
  NOOPT_VAL(nana);
  NOOPT_VAL(nanb);
  NOOPT_VAL(zero);
  NOOPT_VAL(negzero);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%f == %f: %d\n", a, b, a == b);
    fprintf(stderr, "%f == %f: %d\n", a, aa, a == aa);
    fprintf(stderr, "%f != %f: %d\n", a, b, a != b);
    fprintf(stderr, "%f > %f: %d\n", a, b, a > b);
    fprintf(stderr, "%f < %f: %d\n", a, b, a < b);
    fprintf(stderr, "%f >= %f: %d\n", a, b, a > b);
    fprintf(stderr, "%f <= %f: %d\n", a, b, a < b);
    fprintf(stderr, "%f >= %f: %d\n", a, aa, a >= aa);
    fprintf(stderr, "%f <= %f: %d\n", a, aa, a <= aa);
    fprintf(stderr, "%f <= %f: %d\n", a, nana, a <= nana);
    fprintf(stderr, "%f == %f: %d\n", nana, nanb, nana == nanb);
    fprintf(stderr, "%f != %f: %d\n", nana, nanb, nana != nanb);
    fprintf(stderr, "%f == %f: %d\n", zero, negzero, zero == negzero);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
