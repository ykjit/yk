// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     func lt(...
//       ...
//       ...f_ult...
//       ...
//     }
//     ...
//     func lte(...
//       ...
//       ...f_ule...
//       ...
//     }
//     ...
//     func gt(...
//       ...
//       ...f_ugt...
//       ...
//     }
//     ...
//     func gte(...
//       ...
//       ...f_uge...
//       ...
//     }
//     ...
//     func eq(...
//       ...
//       ...f_ueq...
//       ...
//     }
//     ...
//     func ne(...
//       ...
//       ...f_une...
//       ...
//     }
//     ...
//     --- End aot ---
//     ...
//     yk-execution: enter-jit-code
//     1.000000 < 1.000000: 0
//     2.000000 < 2.000000: 0
//     1.000000 < 2.000000: 1
//     2.000000 < 1.000000: 0
//     1.000000 <      nan: 1
//          nan < 1.000000: 1
//          nan <      nan: 1
//     ---
//     1.000000 <= 1.000000: 1
//     2.000000 <= 2.000000: 1
//     1.000000 <= 2.000000: 1
//     2.000000 <= 1.000000: 0
//     1.000000 <= nan: 1
//     nan <= 1.000000: 1
//     nan <= nan: 1
//     ---
//     1.000000 > 1.000000: 0
//     2.000000 > 2.000000: 0
//     1.000000 > 2.000000: 0
//     2.000000 > 1.000000: 1
//     1.000000 > nan: 1
//     nan > 1.000000: 1
//     nan > nan: 1
//     ---
//     1.000000 >= 1.000000: 1
//     2.000000 >= 2.000000: 1
//     1.000000 >= 2.000000: 0
//     2.000000 >= 1.000000: 1
//     1.000000 >= nan: 1
//     nan >= 1.000000: 1
//     nan >= nan: 1
//     ---
//     1.000000 == 1.000000: 1
//     2.000000 == 2.000000: 1
//     1.000000 == 2.000000: 0
//     2.000000 == 1.000000: 0
//     1.000000 == nan: 1
//     nan == 1.000000: 1
//     nan == nan: 1
//     ---
//     1.000000 != 1.000000: 0
//     2.000000 != 2.000000: 0
//     1.000000 != 2.000000: 1
//     2.000000 != 1.000000: 1
//     1.000000 != nan: 1
//     nan != 1.000000: 1
//     nan != nan: 1
//     ---
//     yk-execution: deoptimise ...

// Check unordered comparisons are JITted correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <yk.h>
#include <yk_testing.h>

// Note: Since there's no direct way to do unordered comparisons in the C
// language, this test relies heavily on LLVM's optimiser collapsing certain
// compound comparisons into simpler unordered ones. Hence this test only works
// with AOT optimisations enabled.
//
// The following helpers are convoluted ways to get LLVM to emit unordered
// comparisons.

__attribute__((noinline))
void lt(float x, float y) {
    // LLVM optimises to `fcmp ugt y, x`
    fprintf(stderr, "%8f < %8f: %d\n", x, y, x < y || x!=x || y!=y);
}

__attribute__((noinline))
void lte(float x, float y) {
    // LLVM optimises to `fcmp uge y, x`
    fprintf(stderr, "%f <= %f: %d\n", x, y, x <= y || x!=x || y!=y);
}

__attribute__((noinline))
void gt(float x, float y) {
    // LLVM optimises to `fcmp ult y, x`
    fprintf(stderr, "%f > %f: %d\n", x, y, x > y || x!=x || y!=y);
}

__attribute__((noinline))
void gte(float x, float y) {
    // LLVM optimises to `fcmp ule y, x`
    fprintf(stderr, "%f >= %f: %d\n", x, y, x >= y || x!=x || y!=y);
}

__attribute__((noinline))
void eq(float x, float y) {
    // LLVM optimises to `fcmp ueq y, x`
    fprintf(stderr, "%f == %f: %d\n", x, y, x == y || x!=x || y!=y);
}

__attribute__((noinline))
void ne(float x, float y) {
    // LLVM optimises to `fcmp une y, x`
    fprintf(stderr, "%f != %f: %d\n", x, y, x != y || x!=x || y!=y);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3;
  float f_one = 1.0, f_two = 2.0, nan = NAN;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(f_one);
  NOOPT_VAL(f_two);
  NOOPT_VAL(nan);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);

    lt(f_one, f_one);
    lt(f_two, f_two);
    lt(f_one, f_two);
    lt(f_two, f_one);
    lt(f_one, nan);
    lt(nan, f_one);
    lt(nan, nan);
    fprintf(stderr, "---\n");
    lte(f_one, f_one);
    lte(f_two, f_two);
    lte(f_one, f_two);
    lte(f_two, f_one);
    lte(f_one, nan);
    lte(nan, f_one);
    lte(nan, nan);
    fprintf(stderr, "---\n");
    gt(f_one, f_one);
    gt(f_two, f_two);
    gt(f_one, f_two);
    gt(f_two, f_one);
    gt(f_one, nan);
    gt(nan, f_one);
    gt(nan, nan);
    fprintf(stderr, "---\n");
    gte(f_one, f_one);
    gte(f_two, f_two);
    gte(f_one, f_two);
    gte(f_two, f_one);
    NOOPT_VAL(nan);
    NOOPT_VAL(f_one);
    gte(f_one, nan);
    gte(nan, f_one);
    gte(nan, nan);
    fprintf(stderr, "---\n");
    eq(f_one, f_one);
    eq(f_two, f_two);
    eq(f_one, f_two);
    eq(f_two, f_one);
    eq(f_one, nan);
    eq(nan, f_one);
    eq(nan, nan);
    fprintf(stderr, "---\n");
    ne(f_one, f_one);
    ne(f_two, f_two);
    ne(f_one, f_two);
    ne(f_two, f_one);
    ne(f_one, nan);
    ne(nan, f_one);
    ne(nan, nan);
    fprintf(stderr, "---\n");

    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
