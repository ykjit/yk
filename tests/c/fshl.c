// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=aot,hir
//   stderr:
//     5: 53 34 63
//       rotl16(21) = 31157
//     --- Begin aot ---
//     ...
//     %{{rotl_res16}}: i16 = call llvm.fshl.i16(-21555i16, -21555i16, %{{n16w}})
//     ...
//     %{{shift_res16}}: i16 = call llvm.fshl.i16(1i16, -21555i16, %{{n16}})
//     ...
//     %{{shift_res32}}: i32 = call llvm.fshl.i32(1i32, 305419896i32, %{{n32}})
//     ...
//     %{{shift_res64}}: i64 = call llvm.fshl.i64(1i64, -81985529216486896i64, %{{n64}})
//     ...
//     %{{_}}: i32 = call fprintf(%{{_}}, @.str, %{{_}}, %{{_}}, %{{_}}, %{{_}})
//     ...
//     --- End aot ---
//     --- Begin hir ---
//     ...
//     %{{hrotl_res16}}: i16 = fshl %{{_}}, %{{_}}, %{{_}}
//     ...
//     %{{hshift_res16}}: i16 = fshl %{{_}}, %{{_}}, %{{_}}
//     ...
//     %{{hzext16}}: i32 = zext %{{hshift_res16}}
//     ...
//     %{{hsel32}}: i32 = fshl %{{_}}, %{{_}}, %{{_}}
//     ...
//     %{{hshift_res64}}: i64 = fshl %{{_}}, %{{_}}, %{{_}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{hzext16}}, %{{hsel32}}, %{{hshift_res64}}) ; @fprintf
//     ...
//     %{{hzextrotl16}}: i32 = zext %{{hrotl_res16}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{hzextrotl16}}) ; @fprintf
//     ...
//     --- End hir ---
//     4: 26 17 31
//       rotl16(20) = 48346
//     3: 13 8 15
//       rotl16(19) = 24173
//     2: 6 4 7
//       rotl16(18) = 44854
//     exit

// Check that the funnel shift left is supported by the yk compiler, for i16/i32/i64.

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
  while (i > 1) {
    yk_mt_control_point(mt, &loc);

    // `__builtin_rotateleft16` maps directly to `llvm.fshl.i16`.
    unsigned int n16w = i + 16;
    NOOPT_VAL(n16w);
    uint16_t e = __builtin_rotateleft16(0xABCD, n16w);

    // This masking/guarding is needed to get clang to 
    // emit `llvm.fshl.i16/i32/i64` instead of plain shifts.
    int n16 = i & 15;
    uint16_t b = 1;
    if (n16 != 0) {
      b = ((1 << n16) | (0xABCD >> (16 - n16)));
    }

    int n32 = i & 31;
    uint32_t c = 1u;
    if (n32 != 0) {
      c = (1u << n32) | (0x12345678u >> (32 - n32));
    }

    int n64 = i & 63;
    uint64_t d = 1ull;
    if (n64 != 0) {
      d = (1ull << n64) | (0xFEDCBA9876543210ull >> (64 - n64));
    }

    fprintf(stderr, "%d: %u %u %llu\n", i, b, c, (unsigned long long)d);
    fprintf(stderr, "  rotl16(%u) = %u\n", n16w, e);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
