// ignore-if: test $(uname -m) != x86_64
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=-:jit-asm
//   env-var: YKD_SERIALISE_COMPILATION=1
//
//   stderr:
//     ...
//     --- Begin jit-asm ---
//     ...
//     ; %{{_}}: i64 = add %{{_}}, 9223372036854775807i64
//     {{_}} {{_}}: mov r{{a}}, 0x7FFFFFFFFFFFFFFF
//     {{_}} {{_}}: add r{{_}}, r{{a}}
//     ...
//     ; %{{_}}: i64 = add %{{_}}, 30i64
//     {{_}} {{_}}: add r{{_}}, 0x1E
//     ...
//     ; %{{_}}: i64 = add %{{_}}, 18446744073709551615i64
//     {{_}} {{_}}: add r{{_}}, 0xFFFFFFFFFFFFFFFF
//     ...
//     --- End jit-asm ---
//     ...

// Test emission of X64 add instructions with immediate operands.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  uint64_t x = 100;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(x);
    // INT64_MAX too big for an immediate operand.
    fprintf(stderr, "%" PRIu64 "\n", x + INT64_MAX);

    // 30 can be imm8 operand.
    fprintf(stderr, "%" PRIu64 "\n", x + 30);

    // Can use imm8 operand, but note that the disassembler sign extends the
    // operand up to 64-bit when it prints the operand.
    fprintf(stderr, "%" PRIu64 "\n", x + -1);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
