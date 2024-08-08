// ## The function __yktrace_hwt_mapper_blockmap_len is compile-time guarded for hwt only.
// ignore-if: test "$YKB_TRACER" != "hwt"
// Compiler:
// Run-time:

// Check the blockmap for this test program contains blocks.

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  assert(__yktrace_hwt_mapper_blockmap_len() > 0);
  return (EXIT_SUCCESS);
}

// This isn't used as part of the test, but is required for this file to
// compile with ykllvm.
uint32_t unused() {
  YkMT *mt = yk_mt_new(NULL);
  YkLocation loc = yk_location_new();
  while (true) {
    yk_mt_control_point(mt, &loc);
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return 0;
}
