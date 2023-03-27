// Compiler:
// Run-time:

// Check the blockmap for this test program contains blocks.

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  void *bm = __yktrace_hwt_mapper_blockmap_new();
  assert(__yktrace_hwt_mapper_blockmap_len(bm) > 0);
  __yktrace_hwt_mapper_blockmap_free(bm);
  return (EXIT_SUCCESS);
}

// This isn't used as part of the test, but is required for this file to
// compile with ykllvm.
//
// FIXME: This only returns an integer due to a shortcoming of the stopgap interpreter:
// https://github.com/ykjit/yk/issues/537
uint32_t unused() {
  YkMT *mt = yk_mt_new(NULL);
  YkLocation loc = yk_location_new();
  while (true) {
    yk_mt_control_point(mt, &loc);
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return 0;
}
