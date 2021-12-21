// Compiler:
// Run-time:

// Check the blockmap for this test program contains blocks.

#include <assert.h>
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
void unused() {
  YkLocation loc = yk_location_new();
  yk_control_point(&loc);
}
