// ignore: broken during new control point design
// Compiler:
// Run-time:

// Check the blockmap for this test program contains blocks.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  void *bm = __yktrace_hwt_mapper_blockmap_new();
  assert(__yktrace_hwt_mapper_blockmap_len(bm) > 0);
  __yktrace_hwt_mapper_blockmap_free(bm);
  return (EXIT_SUCCESS);
}
