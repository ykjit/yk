// Compiler:
// Run-time:

// Check the blockmap for this test program contains blocks.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// FIXME find a way to auto-generate these protos.
void *yktrace_hwt_mapper_blockmap_new(void);
size_t yktrace_hwt_mapper_blockmap_len(void *mapper);
void yktrace_hwt_mapper_blockmap_free(void *mapper);

int
main(int argc, char **argv)
{
    void *bm = yktrace_hwt_mapper_blockmap_new();
    assert(yktrace_hwt_mapper_blockmap_len(bm) > 0);
    yktrace_hwt_mapper_blockmap_free(bm);
    return (EXIT_SUCCESS);
}
