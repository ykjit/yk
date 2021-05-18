// Compiler:
// Run-time:

// Check that inter-procedural tracing works.

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline))
int
add_one_or_two(int arg)
{
    if (arg % 2 == 0) {
        arg++;
    } else {
        arg += 2;
    }
    return arg;
}

int
main(int argc, char **argv)
{
    void *tt = __yktrace_start_tracing(HW_TRACING);
    argc = add_one_or_two(argc);
    void *tr = __yktrace_stop_tracing(tt);

    size_t len = __yktrace_irtrace_len(tr);
    assert(len >= 4); // At least one block for `main`, at least 3 blocks for `add_one_or_two`.
    for (int i = 0; i < len; i++) {
        char *func_name = NULL;
        size_t bb = 0;
        __yktrace_irtrace_get(tr, i, &func_name, &bb);

        // We expect blocks from `add_one_or_two` to be book-ended by `main` bb0.
        // Note that in LLVM, a call does not terminate a block.
        if ((i == 0) || (i == len - 1)) {
            assert((strcmp(func_name, "main") == 0) && (bb == 0));
        } else {
            assert(strcmp(func_name, "add_one_or_two") == 0);
        }
    }

    __yktrace_drop_irtrace(tr);
    return (EXIT_SUCCESS);
}
