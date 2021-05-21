// Compiler:
// Run-time:

// Check that basic trace compilation works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int
main(int argc, char **argv)
{
    void *tt = __yktrace_start_tracing(HW_TRACING);
    int res = argc + 1;
    void *tr = __yktrace_stop_tracing(tt);

    assert(res == 2);
    __yktrace_irtrace_compile(tr); // FIXME test something.

    __yktrace_drop_irtrace(tr);

    return (EXIT_SUCCESS);
}
