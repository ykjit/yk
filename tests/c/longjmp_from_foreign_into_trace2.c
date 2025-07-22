// ignore-if: test "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     yk-tracing: start-tracing
//     we jumped
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     we jumped
//     exit
//     ...

// Tests that we can deal with setjmp/longjmp when we jump from foreign code
// into a different function than started outlining.

// FIXME: We don't know yet how to handle long jumps in SWT because the
// jump might skip the block with the tracing call.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

jmp_buf buf;

// By annotating this `yk_outline` we don't serialise its IR and therefore
// can't see the `longjmp` inside. The tracebuilder is expected to notice the
// "odd" control flow that results.
__attribute__((noinline, yk_outline))
void ljmp() {
  longjmp(buf, 1);
}

__attribute__((noinline))
void inner() {
    ljmp();
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 2;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  if (setjmp(buf) != 0) {
    fprintf(stderr, "we jumped\n");
  }
  while (i > 0) {
    i--;
    yk_mt_control_point(mt, &loc);
    inner();
    fprintf(stderr, "unreachable\n");
  }
  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
