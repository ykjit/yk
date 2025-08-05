// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     in interp loop: i=4
//     yk-tracing: start-tracing
//     maybe_hidden_longjump: x=4
//     longjumped 1!
//     I'm in g via a longjump
//     in interp loop: i=3
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     maybe_hidden_longjump: x=3
//     I'm in g via calling deeper
//     in interp loop: i=2
//     yk-tracing: start-tracing
//     maybe_hidden_longjump: x=2
//     I'm in g via calling deeper
//     in interp loop: i=1
//     yk-tracing: stop-tracing
//     maybe_hidden_longjump: x=1
//     I'm in g via calling deeper
//     exit

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define G_LONGJMP     1
#define G_CALL_DEEPER 2

jmp_buf buf;

void g(int x) {
  if (x == G_LONGJMP) {
    fprintf(stderr, "I'm in g via a longjump\n");
  } else {
    fprintf(stderr, "I'm in g via calling deeper\n");
  }
}

// This function is opaque to the JIT.
__attribute__((noinline, yk_outline))
void maybe_hidden_longjump(int x) {
  fprintf(stderr, "%s: x=%d\n", __func__, x);
  if (x == 4) {
    // Jumps to main(), unwinding the interpreter loop, but leaving the JIT
    // state intact (i.e. still tracing).
    longjmp(buf, G_LONGJMP);
  } else {
    g(G_CALL_DEEPER);
  }
}

void loop(YkMT *mt, YkLocation *loc, int i) {
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    fprintf(stderr, "in interp loop: i=%d\n", i);
    yk_mt_control_point(mt, loc);
    maybe_hidden_longjump(i);
    i--;
  }

}

// This function is also opaque to the JIT.
__attribute__((noinline, yk_outline))
int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int r = setjmp(buf);
  if (r == 0) {
    loop(mt, &loc, 4);
  } else {
    fprintf(stderr, "longjumped %d!\n", r);
    g(r);
    loop(mt, &loc, 3);
  }

  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
