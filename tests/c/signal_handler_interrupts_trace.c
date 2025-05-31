// ## FIXME: currently we can't distinguish a signal handler from a regular
// ## call (or call back from unmappable code). In this test the signal handler
// ## is INLINED into trace!
// ignore-if: true
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt,aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     i=3
//     yk-tracing: start-tracing
//     i=2
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     i=1
//     exit

// Check that something sensible happens when a signal handler interrupts
// execution during tracing.

#include <assert.h>
#include <err.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int flag = 0;

__attribute__((yk_outline))
void handler(int sig) {
  flag = 1;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  signal(SIGUSR1, handler);
  pid_t self = getpid();

  int i = 3;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);

    flag = 0;
    if (kill(self, SIGUSR1) != 0) {
      errx(EXIT_FAILURE, "kill");
    }
    while (flag == 0) {
      usleep(1000); // attempt to not blow the trace buffer.
    }

    i--;
  }
  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
