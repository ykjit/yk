// Compiler:
// Run-time:

// Check that compiling and running traces in parallel works.
//
// FIXME An optimising compiler can remove all of the code between start/stop
// tracing.

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

#ifdef linux
#include <sys/sysinfo.h>
#endif

static void *trace(void *unused) {
  for (int i = 0; i < 3; i++) {
    void *tt = __yktrace_start_tracing(HW_TRACING);
    int res = 1 + 1;
    void *tr = __yktrace_stop_tracing(tt);
    assert(res == 2);

    void *ptr = __yktrace_irtrace_compile(tr);
    __yktrace_drop_irtrace(tr);
    void (*func)() = (void (*)())ptr;
    func();
  }
  return NULL;
}

int main() {
#ifdef linux
  int n_thr = get_nprocs();
#else
#error unimplemented
#endif

  pthread_t tids[n_thr];
  for (int i = 0; i < n_thr; i++)
    if (pthread_create(&tids[i], NULL, trace, NULL) != 0)
      err(EXIT_FAILURE, "pthread_create");

  for (int i = 0; i < n_thr; i++)
    if (pthread_join(tids[i], NULL) != 0)
      err(EXIT_FAILURE, "pthread_join");

  return (EXIT_SUCCESS);
}
