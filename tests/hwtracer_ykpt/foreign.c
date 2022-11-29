// Run-time:

#include <assert.h>
#include <stdio.h>
#include <yk_testing.h>

extern int call_me_add(int); // opaque to the JIT.

int main(void) {
  int i = 0;
  NOOPT_VAL(i);

  void *tc = __hwykpt_start_collector();
  i++;
  i = call_me_add(i);
  i++;
  void *trace = __hwykpt_stop_collector(tc);

  NOOPT_VAL(i);
  assert(i == 3);

  __hwykpt_libipt_vs_ykpt(trace);
}
