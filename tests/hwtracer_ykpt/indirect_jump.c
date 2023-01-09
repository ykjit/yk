// ignore: Requires ykllvm support for indirect jumps.
// Run-time:
//   stderr:
//    l2
//    l1
//    l3

// FIXME: The BlockDisambiguate pass in ykllvm currently crashes on
// indirect jumps due to unimplemented logic.

#include <inttypes.h>
#include <stdio.h>
#include <yk_testing.h>

int main(void) {
  void *t1 = &&l1;
  void *t2 = &&l2;
  void *t3 = &&l3;
  NOOPT_VAL(t1);
  NOOPT_VAL(t2);
  NOOPT_VAL(t3);

  void *tc = __hwykpt_start_collector();
  goto *t2;
l1:
  fprintf(stderr, "l1\n");
  goto *t3;
l2:
  fprintf(stderr, "l2\n");
  goto *t1;
l3:
  fprintf(stderr, "l3\n");
  void *trace = __hwykpt_stop_collector(tc);
  __hwykpt_libipt_vs_ykpt(trace);
}
