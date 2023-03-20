// Run-time:

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk_testing.h>

int cmp(const void *a, const void *b) {
  int aa = *(int *)a;
  int bb = *(int *)b;

  if (aa < bb)
    return -1;
  else if (aa == bb)
    return 0;
  else
    return 1;
}

int main(void) {
  int elems[] = {4, 1, 2, 3, 0};
  int n = sizeof(elems) / sizeof(elems[0]);
  NOOPT_VAL(elems);
  NOOPT_VAL(n);

  void *tc = __hwykpt_start_collector();
  qsort(elems, n, sizeof(elems[0]), cmp);
  void *trace = __hwykpt_stop_collector(tc);

  for (int i = 0; i < n; i++)
    assert(elems[i] == i);

  __hwykpt_libipt_vs_ykpt(trace);
}
