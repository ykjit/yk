// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=2
//     4 6 1 3 2 5 end
//     yk-tracing: stop-tracing
//     i=3
//     1 4 6 3 2 5 end
//     yk-execution: enter-jit-code
//     i=4
//     1 3 4 6 2 5 end
//     i=5
//     1 2 3 4 6 5 end
//     yk-execution: deoptimise ...

// Check that foreign code calling back to "native" code works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

#define N_ELEMS 6

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

void print_elems(int elems[]) {
  for (int i = 0; i < N_ELEMS; i++)
    fprintf(stderr, "%d ", elems[i]);
  fprintf(stderr, "end\n");
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 2;
  int elems[N_ELEMS] = {6, 4, 1, 3, 2, 5};
  NOOPT_VAL(elems);
  NOOPT_VAL(i);
  NOOPT_VAL(loc);

  while (i < N_ELEMS) {
    yk_mt_control_point(mt, &loc);
    // sort the first `i` elements.
    qsort(elems, i, sizeof(elems[0]), cmp);
    fprintf(stderr, "i=%d\n", i);
    print_elems(elems);
    i++;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
