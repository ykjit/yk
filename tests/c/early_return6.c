// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     i0: 5
//     yk-execution: enter-jit-code ...
//     yk-execution: return ...
//     yk-execution: enter-jit-code ...
//     yk-execution: enter-jit-code ...
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     yk-execution: deoptimise ...
//     yk-tracing: stop-tracing
//     yk-tracing: start-side-tracing
//     out1
//     yk-tracing: stop-tracing
//     yk-execution: enter-jit-code ...
//     yk-execution: return ...
//     out2
//     yk-execution: enter-jit-code ...
//     yk-execution: enter-jit-code ...
//     yk-execution: return ...
//     yk-execution: return ...
//     exit

// Check that if an executing side-trace calls into the interpreter loop, a
// trace is then started, and it returns to the sidetrace, that is noticed
// during deopt.

#include <assert.h>
#include <stdio.h>
#include <yk.h>

static YkMT *mt;
static YkLocation loc[2];
static volatile int nested, leave_inner, leave_outer;

int interp(int, int);

__attribute__((yk_outline, noinline))
int callback(int depth) {
  return depth || !nested ? 0 : interp(1, 3);
}

__attribute__((noinline))
int interp(int depth, int n) {
  int total = 0;
  while (n > 0) {
    yk_mt_control_point(mt, &loc[depth]);
    if (depth && leave_inner)
      return 7;
    total += callback(depth);
    if (!depth && leave_outer)
      return total + 111;
    n -= 1;
  }
  return total;
}

int main(void) {
  mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 1);
  loc[0] = yk_location_new();
  loc[1] = yk_location_new();
  assert(interp(1, 8) == 0);
  assert(interp(0, 8) == 0);
  for (int i = 0; i < 6; i++) {
    fprintf(stderr, "i1: %d\n", i);
    assert(interp(1, 8) == 0);
    fprintf(stderr, "i0: %d\n", i);
    assert(interp(0, 8) == 0);
  }
  nested = 1;
  leave_inner = 1;
  leave_outer = 1;
  assert(interp(0, 2) == 118);
  fprintf(stderr, "out1\n");
  assert(interp(0, 2) == 118);
  fprintf(stderr, "out2\n");
  assert(interp(0, 2) == 118);
  yk_location_drop(loc[0]);
  yk_location_drop(loc[1]);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
}
