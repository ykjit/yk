// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=jit-pre-opt

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define ITERS 5

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
    // We are about to call to an opaque (to the JIT) function.
    //
    // Generally speaking, one of three things can happen. Either:
    //
    //  1. the opaque callee returns normally, without calling back to any
    //  non-opaque code,  and without any longjmp, thus the next block SWT
    //  sees is the one after the call. In this case, we'd see the `i--`
    //  block next.
    //
    //  2. the opaque callee at some point calls a non-opaque function, in which
    //    case the next block SWT sees is the entry block to that function.
    //
    //  3. the opaque callee calls longjmp at some point, in which case we
    //  **might** be able to detect it: if the next block SWT sees is neither
    //  the `i--` block, nor an function entry block.
    //
    // But for this test, we have a scenario where the callee longjumps to
    // another opaque function further up the stack which then calls the
    // non-opaque function g(), so the next block SWT sees is the entry block
    // to g().
    //
    // The problem is that there is now ambiguity for cases 2 and 3 above.
    // The next block SWT will see is the entry block to g(), but there are two
    // ways to reach that successor: via calling deeper to g(), or via a
    // longjmp to g() (via main()).
    //
    // As it stands, the JIT assumes that the entry block to g() is a deeper
    // call to g(). It doesn't realise that a longjmp can happen and this
    // manifests as miscompilation. If you disable the JIT (by setting a high
    // hot threshold) this is a terminating test, but with the JIT enabled we
    // compile an infinite loop.
    //
    // Maybe we can fix this by checking the stack depth when encounter the
    // entry point to a function?
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
