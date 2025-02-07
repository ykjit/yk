// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stdout:
//     exit

// Test case extracted from lua.
// Testing sidetraces where the first traced block contains a PHI node. We need
// to make sure to set LastBB when initialising the CallStack. There's nothing
// specific we want to check here, except that this doesn't crash.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

struct CallInfo {
  void *savedpc;
};

struct lua_State {
  struct CallInfo *ci;
  int hookmask;
};

int luaG_traceexec(struct lua_State *L, void *pc, int hookcount) {
  struct CallInfo *ci = L->ci;
  int mask = L->hookmask;
  int counthook;
  ci->savedpc = pc; /* save 'pc' */
  counthook = (hookcount > 10 && mask);
  if (counthook)
    return 2; /* reset count */
  return 1;   /* keep 'trap' on */
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 5);
  YkLocation loc = yk_location_new();

  int i = 20;
  struct CallInfo ci = {NULL};
  struct lua_State L = {&ci, 0};
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    luaG_traceexec(&L, NULL, i);
    i--;
  }
  printf("exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
