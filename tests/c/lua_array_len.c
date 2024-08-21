// ## yk-config-env: YKB_AOT_OPTLEVEL=1
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4

// This was reduced from yklua.
//
// In short a `ptradd %ptr, 0` was reducing to nothing in the tracebuilder and
// the AOT instruction was being associated with the last JIT instruction
// (which was nothing to do with the `ptr_add`!).
//
// This caused a runtime crash.
//
// I tried to reduce this to a test with a simple constant 0-byte offset field
// access, but I can't get ykllvm to emit a constant zero offset GEP when
// mem2reg is enabled.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

typedef unsigned char lu_byte;
typedef long long lua_Integer;
typedef unsigned long long lua_Unsigned;

typedef union Value {
  lua_Integer i;
} Value;

typedef struct TValue {
  Value value_; lu_byte tt_;
} TValue;

typedef union StackValue {
  TValue val;
  struct {
    Value value_; lu_byte tt_;
    unsigned short delta;
  } tbclist;
} StackValue;

typedef StackValue *StkId;
typedef struct Table {} Table;

union GCUnion {
  struct Table h;
};

__attribute__((noinline))
lua_Unsigned luaH_getn(Table *t) { return 0; }

__attribute__((noinline))
void luaV_objlen(StkId ra, const TValue *rb) {
  Table *h = &((union GCUnion *)rb)->h;
  TValue *io = (TValue *) ra;
  io->value_.i = luaH_getn(h);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  StackValue sv;
  sv.val.value_.i = 1;
  TValue tv;
  tv.value_.i = 1;
  StkId ra = &sv;
  TValue *rb = &tv;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(ra);
  NOOPT_VAL(rb);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    luaV_objlen(ra, rb);
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
