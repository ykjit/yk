// Compiler:
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     ...
//     define internal void @__yk_compiled_trace_0(i32* %0) {
//       %2 = alloca i8*, align 8
//       ; main() tests/debug_debuginfo.c:28:9
//       store i8* null, i8** %2, align 8, !dbg !3
//       ; main() tests/debug_debuginfo.c:29:7
//       store i32 2, i32* %0, align 4, !dbg !12
//       ; main() tests/debug_debuginfo.c:30:37
//       %3 = load i8*, i8** %2, align 8, !dbg !13
//       ret void
//     }
//     ...

// Check that debug information is included in module prints.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &res);
  res = 2;
  void *tr = __yktrace_stop_tracing(tt);
  __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  return (EXIT_SUCCESS);
}
