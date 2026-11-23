// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   stderr:
//     ...
//     ;   "trid": "0",
//     ...
//     %{{val1}}: i32 = 101
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{val1}}) ; @fprintf
//     ...
//     ;   "trid": "1",
//     ...
//     %{{val2}}: i32 = 202
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{val2}}) ; @fprintf
//     ...
//     ;   "trid": "2",
//     ...
//     %{{val3}}: i32 = 303
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{val3}}) ; @fprintf
//     ...
//     ;   "trid": "3",
//     ...
//     %{{val4}}: i32 = 404
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{val4}}) ; @fprintf
//     ...
//     ;   "trid": "4",
//     ...
//     %{{val5}}: i32 = 505
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{val5}}) ; @fprintf
//     ...
//     ;   "trid": "5",
//     ...
//     %{{x}}: i32 = arg
//     ...
//     %{{factor}}: i32 = 101
//     %{{product}}: i32 = mul %{{x}}, %{{factor}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{product}}) ; @fprintf
//     ...
//     --- End hir ---
//     606
//     606
//     707
//     808
//     exit

// Check that promote calls become megamorphic (i.e. non-specialising) past the
// relevant threshold.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  int values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8};
  NOOPT_VAL(values);
  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
    yk_mt_control_point(mt, &loc);
    int x = yk_promote(values[i]);
    fprintf(stderr, "%d\n", x * 101);
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return EXIT_SUCCESS;
}
