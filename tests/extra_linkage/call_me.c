#include <setjmp.h>
#include <stdbool.h>
#include <stdio.h>

#include "yk_testing.h"

int call_me(int x) { return 5; }

int call_me_add(int x) { return x + 1; }

int call_callback(int (*callback)(int, int), int x, int y) {
  return callback(x, y);
}
