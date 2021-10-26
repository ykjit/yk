// ignore: broken during new control point design
// Compiler:
// Run-time:
//   stdout:
//     Hello World!
//     Hello World!

// This is bf_base.c from https://github.com/ykjit/ykcbf modified to:
//  - hard-code the input to the interpreter (hello.bf from the same repo).
//  - replay the entire program execution via the JIT.

#include <err.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

#define CELLS_LEN 30000
#define INPUT_PROG                                                             \
  "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<++++++" \
  "+++++++++.>.+++.------.--------.>+.>."

void interp(char *prog, char *prog_end, char *cells, char *cells_end) {
  char *instr = prog;
  char *cell = cells;
  while (instr < prog_end) {
    switch (*instr) {
    case '>': {
      if (cell++ == cells_end)
        errx(1, "out of memory");
      break;
    }
    case '<': {
      if (cell > cells)
        cell--;
      break;
    }
    case '+': {
      (*cell)++;
      break;
    }
    case '-': {
      (*cell)--;
      break;
    }
    case '.': {
      if (putchar(*cell) == EOF)
        err(1, "(stdout)");
      break;
    }
    case ',': {
      if (read(STDIN_FILENO, cell, 1) == -1)
        err(1, "(stdin)");
      break;
    }
    case '[': {
      if (*cell == 0) {
        int count = 0;
        while (true) {
          instr++;
          if (*instr == ']') {
            if (count == 0)
              break;
            count--;
          } else if (*instr == '[')
            count++;
        }
      }
      break;
    }
    case ']': {
      if (*cell != 0) {
        int count = 0;
        while (true) {
          instr--;
          if (*instr == '[') {
            if (count == 0)
              break;
            count--;
          } else if (*instr == ']')
            count++;
        }
      }
      break;
    }
    default:
      break;
    }
    instr++;
  }
}

// Traces an entire execution of the program and then runs is a second time
// using JITted code. Expect all output twice in sequence.
void jit(char *prog, char *prog_end) {
  // First run collects a trace.
  char *cells = calloc(1, CELLS_LEN);
  if (cells == NULL)
    err(1, "out of memory");
  char *cells_end = cells + CELLS_LEN;

  __yktrace_start_tracing(HW_TRACING, &prog, &prog_end, &cells, &cells_end);
  NOOPT_VAL(prog);
  NOOPT_VAL(prog_end);
  NOOPT_VAL(cells);
  NOOPT_VAL(cells_end);
  interp(prog, prog_end, cells, cells_end);
  CLOBBER_MEM();
  void *tr = __yktrace_stop_tracing();

  // Compile and run trace.
  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);

  memset(cells, '\0', CELLS_LEN);
  void (*func)(void *, void *, void *, void *) =
      (void (*)(void *, void *, void *, void *))ptr;
  func(&prog, &prog_end, &cells, &cells_end);
}

int main(int argc, char *argv[]) {
  jit(INPUT_PROG, &INPUT_PROG[strlen(INPUT_PROG)]);
}
