// ignore: Consistently hits `Assertion `!CallStack.curMappableFrame()' failed`
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     jit-state: enter-jit-code
//     jit-state: exit-jit-code
//     ...
//     jit-state: deoptimise
//     ...
//   stdin:
//     1234

// This is yk_simple_bf.c from https://github.com/ykjit/ykcbf modified to
// hard-code the input to the interpreter (hello.bf from the same repo).

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
// Licensed CC BY-SA 4.0 http://brainfuck.org/
// Daniel B Cristofani (cristofdathevanetdotcom)
// http://www.hevanet.com/cristofd/brainfuck/
#define INPUT_PROG                                                             \
  "+++++++++++++++++++++++++++++++++"                                          \
  "."                                                                          \
  "-----------------------"                                                    \
  "."                                                                          \
  "----------"                                                                 \
  ">>>>+>+++>+++>>>>>+++["                                                     \
  "  >,+>++++[>++++<-]>[<<[-[->]]>[<]>-]<<["                                   \
  "    >+>+>>+>+[<<<<]<+>>[+<]<[>]>+[[>>>]>>+[<<<<]>-]+<+>>>-["                \
  "      <<+[>]>>+<<<+<+<--------["                                            \
  "        <<-<<+[>]>+<<-<<-["                                                 \
  "          <<<+<-[>>]<-<-<<<-<----["                                         \
  "            <<<->>>>+<-["                                                   \
  "              <<<+[>]>+<<+<-<-["                                            \
  "                <<+<-<+[>>]<+<<<<+<-["                                      \
  "                  <<-[>]>>-<<<-<-<-["                                       \
  "                    <<<+<-[>>]<+<<<+<+<-["                                  \
  "                      <<<<+[>]<-<<-["                                       \
  "                        <<+[>]>>-<<<<-<-["                                  \
  "                          >>>>>+<-<<<+<-["                                  \
  "                            >>+<<-["                                        \
  "                              <<-<-[>]>+<<-<-<-["                           \
  "                                <<+<+[>]<+<+<-["                            \
  "                                  >>-<-<-["                                 \
  "                                    <<-[>]<+<++++[<-------->-]++<["         \
  "                                      <<+[>]>>-<-<<<<-["                    \
  "                                        <<-<<->>>>-["                       \
  "                                          <<<<+[>]>+<<<<-["                 \
  "                                            <<+<<-[>>]<+<<<<<-["            \
  "                                              >>>>-<<<-<-"                  \
  "  ]]]]]]]]]]]]]]]]]]]]]]>[>[[[<<<<]>+>>[>>>>>]<-]<]>>>+>>>>>>>+>]<"         \
  "]<[-]<<<<<<<++<+++<+++["                                                    \
  "  [>]>>>>>>++++++++[<<++++>++++++>-]<-<<[-[<+>>.<-]]<<<<["                  \
  "    -[-[>+<-]>]>>>>>[.[>]]<<[<+>-]>>>[<<++[<+>--]>>-]"                      \
  "    <<[->+<[<++>-]]<<<[<+>-]<<<<"                                           \
  "  ]>>+>>>--[<+>---]<.>>[[-]<<]<"                                            \
  "]"

// FIXME: This only returns an integer due to a shortcoming of the stopgap interpreter:
// https://github.com/ykjit/yk/issues/537
int interp(char *prog, char *prog_end, char *cells, char *cells_end, YkMT *mt,
           YkLocation *yklocs) {
  char *instr = prog;
  char *cell = cells;
  while (instr < prog_end) {
    YkLocation *loc = NULL;
    if (*instr == ']')
      loc = &yklocs[instr - prog];
    yk_mt_control_point(mt, loc);
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
  return 0;
}

int main(void) {
  char *cells = calloc(1, CELLS_LEN);
  if (cells == NULL)
    err(1, "out of memory");
  char *cells_end = cells + CELLS_LEN;

  YkMT *mt = yk_mt_new();
  yk_mt_hot_threshold_set(mt, 5);

  size_t prog_len = sizeof(INPUT_PROG);
  YkLocation *yklocs = calloc(prog_len, sizeof(YkLocation));
  if (yklocs == NULL)
    err(1, "out of memory");
  for (YkLocation *ykloc = yklocs; ykloc < yklocs + prog_len; ykloc++)
    *ykloc = yk_location_new();

  interp(INPUT_PROG, &INPUT_PROG[prog_len], cells, cells_end, mt, yklocs);

  free(cells);
  free(yklocs);
}
