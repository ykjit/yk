// Compiler:
// Run-time:
//   status: success
//   stdout: 1,10,11,12,1,
//   stderr:

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

void *thread_function(void *arg) {
  int intVar = 42;
  char charVar = 'A';
  float floatVar = 3.14f;
  double doubleVar = 2.71828;
  long longVar = 1234567890;
  short shortVar = 10;
  unsigned int uintVar = 100;
  unsigned char ucharVar = 'B';
  unsigned short ushortVar = 20;
  long double longDoubleVar = 1.234567890123456789;
  unsigned long ulongVar = 987654321;
  unsigned long long ulongLongVar = 9876543210;
  signed char scharVar = -1;
  signed short sshortVar = -10;
  signed int sintVar = -100;
  signed long slongVar = -1000;
  signed long long slongLongVar = -10000;
  unsigned long long uslongLongVar = 18446744;

  int *loop_count = (int *)arg;
  double timestamp = (double)clock();
  for (int i = 0; i < *loop_count; i++) {

    int intVar2 = 42 + i;
    char charVar2 = 'A' + i;
    float floatVar2 = 3.14f + i;
    double doubleVar2 = 2.71828 + i;
    long longVar2 = 1234567890 + i;
    short shortVar2 = 10 + i;
    unsigned int uintVar2 = 100 + i;
    unsigned char ucharVar2 = 'B' + i;
    unsigned short ushortVar2 = 20 + i;
    long double longDoubleVar2 = 1.234567890123456789 + i;
    unsigned long ulongVar2 = 987654321 + i;
    signed char scharVar2 = -1 + i;
    signed short sshortVar2 = -10 + i;
    signed int sintVar2 = -100 + i;
    signed long slongVar2 = -1000 + i;
    unsigned long long ulongLongVar2 = 9876543210 + i;
    signed long long slongLongVar2 = -10000 + i;
    unsigned long long uslongLongVar2 = 18446744 + i;

    intVar += intVar2 + timestamp;
    charVar += charVar2 + timestamp;
    floatVar += floatVar2 + timestamp;
    doubleVar -= doubleVar2 + timestamp;
    longVar *= longVar2 + timestamp;
    shortVar += shortVar2 + timestamp;
    uintVar += uintVar2 + timestamp;
    ucharVar += ucharVar2 + timestamp;
    ushortVar += ushortVar2 + timestamp;
    longDoubleVar += longDoubleVar2 + timestamp;
    ulongVar -= ulongVar2 + timestamp;
    ulongLongVar += ulongLongVar2 + timestamp;
    scharVar += scharVar2 + timestamp;
    sshortVar += sshortVar2 + timestamp;
    sintVar -= sintVar2 + timestamp;
    slongVar += slongVar2 + timestamp;
    slongLongVar -= slongLongVar2 + timestamp;
    uslongLongVar += uslongLongVar2 + timestamp;
  }

  unsigned long long overflowed_var =
      intVar + charVar + floatVar + doubleVar + longVar + shortVar + uintVar +
      ucharVar + ushortVar + longDoubleVar + ulongVar + ulongLongVar +
      scharVar + sshortVar + sintVar + slongVar + slongLongVar + uslongLongVar;

  // Note: This condition is here only so that overflowed_var will be used.
  if (overflowed_var == 0) {
    printf("");
  }
  printf("%d,", *loop_count);
  return NULL;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int MAX_THREADS = 3;
  int thread_args[3] = {10, 11, 12};
  pthread_t thread_handles[MAX_THREADS];
  printf("%d,", argc);

  int a = argc;

  for (int i = 0; i < MAX_THREADS; i++) {
    yk_mt_control_point(mt, &loc);
    pthread_create(&thread_handles[i], NULL, thread_function, &thread_args[i]);
    pthread_join(thread_handles[i], NULL);
  }
  sleep(4);
  // TODO: we expect variable a to be overriden by thread_function as it is currently not using the thread-local stack
  printf("%d,", a);

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return 0;
}
