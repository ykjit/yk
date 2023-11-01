// Compiler:
// Run-time:
//   status: success
//   stdout: 704982704 704982704 704982704
//   stderr:

// This test tries to reproduce the thread stack race condition.
//
// Assumption:
//  Threads should have their own copy of the thread-local stack, that way there should be no race conditions between thread evaluations.
//
// Testing strategy:
//  1. Spawns `MAX_THREADS` and wait for Semaphore signal.
//     Each thread computes an iteration-based `sum` variable and prints it to `stdout`.
//  2. Assert that each thread computation is consistent and there are no race conditions.
//     For a range of 0..10000; the `sum` value is expected to be 704982704.
// Run:
//     cargo test thread_local_stack.c -- -- nocapture

#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

const int MAX_THREADS = 3;
sem_t semaphore;

void *thread_function(void *arg) {
  sem_wait(&semaphore);
  int sum = 0;
  for (int i = 0; i < 100000; i++) {
    sum += i;
  }
  printf("%d ", sum);
  return NULL;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  pthread_t thread_handles[MAX_THREADS];
  sem_init(&semaphore, MAX_THREADS, 0);

  for (int i = 0; i < MAX_THREADS; i++) {
    yk_mt_control_point(mt, &loc);
    pthread_create(&thread_handles[i], NULL, thread_function, NULL);
  }
  // Sending the signal to wake up threads
  for (int i = 0; i < MAX_THREADS; i++) {
    sem_post(&semaphore);
  }
  // Wait for threads to complete
  for (int i = 0; i < MAX_THREADS; i++) {
    pthread_join(thread_handles[i], NULL);
  }

  // Release semaphore
  sem_destroy(&semaphore);
  // Release yk resources
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return 0;
}
