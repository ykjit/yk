#define _GNU_SOURCE

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <syscall.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>

// FIXME: This file currently only expects to collect PT traces.
#define SYSFS_PT_TYPE "/sys/bus/event_source/devices/intel_pt/type"
#define MAX_PT_TYPE_STR 8

#define AUX_BUF_WAKE_RATIO 0.5

#ifndef INFTIM
#define INFTIM -1
#endif

// The bit in the IA32_RTIT_CTL MSR that disables compressed returns.
#define IA32_RTIT_CTL_DISRETC 1 << 11

void get_tracing_extent(char **, size_t *, size_t *);

enum hwt_cerror_kind {
  hwt_cerror_unused = 0,
  hwt_cerror_unknown = 1,
  hwt_cerror_errno = 2,
#if defined(__x86_64__)
  hwt_cerror_pt = 3,
#endif
};

#if defined(__x86_64__)
/// A mirror of PTErrorCode in the Rust code.
///
/// See the Rust code for descriptions of these error kinds.
enum hwt_pt_kind {
  hwt_pt_aux_overflow = 0,
  hwt_pt_trace_capacity = 1,
  hwt_pt_event_lost = 2
};
#endif

struct hwt_cerror {
  enum hwt_cerror_kind kind; // What sort of error is this?
  int code;                  // The error code itself.
};

/*
 * Stores all information about the collector.
 * Exposed to Rust only as an opaque pointer.
 */
struct hwt_perf_ctx {
  pthread_t collector_thread; // Collector thread handle.
  struct hwt_cerror
      collector_thread_err; // Errors from inside the tracer thread.
  int stop_fds[2];          // Pipe used to stop the poll loop.
  int perf_fd;              // FD used to talk to the perf API.
  void *aux_buf;            // Ptr to the start of the the AUX buffer.
  size_t aux_bufsize;       // The size of the AUX buffer's mmap(2).
  void *base_buf;           // Ptr to the start of the base buffer.
  size_t base_bufsize;      // The size the base buffer's mmap(2).
};

/*
 * Setting up perf is slow, so we cache a number of things.
 *
 * Since we attach perf to a thread, all of this is per-thread too.
 */
struct hwt_perf_thread_cached {
  int perf_fd;
  void *aux_buf;
  size_t aux_bufsize;
  void *base_buf;
  size_t base_bufsize;
};

thread_local struct hwt_perf_thread_cached thread_cached = { -1, NULL, 0, NULL, 0 };

/*
 * Passed from Rust to C to configure tracing.
 * Must stay in sync with the Rust struct `PerfCollectorConfig`.
 */
struct hwt_perf_collector_config {
  size_t data_bufsize;      // Data buf size (in pages).
  size_t aux_bufsize;       // AUX buf size (in pages).
  size_t trace_result_size; // Size of the `malloc`ed return block (in bytes).
  bool use_pt_filtering;    // When true use PT IP filtering.
};

/*
 * The manually malloc/free'd buffer managed by the Rust side.
 * To understand why this is split out from `struct hwt_perf_trace`, see the
 * corresponding struct in the Rust side.
 */
struct hwt_perf_trace_buf {
  void *p;
};

/*
 * Storage for a trace.
 *
 * Shared with Rust code. Must stay in sync.
 */
struct hwt_perf_trace {
  struct hwt_perf_trace_buf buf;
  __u64 len;
  __u64 capacity;
};

/*
 * Stuff used in the collector thread
 */
struct collector_thread_args {
  int perf_fd;                  // Perf notification fd.
  int stop_fd_rd;               // Polled for "stop" event.
  sem_t *collector_init_sem;    // Tracer init sync.
  struct hwt_perf_trace *trace; // Pointer to trace storage.
  void *aux_buf;                // The AUX buffer itself;
  struct perf_event_mmap_page
      *base_header;       // Pointer to the header in the base buffer.
  struct hwt_cerror *err; // Errors generated inside the thread.
};

// A data buffer sample indicating that new data is available in the AUX
// buffer. This struct is not defined in a perf header, so we have to define it
// ourselves.
struct perf_record_aux_sample {
  struct perf_event_header header;
  __u64 aux_offset;
  __u64 aux_size;
  __u64 flags;
  // ...
  // More variable-sized data follows, but we don't use it.
};

// The format of the data returned by read(2) on a Perf file descriptor.
// Note that the size of this will change if you change the Perf `read_format`
// config field (more fields become available).
struct read_format {
  __u64 value;
};

// Private prototypes.
static bool handle_sample(void *, struct perf_event_mmap_page *,
                          struct hwt_perf_trace *, void *, struct hwt_cerror *);
static bool read_aux(void *, struct perf_event_mmap_page *,
                     struct hwt_perf_trace *, struct hwt_cerror *);
static bool poll_loop(int, int, struct perf_event_mmap_page *, void *,
                      struct hwt_perf_trace *, struct hwt_cerror *);
static void *collector_thread(void *);
static int open_perf(size_t, bool, struct hwt_cerror *);
void hwt_set_cerr(struct hwt_cerror *, int, int);

// Exposed Prototypes.
struct hwt_perf_ctx *hwt_perf_init_collector(struct hwt_perf_collector_config *,
                                             struct hwt_cerror *);
bool hwt_perf_start_collector(struct hwt_perf_ctx *, struct hwt_perf_trace *,
                              struct hwt_cerror *);
bool hwt_perf_stop_collector(struct hwt_perf_ctx *tr_ctx, struct hwt_cerror *);
bool hwt_perf_free_collector(struct hwt_perf_ctx *tr_ctx, struct hwt_cerror *);

// Sets C error information (if not already set).
void hwt_set_cerr(struct hwt_cerror *err, int kind, int code) {
  // Only set the error info if we would not be overwriting an earlier error.
  if (err->kind == hwt_cerror_unused) {
    err->kind = kind;
    err->code = code;
  }
}

/*
 * Called when the poll(2) loop is woken up with a POLL_IN. Samples are read
 * from the Perf data buffer and an action is invoked for each depending its
 * type.
 *
 * Returns true on success, or false otherwise.
 */
static bool handle_sample(void *aux_buf, struct perf_event_mmap_page *hdr,
                          struct hwt_perf_trace *trace, void *data_tmp,
                          struct hwt_cerror *err) {
  // We need to use atomics with orderings to protect against 2 cases.
  //
  // 1) It must not be possible to read the data buffer before the most
  //    recent head is obtained. This would mean that we may read nothing when
  //    there is really data available.
  //
  // 2) We must ensure that we have already copied out of the data buffer
  //    before we update the tail. Failure to do so would allow the kernel to
  //    re-use the space we have just "marked free" before we copied it.
  //
  // The initial load of the tail is relaxed since we are the only thread
  // mutating it and we don't mind variations on the ordering.
  //
  // See the following comment in the Linux kernel sources for more:
  // https://github.com/torvalds/linux/blob/3be4aaf4e2d3eb95cce7835e8df797ae65ae5ac1/kernel/events/ring_buffer.c#L60-L85
  void *data = (void *)hdr + hdr->data_offset;
  __u64 head_monotonic = atomic_load_explicit((_Atomic __u64 *)&hdr->data_head,
                                              memory_order_acquire);
  __u64 size = hdr->data_size;        // No atomic load. Constant value.
  __u64 head = head_monotonic % size;
  __u64 tail_monotonic = atomic_load_explicit((_Atomic __u64 *)&hdr->data_tail,
                                    memory_order_relaxed);
  __u64 tail = tail_monotonic % size;

  // Copy samples out, removing wrap in the process.
  void *data_tmp_end = data_tmp;
  if (tail <= head) {
    // Not wrapped.
    memcpy(data_tmp, data + tail, head - tail);
    data_tmp_end += head - tail;
  } else {
    // Wrapped.
    memcpy(data_tmp, data + tail, size - tail);
    data_tmp_end += size - tail;
    memcpy(data_tmp + size - tail, data, head);
    data_tmp_end += head;
  }

  // Inform the kernel that we've read up until the head. We write back the
  // *unwrapped* value.
  atomic_store_explicit((_Atomic __u64 *)&hdr->data_tail, head_monotonic,
                        memory_order_relaxed);

  void *next_sample = data_tmp;
  while (next_sample < data_tmp_end) {
    struct perf_event_header *sample_hdr = next_sample;
    struct perf_record_aux_sample *rec_aux_sample;
    switch (sample_hdr->type) {
    case PERF_RECORD_AUX:
      // Data was written to the AUX buffer.
      rec_aux_sample = next_sample;
      // Check that the data written into the AUX buffer was not
      // truncated. If it was, then we didn't read out of the data buffer
      // quickly/frequently enough.
      if (rec_aux_sample->flags & PERF_AUX_FLAG_TRUNCATED) {
        hwt_set_cerr(err, hwt_cerror_pt, hwt_pt_aux_overflow);
        return false;
      }
      if (read_aux(aux_buf, hdr, trace, err) == false) {
        return false;
      }
      break;
    case PERF_RECORD_LOST:
      hwt_set_cerr(err, hwt_cerror_pt, hwt_pt_event_lost);
      return false;
      break;
    case PERF_RECORD_LOST_SAMPLES:
      // Shouldn't happen with PT.
      errx(EXIT_FAILURE, "Unexpected PERF_RECORD_LOST_SAMPLES sample");
      break;
    case PERF_RECORD_AUX_OUTPUT_HW_ID:
      errx(EXIT_FAILURE, "Unimplemented PERF_RECORD_AUX_OUTPUT_HW_ID sample");
      break;
    case PERF_RECORD_ITRACE_START:
      // These sample kinds are uninteresting to us.
      break;
    default:
      // In the interest of exhaustive matching, catch anything we've not seen
      // or thought about before and handle them if/when they arise.
      errx(EXIT_FAILURE, "unhandled sample type: %d\n", sample_hdr->type);
      break;
    }
    next_sample += sample_hdr->size;
  }
  assert(next_sample == data_tmp_end);

  return true;
}

/*
 * Read data out of the AUX buffer.
 *
 * Reads from `aux_buf` (whose meta-data is in `hdr`) into `trace`.
 */
bool read_aux(void *aux_buf, struct perf_event_mmap_page *hdr,
              struct hwt_perf_trace *trace, struct hwt_cerror *err) {
  // Use of atomics here for the same reasons as for handle_sample().
  __u64 head_monotonic = atomic_load_explicit((_Atomic __u64 *)&hdr->aux_head,
                                              memory_order_acquire);
  __u64 size = hdr->aux_size;         // No atomic load. Constant value.
  __u64 head = head_monotonic % size;
  __u64 tail_monotonic = atomic_load_explicit((_Atomic __u64 *)&hdr->aux_tail,
                                    memory_order_relaxed);
  __u64 tail = tail_monotonic % size;

  // Figure out how much more space we need in the trace storage buffer.
  __u64 new_data_size;
  if (tail <= head) {
    // No wrap-around.
    new_data_size = head - tail;
  } else {
    // Wrap-around.
    new_data_size = (size - tail) + head;
  }

  __u64 required_capacity = trace->len + new_data_size;
  if (required_capacity > trace->capacity) {
    // FIXME: Reallocate the trace storage buffer if more space is required.
    // Note that this requires careful synchronisation between the collection
    // thread, the main thread, and Rust code.
    hwt_set_cerr(err, hwt_cerror_pt, hwt_pt_trace_capacity);
    return false;
  }

  // Finally append the new AUX data to the end of the trace storage buffer.
  if (tail <= head) {
    memcpy(trace->buf.p + trace->len, aux_buf + tail, head - tail);
    trace->len += head - tail;
  } else {
    memcpy(trace->buf.p + trace->len, aux_buf + tail, size - tail);
    trace->len += size - tail;
    memcpy(trace->buf.p + trace->len, aux_buf, head);
    trace->len += head;
  }

  // Inform the kernel that we've read up until the head. We write back the
  // *unwrapped* value.
  atomic_store_explicit((_Atomic __u64 *)&hdr->aux_tail, head_monotonic,
                        memory_order_release);
  return true;
}

/*
 * Take trace data out of the AUX buffer.
 *
 * Returns true on success and false otherwise.
 */
static bool poll_loop(int perf_fd, int stop_fd,
                      struct perf_event_mmap_page *mmap_hdr, void *aux,
                      struct hwt_perf_trace *trace, struct hwt_cerror *err) {
  int n_events = 0;
  bool ret = true;
  struct pollfd pfds[2] = {{perf_fd, POLLIN | POLLHUP, 0},
                           {stop_fd, POLLHUP, 0}};

  // Temporary space for new samples in the data buffer.
  void *data_tmp = malloc(mmap_hdr->data_size);
  if (data_tmp == NULL) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto done;
  }

  while (1) {
    n_events = poll(pfds, 2, INFTIM);
    if (n_events == -1) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      ret = false;
      goto done;
    }

    // POLLIN on pfds[0]: Overflow event on either the Perf AUX or data buffer.
    // POLLHUP on pfds[1]: Trace collection stopped by parent.
    if ((pfds[0].revents & POLLIN) || (pfds[1].revents & POLLHUP)) {
      // Read from the Perf file descriptor.
      // We don't actually use any of what we read, but it's probably
      // best that we drain the fd anyway.
      struct read_format fd_data;
      if (pfds[0].revents & POLLIN) {
        if (read(perf_fd, &fd_data, sizeof(fd_data)) == -1) {
          hwt_set_cerr(err, hwt_cerror_errno, errno);
          ret = false;
          break;
        }
      }

      if (!handle_sample(aux, mmap_hdr, trace, data_tmp, err)) {
        ret = false;
        break;
      }

      if (pfds[1].revents & POLLHUP) {
        break;
      }
    }

    // The traced thread exited.
    if (pfds[0].revents & POLLHUP) {
      break;
    }
  }

done:
  if (data_tmp != NULL) {
    free(data_tmp);
  }

  return ret;
}

/*
 * Opens the perf file descriptor and returns it.
 *
 * Returns a file descriptor, or -1 on error.
 */
static int open_perf(size_t aux_bufsize, bool filter, struct hwt_cerror *err) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.size = sizeof(struct perf_event_attr);

  // Disable compressed returns for now.
  //
  // FIXME: https://github.com/ykjit/yk/issues/874
  attr.config |= IA32_RTIT_CTL_DISRETC;

  int ret = -1;

  // Get the perf "type" for Intel PT.
  FILE *pt_type_file = fopen(SYSFS_PT_TYPE, "r");
  if (pt_type_file == NULL) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = -1;
    goto clean;
  }
  char pt_type_str[MAX_PT_TYPE_STR];
  if (fgets(pt_type_str, sizeof(pt_type_str), pt_type_file) == NULL) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = -1;
    goto clean;
  }
  attr.type = atoi(pt_type_str);

  // Exclude the kernel.
  attr.exclude_kernel = 1;

  // Exclude the hyper-visor.
  attr.exclude_hv = 1;

  // Start disabled.
  attr.disabled = 1;

  // No skid.
  attr.precise_ip = 3;

  // Notify for every sample.
  attr.watermark = 1;
  attr.wakeup_watermark = 1;

  // Generate a PERF_RECORD_AUX sample when the AUX buffer is almost full.
  attr.aux_watermark =
      (size_t)((double)aux_bufsize * getpagesize()) * AUX_BUF_WAKE_RATIO;

  pid_t target_tid = syscall(__NR_gettid);
  ret = syscall(SYS_perf_event_open, &attr, target_tid, -1, -1, 0);
  if (ret == -1) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    goto clean;
  }

  // `ret` now holds an open perf fd.
  //
  // If requested, now configure the address range to trace. We do this so that
  // we don't trace (and later, map) externally compiled code that we could
  // never build traces for anyway.
  if (filter) {
    char *fltr_obj = NULL;
    size_t fltr_base_off = 0, fltr_size = 0;
    get_tracing_extent(&fltr_obj, &fltr_base_off, &fltr_size);

    char f_str[512];
    if ((size_t) snprintf(f_str, sizeof(f_str), "filter 0x%zx/%zu@%s",
          fltr_base_off, fltr_size, fltr_obj) >= sizeof(f_str))
    {
      hwt_set_cerr(err, hwt_cerror_unknown, 0);
      close(ret);
      ret = -1;
      goto clean;
    }

    if (ioctl(ret, PERF_EVENT_IOC_SET_FILTER, f_str) == -1) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      close(ret);
      ret = -1;
      goto clean;
    }
  }

clean:
  if ((pt_type_file != NULL) && (fclose(pt_type_file) == -1)) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = -1;
  }

  return ret;
}

/*
 * Set up Intel PT buffers and start a poll() loop for reading out the trace.
 *
 * Returns true on success and false otherwise.
 */
static void *collector_thread(void *arg) {
  struct collector_thread_args *thr_args = (struct collector_thread_args *)arg;
  int sem_posted = false;
  bool ret = true;

  // Copy arguments for the poll loop, as when we resume the parent thread,
  // `thr_args', which is on the parent thread's stack, will become unusable.
  int perf_fd = thr_args->perf_fd;
  int stop_fd_rd = thr_args->stop_fd_rd;
  struct hwt_perf_trace *trace = thr_args->trace;
  void *aux_buf = thr_args->aux_buf;
  struct perf_event_mmap_page *base_header = thr_args->base_header;
  struct hwt_cerror *err = thr_args->err;

  // Resume the interpreter loop.
  if (sem_post(thr_args->collector_init_sem) != 0) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto clean;
  }
  sem_posted = true;

  // Start reading out of the AUX buffer.
  if (!poll_loop(perf_fd, stop_fd_rd, base_header, aux_buf, trace, err)) {
    ret = false;
    goto clean;
  }

clean:
  if (!sem_posted) {
    sem_post(thr_args->collector_init_sem);
  }

  return (void *)ret;
}

/*
 * --------------------------------------
 * Functions exposed to the outside world
 * --------------------------------------
 */

/*
 * Initialise a collector context.
 */
struct hwt_perf_ctx *
hwt_perf_init_collector(struct hwt_perf_collector_config *tr_conf,
                        struct hwt_cerror *err) {
  struct hwt_perf_ctx *tr_ctx = NULL;
  bool failing = false;

  // Allocate and initialise collector context.
  tr_ctx = malloc(sizeof(*tr_ctx));
  if (tr_ctx == NULL) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    failing = true;
    goto clean;
  }

  // Set default values.
  memset(tr_ctx, 0, sizeof(*tr_ctx));
  tr_ctx->stop_fds[0] = tr_ctx->stop_fds[1] = -1;
  tr_ctx->perf_fd = -1;

  // Obtain a file descriptor through which to speak to perf.
  if (thread_cached.perf_fd == -1) {
    tr_ctx->perf_fd = open_perf(tr_conf->aux_bufsize, tr_conf->use_pt_filtering, err);
    if (tr_ctx->perf_fd == -1) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      failing = true;
      goto clean;
    }
    thread_cached.perf_fd = tr_ctx->perf_fd;
  } else {
    tr_ctx->perf_fd = thread_cached.perf_fd;
  }

  // Allocate mmap(2) buffers for speaking to perf.
  //
  // We mmap(2) two separate regions from the perf file descriptor into our
  // address space:
  //
  // 1) The base buffer (tr_ctx->base_buf), which looks like this:
  //
  // -----------------------------------
  // | header  |       data buffer     |
  // -----------------------------------
  //           ^ header->data_offset
  //
  // 2) The AUX buffer (tr_ctx->aux_buf), which is a simple array of bytes.
  //
  // The AUX buffer is where the kernel exposes control flow packets, whereas
  // the data buffer is used for all other kinds of packet.

  // Allocate the base buffer.
  //
  // Data buffer is preceded by one management page (the header), hence `1 +
  // data_bufsize'.
  int page_size = getpagesize();
  size_t base_bufsize = (1 + tr_conf->data_bufsize) * page_size;
  size_t aux_bufsize = tr_conf->aux_bufsize * page_size;
  if (thread_cached.base_buf == NULL) {
    tr_ctx->base_bufsize = base_bufsize;
    tr_ctx->base_buf = mmap(NULL, tr_ctx->base_bufsize, PROT_WRITE, MAP_SHARED,
        tr_ctx->perf_fd, 0);
    if (tr_ctx->base_buf == MAP_FAILED) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      failing = true;
      goto clean;
    }

    // Populate the header part of the base buffer.
    struct perf_event_mmap_page *base_header = tr_ctx->base_buf;
    base_header->aux_offset = base_header->data_offset + base_header->data_size;
    base_header->aux_size = tr_ctx->aux_bufsize = aux_bufsize;

    thread_cached.base_bufsize = tr_ctx->base_bufsize;
    thread_cached.base_buf = tr_ctx->base_buf;
    thread_cached.aux_bufsize = tr_ctx->aux_bufsize;
  } else {
    if (base_bufsize != thread_cached.base_bufsize) {
      errx(EXIT_FAILURE, "base buffer size mismatch");
    }
    if (aux_bufsize != thread_cached.aux_bufsize) {
      errx(EXIT_FAILURE, "aux buffer size mismatch");
    }
    tr_ctx->base_bufsize = thread_cached.base_bufsize;
    tr_ctx->base_buf = thread_cached.base_buf;
    tr_ctx->aux_bufsize = thread_cached.aux_bufsize;
  }

  // Allocate the AUX buffer.
  struct perf_event_mmap_page *base_header = tr_ctx->base_buf;
  if (thread_cached.aux_buf == NULL) {
    tr_ctx->aux_buf = mmap(NULL, base_header->aux_size, PROT_READ | PROT_WRITE,
        MAP_SHARED, tr_ctx->perf_fd, base_header->aux_offset);
    if (tr_ctx->aux_buf == MAP_FAILED) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      failing = true;
      goto clean;
    }
    thread_cached.aux_buf = tr_ctx->aux_buf;
  } else {
    tr_ctx->aux_buf = thread_cached.aux_buf;
  }

clean:
  if (failing && (tr_ctx != NULL)) {
    hwt_perf_free_collector(tr_ctx, err);
    return NULL;
  }
  return tr_ctx;
}

/*
 * Turn on Intel PT.
 *
 * `trace_bufsize` is the starting capacity of the trace buffer.
 *
 * The trace is written into `*trace_buf` which may be realloc(3)d. The trace
 * length is written into `*trace_len`.
 *
 * Returns true on success or false otherwise.
 */
bool hwt_perf_start_collector(struct hwt_perf_ctx *tr_ctx,
                              struct hwt_perf_trace *trace,
                              struct hwt_cerror *err) {
  int clean_sem = 0, clean_thread = 0;
  int ret = true;

  // When a tracing session bombs out prematurely (e.g. PT overflow), it can
  // leave the ring buffers in an inconsistent state meaning that they can't be
  // re-used properly. Here we reset the tails to the heads thus resetting the
  // ring buffers to the empty state.
  //
  // Data buffer:
  struct perf_event_mmap_page *base_header = tr_ctx->base_buf;
  __u64 data_head_monotonic = atomic_load_explicit(
      (_Atomic __u64 *)&base_header->data_head,
      memory_order_acquire);
  atomic_store_explicit((_Atomic __u64 *) &base_header->data_tail,
      data_head_monotonic, memory_order_relaxed);
  // AUX buffer:
  __u64 aux_head_monotonic = atomic_load_explicit(
      (_Atomic __u64 *)&base_header->aux_head, memory_order_acquire);
  atomic_store_explicit((_Atomic __u64 *) &base_header->aux_tail,
      aux_head_monotonic, memory_order_relaxed);

  // A pipe to signal the trace thread to stop.
  //
  // It has to be a pipe becuase it needs to be used in a poll(6) loop later.
  if (pipe(tr_ctx->stop_fds) != 0) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto clean;
  }

  // Use a semaphore to wait for the collector to be ready.
  sem_t collector_init_sem;
  if (sem_init(&collector_init_sem, 0, 0) == -1) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto clean;
  }
  clean_sem = 1;

  // The collector context contains an error struct for tracking any errors
  // coming from inside the thread. We initialise it to "no errors".
  tr_ctx->collector_thread_err.kind = hwt_cerror_unused;
  tr_ctx->collector_thread_err.code = 0;

  // Build the arguments struct for the collector thread.
  struct collector_thread_args thr_args = {
      tr_ctx->perf_fd,
      tr_ctx->stop_fds[0],
      &collector_init_sem,
      trace,
      tr_ctx->aux_buf,
      tr_ctx->base_buf, // The header is the first region in the base buf.
      &tr_ctx->collector_thread_err,
  };

  // Spawn a thread to deal with copying out of the PT AUX buffer.
  int rc = pthread_create(&tr_ctx->collector_thread, NULL, collector_thread,
                          &thr_args);
  if (rc) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto clean;
  }
  clean_thread = 1;

  // Wait for the collector to initialise, and check it didn't fail.
  rc = -1;
  while (rc == -1) {
    rc = sem_wait(&collector_init_sem);
    if ((rc == -1) && (errno != EINTR)) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      ret = false;
      goto clean;
    }
  }

  // Turn on tracing hardware.
  if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
    goto clean;
  }

clean:
  if ((clean_sem) && (sem_destroy(&collector_init_sem) == -1)) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
  }

  if (!ret) {
    if (clean_thread) {
      close(tr_ctx->stop_fds[1]); // signals thread to stop.
      tr_ctx->stop_fds[1] = -1;
      pthread_join(tr_ctx->collector_thread, NULL);
      close(tr_ctx->stop_fds[0]);
      tr_ctx->stop_fds[0] = -1;
    }
  }

  return ret;
}

/*
 * Turn off trace collection.
 *
 * Arguments:
 *   tr_ctx: The tracer context returned by hwt_perf_start_collector.
 *
 * Returns true on success or false otherwise.
 */
bool hwt_perf_stop_collector(struct hwt_perf_ctx *tr_ctx,
                             struct hwt_cerror *err) {
  int ret = true;

  // Turn off tracer hardware.
  if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
  }

  // Signal poll loop to end.
  if (close(tr_ctx->stop_fds[1]) == -1) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
  }
  tr_ctx->stop_fds[1] = -1;

  // Wait for poll loop to exit.
  void *thr_exit;
  if (pthread_join(tr_ctx->collector_thread, &thr_exit) != 0) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
  }
  if ((bool)thr_exit != true) {
    hwt_set_cerr(err, tr_ctx->collector_thread_err.kind,
                 tr_ctx->collector_thread_err.code);
    ret = false;
  }

  // Clean up
  if (close(tr_ctx->stop_fds[0]) == -1) {
    hwt_set_cerr(err, hwt_cerror_errno, errno);
    ret = false;
  }
  tr_ctx->stop_fds[0] = -1;

  return ret;
}

/*
 * Clean up and free a hwt_perf_ctx and its contents.
 *
 * Returns true on success or false otherwise.
 *
 * Note that anything which is cached is deliberately not freed. See
 * `thread_cached` above.
 */
bool hwt_perf_free_collector(struct hwt_perf_ctx *tr_ctx,
                             struct hwt_cerror *err) {
  int ret = true;

  if (tr_ctx->stop_fds[1] != -1) {
    // If the write end of the pipe is still open, the thread is still running.
    close(tr_ctx->stop_fds[1]); // signals thread to stop.
    if (pthread_join(tr_ctx->collector_thread, NULL) != 0) {
      hwt_set_cerr(err, hwt_cerror_errno, errno);
      ret = false;
    }
  }
  if (tr_ctx->stop_fds[0] != -1) {
    close(tr_ctx->stop_fds[0]);
  }
  if (tr_ctx != NULL) {
    free(tr_ctx);
  }
  return ret;
}
