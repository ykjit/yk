// Copyright (c) 2017-2018 King's College London
// created by the Software Development Team <http://soft-dev.org/>
//
// The Universal Permissive License (UPL), Version 1.0
//
// Subject to the condition set forth below, permission is hereby granted to any
// person obtaining a copy of this software, associated documentation and/or
// data (collectively the "Software"), free of charge and under any and all
// copyright rights in the Software, and any and all patent rights owned or
// freely licensable by each licensor hereunder covering either (i) the
// unmodified Software as contributed to or provided by such licensor, or (ii)
// the Larger Works (as defined below), to deal in both
//
// (a) the Software, and
// (b) any piece of software and/or hardware listed in the lrgrwrks.txt file
// if one is included with the Software (each a "Larger Work" to which the Software
// is contributed by such licensors),
//
// without restriction, including without limitation the rights to copy, create
// derivative works of, display, perform, and distribute the Software and make,
// use, sell, offer for sale, import, export, have made, and have sold the
// Software and the Larger Work(s), and to sublicense the foregoing rights on
// either these or other terms.
//
// This license is subject to the following condition: The above copyright
// notice and either this complete permission notice or at a minimum a reference
// to the UPL must be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <err.h>
#include <fcntl.h>
#include <syscall.h>
#include <sys/mman.h>
#include <poll.h>
#include <inttypes.h>
#include <errno.h>
#include <pthread.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <semaphore.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>

#define SYSFS_PT_TYPE   "/sys/bus/event_source/devices/intel_pt/type"
#define MAX_PT_TYPE_STR 8

#ifndef INFTIM
#define INFTIM -1
#endif

#define DEBUG(x...)                       \
    do {                                  \
        fprintf(stderr, "%s:%d [%s]: ",   \
           __FILE__, __LINE__, __func__); \
        fprintf(stderr, x);               \
        fprintf(stderr, "\n");            \
    } while (0)

/*
 * Stores all information about the tracer.
 * Exposed to Rust only as an opaque pointer.
 */
struct tracer_ctx {
    pthread_t           tracer_thread;      // Tracer thread handle.
    int                 stop_fd_wr;         // Close to halt trace thread.
    int                 stop_fd_rd;         // Read end of stop_fd_wr pipe.
    int                 perf_fd;            // FD used to talk to the perf API.
    void                *trace_buf;         // The trace copied from the AUX buffer.
    __u64               trace_len;          // The length of the trace (in bytes).
};

/*
 * Passed from Rust to C to configure tracing.
 * Must stay in sync with the Rust-side.
 */
struct tracer_conf {
    pid_t       target_tid;         // Thread ID to trace.
    size_t      data_bufsize;       // Data buf size (in pages).
    size_t      aux_bufsize;        // AUX buf size (in pages).
};

/*
 * Stuff used in the tracer thread
 */
struct tracer_thread_args {
    int                 perf_fd;            // Perf notification fd.
    int                 stop_fd_rd;         // Polled for "stop" event.
    sem_t               *tracer_init_sem;   // Tracer init sync.
    size_t              data_bufsize;       // Data buf size (in pages).
    size_t              aux_bufsize;        // AUX buf size (in pages).
    void                **trace_buf;        // Pointer to the buffer to copy the trace into.
    __u64               trace_bufsize;      // Initial capacity of the trace buffer (in bytes).
    __u64               *trace_len;         // Pointer to the trace length (in bytes).
};


// Private prototypes.
static bool read_aux(void *, __u64, __u64, __u64 *, void **, __u64, __u64 *);
static bool poll_loop(int, int, struct perf_event_mmap_page *, void *, void **,
                      __u64, __u64 *);
static void *tracer_thread(void *);
static int open_perf(pid_t target_tid);

// Exposed Prototypes.
struct tracer_ctx *perf_pt_start_tracer(struct tracer_conf *);
int perf_pt_stop_tracer(struct tracer_ctx *tr_ctx, uint8_t **, __u64 *);
void *perf_pt_trace_buf(struct tracer_ctx *tr_ctx, uint32_t *len);

/*
 * Read data out of the AUX buffer.
 *
 * Reads `size` bytes from `aux_buf` into `trace_buf`, updating `*trace_len`.
 *
 * Returns true on success and false otherwise.
 */
static bool
read_aux(void *aux_buf, __u64 size, __u64 head_monotonic, __u64 *tail_p,
         void **trace_buf, __u64 trace_bufsize, __u64 *trace_len) {
    __u64 head = head_monotonic % size; // Head must be manually wrapped.
    __u64 tail = *tail_p;

    // First check we won't overflow the (destination) trace buffer.
    __u64 new_data_size;
    if (tail <= head) {
        // No wrap-around.
        new_data_size = head - tail;
    } else {
        // Wrap-around.
        new_data_size = (size - tail) + head;
    }

    // For now we simply crash out if we exhaust the buffer.
    assert(*trace_len + new_data_size <= trace_bufsize);

    // Finally copy the AUX buffer into the trace buffer.
    if (tail <= head) {
        memcpy(*trace_buf, aux_buf + tail, head - tail);
    } else {
        memcpy(*trace_buf, aux_buf + tail, size - tail);
        memcpy(*trace_buf, aux_buf, head);
    }
    *trace_len += new_data_size;

    // Update buffer tail, thus marking the space we just read as re-usable.
    *tail_p = head;
    return true;
}

/*
 * Take trace data out of the AUX buffer.
 *
 * Returns true on success and false otherwise.
 */
static bool
poll_loop(int perf_fd, int stop_fd, struct perf_event_mmap_page *mmap_hdr,
          void *aux, void **trace_buf, __u64 trace_bufsize, __u64 *trace_len)
{
    int n_events = 0;
    bool ret = true;
    struct pollfd pfds[2] = {
        {perf_fd,   POLLIN | POLLHUP,   0},
        {stop_fd,   POLLHUP,            0}
    };

    while (1) {
        n_events = poll(pfds, 2, INFTIM);
        if (n_events == -1) {
            ret = false;
            goto done;
        }

        if ((pfds[0].revents & POLLIN) || (pfds[1].revents & POLLHUP)) {
            // See <linux/perf_event.h> for why we need the asm block.
            __u64 head = mmap_hdr->aux_head;
            asm volatile ("" : : : "memory");

            if (!read_aux(aux, mmap_hdr->aux_size, head, &mmap_hdr->aux_tail,
                          trace_buf, trace_bufsize, trace_len)) {
                ret = false;
                goto done;
            }

            if (pfds[1].revents & POLLHUP) {
                break;
            }
        }

        if (pfds[0].revents & POLLHUP) {
            break;
        }
    }

done:
    return ret;
}

/*
 * Opens the perf file descriptor and returns it.
 *
 * Returns a file descriptor, or -1 on error.
 */
static int
open_perf(pid_t target_tid) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.size = sizeof(struct perf_event_attr);

    int ret = -1;

    // Get the perf "type" for Intel PT.
    FILE *pt_type_file = fopen(SYSFS_PT_TYPE, "r");
    if (pt_type_file == NULL) {
        ret = -1;
        goto clean;
    }
    char pt_type_str[MAX_PT_TYPE_STR];
    if (fgets(pt_type_str, sizeof(pt_type_str), pt_type_file) == NULL) {
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

    // Acquire file descriptor through which to talk to Intel PT.
    ret = syscall(SYS_perf_event_open, &attr, target_tid, -1, -1, 0);

clean:
    if ((pt_type_file != NULL) && (fclose(pt_type_file) == -1)) {
        ret = -1;
    }

    return ret;
}

/*
 * Set up Intel PT buffers and start a poll() loop for reading out the trace.
 *
 * Returns true on success and false otherwise.
 */
static void *
tracer_thread(void *arg)
{
    int page_size = getpagesize();
    struct tracer_thread_args *thr_args = (struct tracer_thread_args *) arg;
    int base_size = (1 + thr_args->data_bufsize) * page_size;
    int sem_posted = 0;
    void *base = NULL;
    void *aux = NULL;
    struct perf_event_mmap_page *header = NULL;
    bool ret = true;

    // Allocate buffers.
    // Data buffer is preceded by one management page, hence `1 + data_bufsize'.
    base = mmap(NULL, base_size, PROT_WRITE, MAP_SHARED, thr_args->perf_fd, 0);
    if (base == MAP_FAILED) {
        ret = false;
        goto clean;
    }

    header = base;
    void *data = base + header->data_offset;
    (void) data; // XXX We will need this in the future to detect packet loss.
    header->aux_offset = header->data_offset + header->data_size;
    header->aux_size = thr_args->aux_bufsize * page_size;

    // AUX mapped R/W so as to have a saturating ring buffer.
    aux = mmap(NULL, header->aux_size, PROT_READ | PROT_WRITE,
        MAP_SHARED, thr_args->perf_fd, header->aux_offset);
    if (aux == MAP_FAILED) {
        ret = false;
        goto clean;
    }

    // Copy arguments for the poll loop, as when we resume the parent thread,
    // `thr_args', which is on the parent thread's stack, will become unusable.
    int perf_fd = thr_args->perf_fd;
    int stop_fd_rd = thr_args->stop_fd_rd;
    void **trace_buf = thr_args->trace_buf;
    __u64 trace_bufsize = thr_args->trace_bufsize;
    __u64 *trace_len = thr_args->trace_len;

    // Resume the interpreter loop.
    if (sem_post(thr_args->tracer_init_sem) != 0) {
        ret = false;
        goto clean;
    }
    sem_posted = 1;

    // Start reading out of the AUX buffer.
    if (!poll_loop(perf_fd, stop_fd_rd, header, aux, trace_buf,
                   trace_bufsize, trace_len)) {
        ret = false;
        goto clean;
    }

clean:
    if (aux) {
        if (munmap(aux, header->aux_size) == -1) {
            ret = false;
        }
    }
    if (base) {
        if (munmap(base, base_size) == -1) {
            ret = false;
        }
    }
    if (!sem_posted) {
        assert(!ret);
        sem_post(thr_args->tracer_init_sem);
    }

    return (void *) ret;
}

/*
 * --------------------------------------
 * Functions exposed to the outside world
 * --------------------------------------
 */

/*
 * Turn on Intel PT.
 *
 * Arguments:
 *   tr_conf: Tracer configuration.
 *
 * Returns a pointer to a tracer context, or NULL on error.
 */
struct tracer_ctx *
perf_pt_start_tracer(struct tracer_conf *tr_conf)
{
    bool failing = false;
    struct tracer_ctx *tr_ctx = NULL;
    int perf_fd = -1;
    int clean_sem = 0, clean_thread = 0;
    struct tracer_ctx *ret = NULL;

    // Allocate and initialise tracer context.
    tr_ctx = malloc(sizeof(*tr_ctx));
    if (tr_ctx == NULL) {
        goto clean;
    }
    tr_ctx->trace_buf = NULL;
    tr_ctx->trace_len = 0;

    // Get the perf fd
    perf_fd = open_perf(tr_conf->target_tid);
    if (perf_fd == -1) {
        failing = true;
        goto clean;
    }
    tr_ctx->perf_fd = perf_fd;

    // Pipe used for the VM to flag the user loop is complete.
    int stop_fds[2] = {-1. -1};
    if (pipe(stop_fds) != 0) {
        failing = true;
        goto clean;
    }
    tr_ctx->stop_fd_rd = stop_fds[0];
    tr_ctx->stop_fd_wr = stop_fds[1];

    // Allocate buffer to copy the trace into (currently the same size as the AUX buffer).
    __u64 trace_bufsize = tr_conf->aux_bufsize * getpagesize();
    tr_ctx->trace_buf = malloc(trace_bufsize);
    if (tr_ctx->trace_buf == NULL) {
        failing = true;
        goto clean;
    }

    // Use a semaphore to wait for the tracer to be ready.
    sem_t tracer_init_sem;
    if (sem_init(&tracer_init_sem, 0, 0) == -1) {
        failing = true;
        goto clean;
    }
    clean_sem = 1;

    // Build the arguments struct for the tracer thread.
    struct tracer_thread_args thr_args = {
        perf_fd,
        stop_fds[0],
        &tracer_init_sem,
        tr_conf->data_bufsize,
        tr_conf->aux_bufsize,
        &tr_ctx->trace_buf,
        trace_bufsize,
        &tr_ctx->trace_len,
    };

    // Spawn a thread to deal with copying out of the PT AUX buffer.
    int rc = pthread_create(&tr_ctx->tracer_thread, NULL, tracer_thread, &thr_args);
    if (rc) {
        failing = true;
        goto clean;
    }

    // Wait for the tracer to initialise, and check it didn't fail.
    rc = -1;
    while (rc == -1) {
        rc = sem_wait(&tracer_init_sem);
        if ((rc == -1) && (errno != EINTR)) {
            failing = true;
            goto clean;
        }
    }

    // Turn on tracing hardware.
    if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
        failing = true;
        goto clean;
    }

    ret = tr_ctx;
clean:
    if (failing) {
        ret = NULL;
        // Child thread must be stopped before closing the shared perf_fd.
        if (clean_thread) {
            close(stop_fds[1]);  // signals thread
            pthread_join(tr_ctx->tracer_thread, NULL);
            close(stop_fds[0]);
        }
        if (perf_fd) {
            close(perf_fd);
        }
        if (tr_ctx->trace_buf != NULL) {
            free(tr_ctx->trace_buf);
            tr_ctx->trace_buf = NULL;
        }
        if (tr_ctx != NULL) {
            free(tr_ctx);
        }
    }

    if (clean_sem) {
        if (sem_destroy(&tracer_init_sem) == -1) {
            ret = NULL;
        }
    }

    return ret;
}

/*
 * Turn off the tracer.
 *
 * Arguments:
 *   tr_ctx: The tracer context returned by perf_pt_start_tracer.
 *
 * Returns 0 on success and -1 on failure.
 */
int
perf_pt_stop_tracer(struct tracer_ctx *tr_ctx, uint8_t **buf, __u64 *len) {
    int ret = 0;

    // Turn off tracer hardware.
    if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
        ret = -1;
    }

    // Signal poll loop to end.
    if (close(tr_ctx->stop_fd_wr) == -1) {
        ret = -1;
    }

    // Wait for poll loop to exit.
    void *thr_exit;
    if (pthread_join(tr_ctx->tracer_thread, &thr_exit) != 0) {
        ret = -1;
    }
    if ((bool) thr_exit != true) {
        ret = -1;
    }

    // Clean up
    if (close(tr_ctx->stop_fd_rd) == -1) {
        ret = -1;
    }
    tr_ctx->stop_fd_rd = -1;

    if (close(tr_ctx->perf_fd) != 0) {
        ret = -1;
    }
    tr_ctx->perf_fd = -1;

    if (ret != -1) {
        *buf = tr_ctx->trace_buf;
        *len = tr_ctx->trace_len;
    } else {
        if (tr_ctx->trace_buf != NULL) {
            free(tr_ctx->trace_buf);
        }
        *buf = NULL;
        *len = 0;
    }
    return ret;
}
