// Copyright (c) 2017 King's College London
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

#define TRACE_OUTPUT    "trace.data"
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
    sem_t               tracer_init_sem;    // Tracer init sync.
    int                 loop_done_fds[2];   // Tells the poll loop to exit.
    int                 perf_fd;            // FD used to talk to the perf API.
    pid_t               target_pid;         // PID to trace.
    int                 out_fd;             // Trace written here.
    struct perf_event_mmap_page
                        *mmap_header;       // Aux buffer meta-data.
    void                *aux;               // Aux buffer itself.
    size_t              data_bufsize;       // Data buf size (in pages).
    size_t              aux_bufsize;        // Aux buf size (in pages).
};

/*
 * Passed from Rust to C to configure tracing.
 * Must stay in sync with the Rust-side.
 */
struct tracer_conf {
    pid_t       target_pid;         // PID to trace.
    const char  *trace_filename;    // Filename to store trace into.
    const char  *map_filename;      // Filename to copy linker map to.
    size_t      data_bufsize;       // Data buf size (in pages).
    size_t      aux_bufsize;        // Aux buf size (in pages).
};

// Private prototypes.
static void stash_maps(pid_t, const char *);
static void write_buf_to_disk(int, void *, __u64);
static void read_circular_buf(void *, __u64, __u64, __u64 *, int);
static void poll_loop(struct tracer_ctx *);
static void *tracer_thread(void *);

// Exposed Prototypes.
struct tracer_ctx *traceme_start_tracer(struct tracer_conf *);
void traceme_stop_tracer(struct tracer_ctx *tr_ctx);

/*
 * Save linker relocation decisions so that you can later recover the
 * instruction stream from an on-disk binary.
 */
static void
stash_maps(pid_t pid, const char *map_filename)
{
    DEBUG("saving map to %s", map_filename);

    char *cmd;
    int res = asprintf(&cmd, "cp /proc/%d/maps %s", pid, map_filename);
    if (res == -1) {
        err(EXIT_FAILURE, "asprintf");
    }

    res = system(cmd);
    if (res != 0) {
        err(EXIT_FAILURE, "system");
    }

    free(cmd);
}

/*
 * Write part of a buffer to a file descriptor.
 *
 * The implementation is a little convoluted due to friction between ssize_t
 * and __u64.
 */
static void
write_buf_to_disk(int fd, void *buf, __u64 size) {
    char *buf_c = (char *) buf;
    size_t block_size = SSIZE_MAX;
    while (size > 0) {
        if (block_size > size) {
            block_size = size;
        }
        ssize_t res = write(fd, buf, block_size);
        if (res == -1) {
            if (errno == EINTR) {
                // Write interrupted before anything written.
                continue;
            }
            err(EXIT_FAILURE, "write");
        }
        size -= res;
        buf_c += res;
    }
}

/*
 * Read data out of a circular buffer.
 */
static void
read_circular_buf(void *buf, __u64 size, __u64 head_monotonic, __u64 *tail_p, int out_fd) {
    __u64 head = head_monotonic % size; // Head must be manually wrapped.
    __u64 tail = *tail_p;

    if (tail <= head) {
        // No wrap-around
        DEBUG("read with no wrap");
        write_buf_to_disk(out_fd, buf + tail, head - tail);
    } else {
        // Wrap-around
        DEBUG("read with wrap");
        write_buf_to_disk(out_fd, buf + tail, size - tail);
        write_buf_to_disk(out_fd, buf, head);
    }

    // Update buffer tail, thus marking the space we just read as re-usable
    *tail_p = head;
}

/*
 * Take trace data out of the AUX buffer.
 */
static void
poll_loop(struct tracer_ctx *tr_ctx)
{
    struct perf_event_mmap_page *mmap_header = tr_ctx->mmap_header;
    int n_events = 0;
    size_t num_wakes = 0;
    struct pollfd pfds[2] = {
        {tr_ctx->perf_fd,           POLLIN | POLLHUP,   0},
        {tr_ctx->loop_done_fds[0],  POLLHUP,            0}
    };

    while (1) {
        n_events = poll(pfds, 2, INFTIM);
        if (n_events == -1) {
            err(EXIT_FAILURE, "poll");
        }

        if ((pfds[0].revents & POLLIN) || (pfds[1].revents & POLLHUP)) {
            // See <linux/perf_event.h> for why we need the asm block.
            __u64 head = mmap_header->aux_head;
            asm volatile ("" : : : "memory");

            // We were awoken to read out trace info, or we tracing stopped.
            num_wakes++;
            DEBUG("wake");
            DEBUG("aux_head=  0x%010llu", head);
            DEBUG("aux_tail=  0x%010llu", mmap_header->aux_tail);
            DEBUG("aux_offset=0x%010llu", mmap_header->aux_offset);
            DEBUG("aux_size=  0x%010llu", mmap_header->aux_size);

            read_circular_buf(tr_ctx->aux, mmap_header->aux_size,
                head, &mmap_header->aux_tail, tr_ctx->out_fd);

            if (pfds[1].revents & POLLHUP) {
                break;
            }
        }

        if (pfds[0].revents & POLLHUP) {
            break;
        }
    }

    DEBUG("poll loop exit: awoke %zu times", num_wakes);
}

/*
 * Set up Intel PT buffers and start a poll() loop for reading out the trace.
 */
static void *
tracer_thread(void *arg)
{
    DEBUG("tracer init");

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.size = sizeof(struct perf_event_attr);

    // Get the perf "type" for Intel PT.
    FILE *pt_type_file = fopen(SYSFS_PT_TYPE, "r");
    if (pt_type_file == NULL) {
        err(EXIT_FAILURE, "fopen");
    }
    char pt_type_str[MAX_PT_TYPE_STR];
    if (fgets(pt_type_str, sizeof(pt_type_str), pt_type_file) == NULL) {
        err(EXIT_FAILURE, "fgets");
    }
    attr.type = atoi(pt_type_str);
    if (fclose(pt_type_file) == EOF) {
        err(EXIT_FAILURE, "fclose");
    }

    // Exclude the kernel.
    attr.exclude_kernel = 1;

    // Exclude the hyper-visor.
    attr.exclude_hv = 1;

    // Start disabled.
    attr.disabled = 1;

    // No skid.
    attr.precise_ip = 3;

    // Acquire file descriptor through which to talk to Intel PT.
    struct tracer_ctx *tr_ctx = (struct tracer_ctx *) arg;
    tr_ctx->perf_fd = syscall(SYS_perf_event_open, &attr,
        tr_ctx->target_pid, -1, -1, 0);
    if (tr_ctx->perf_fd == -1) {
        err(EXIT_FAILURE, "syscall");
    }

    /*
     * Allocate buffers.
     *
     * Data buffer is preceded by one management page, hence `1 + data_bufsize'.
     */
    int page_size = getpagesize();
    void *base = mmap(NULL, (1 + tr_ctx->data_bufsize) * page_size, PROT_WRITE,
        MAP_SHARED, tr_ctx->perf_fd, 0);
    if (base == MAP_FAILED) {
        err(EXIT_FAILURE, "mmap");
    }

    struct perf_event_mmap_page *header = base;
    void *data = base + header->data_offset;
    (void) data; // XXX We will need this in the future to detect packet loss.
    header->aux_offset = header->data_offset + header->data_size;
    header->aux_size = tr_ctx->aux_bufsize * page_size;
    tr_ctx->mmap_header = header;

    // AUX mapped R/W so as to have a saturating ring buffer.
    void *aux = mmap(NULL, header->aux_size, PROT_READ | PROT_WRITE,
        MAP_SHARED, tr_ctx->perf_fd, header->aux_offset);
    if (aux == MAP_FAILED) {
        err(EXIT_FAILURE, "mmap2");
    }
    tr_ctx->aux = aux;

    // Resume the interpreter loop.
    DEBUG("resume main thread");
    sem_post(&tr_ctx->tracer_init_sem);

    // Start reading out of the aux buffer.
    poll_loop(tr_ctx);

    DEBUG("tracer thread exit");
    pthread_exit(NULL);
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
 */
struct tracer_ctx *
traceme_start_tracer(struct tracer_conf *tr_conf)
{
    DEBUG("target_pid=%d, trace_filename=%s, map_filename=%s, "
        "data_bufsize=%zd, aux_bufsize=%zd", tr_conf->target_pid,
        tr_conf->trace_filename, tr_conf->map_filename,
        tr_conf->data_bufsize, tr_conf->aux_bufsize);

    /*
     * Dump process map to disk so that we can relate virtual addresses to the
     * on-disk instruction stream.
     */
    stash_maps(tr_conf->target_pid, tr_conf->map_filename);

    // Allocate and initialise tracer context.
    struct tracer_ctx *tr_ctx = malloc(sizeof(*tr_ctx));
    if (tr_ctx == NULL) {
        err(EXIT_FAILURE, "malloc");
    }
    tr_ctx->target_pid = tr_conf->target_pid;
    tr_ctx->data_bufsize = tr_conf->data_bufsize;
    tr_ctx->aux_bufsize = tr_conf->aux_bufsize;

    // Open the trace output file.
    tr_ctx->out_fd = open(tr_conf->trace_filename, O_WRONLY | O_CREAT);
    if (tr_ctx->out_fd < 0) {
        err(EXIT_FAILURE, "open");
    }

    // Pipe used for the VM to flag the user loop is complete.
    if (pipe(tr_ctx->loop_done_fds) != 0) {
        err(EXIT_FAILURE, "pipe");
    }

    // Use a semaphore to wait for the tracer to be ready.
    int rc = sem_init(&tr_ctx->tracer_init_sem, 0, 0);
    if (rc == -1) {
        err(EXIT_FAILURE, "sem_init");
    }

    // Spawn a thread to deal with copying out of the PT aux buffer.
    rc = pthread_create(&tr_ctx->tracer_thread, NULL, tracer_thread,
        (void *) tr_ctx);
    if (rc) {
        err(EXIT_FAILURE, "pthread_create");
    }

    DEBUG("wait for tracer to init");
    sem_wait(&tr_ctx->tracer_init_sem);

    // Turn on tracing hardware.
    if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_ENABLE, 0) < 0)
        err(EXIT_FAILURE, "ioctl to start tracer");

    DEBUG("resume");
    return tr_ctx;
}

/*
 * Turn off the tracer.
 *
 * Arguments:
 *   tr_ctx: The tracer context returned by traceme_start_tracer.
 */
void
traceme_stop_tracer(struct tracer_ctx *tr_ctx) {
    DEBUG("stopping tracer");

    // Turn off tracer hardware.
    if (ioctl(tr_ctx->perf_fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
        err(EXIT_FAILURE, "ioctl");
    }

    // Signal poll loop to end.
    if (close(tr_ctx->loop_done_fds[1]) == -1) {
        err(EXIT_FAILURE, "close");
    }

    // Wait for poll loop to exit.
    DEBUG("wait for trace thread to exit");
    if (pthread_join(tr_ctx->tracer_thread, NULL) != 0) {
        err(EXIT_FAILURE, "pthread_join");
    }

    // Clean up
    if (close(tr_ctx->perf_fd) != 0) {
        err(EXIT_FAILURE, "close");
    }
    if (close(tr_ctx->out_fd) != 0) {
        err(EXIT_FAILURE, "close");
    }
    free(tr_ctx);

    DEBUG("tracing complete");
}
