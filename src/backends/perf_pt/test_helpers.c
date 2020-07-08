// Copyright (c) 2018 King's College London
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

#include <link.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <err.h>
#include <elf.h>

#include "perf_pt_private.h"

// Arguments for perf_pt_append_self_ptxed_raw_args().
struct append_args {
    int vdso_fd;
    char *vdso_filename;
    void *rs_args_vec;
};

// Public prototypes.
bool perf_pt_append_self_ptxed_raw_args(struct append_args *);

// Private prototypes.
static int append_self_ptxed_raw_args_cb(struct dl_phdr_info *, size_t, void *);

// Rust functions.
extern void push_ptxed_arg(void *, const char *);

/*
 * Walks the program headers pushing ptxed `--raw` arguments for the current
 * process into the supplied (pointer to a) Rust `AppendSelfPtxedArgs`
 * instance.
 */
bool
perf_pt_append_self_ptxed_raw_args(struct append_args *rs_args)
{
    return dl_iterate_phdr(append_self_ptxed_raw_args_cb, rs_args) == 0;
}

/*
 * Callback for perf_pt_append_self_ptxed_raw_args.
 *
 * `data` is a pointer to a Rust `AppendSelfPtxedArgs` instance.
 */
static int
append_self_ptxed_raw_args_cb(struct dl_phdr_info *info, size_t size, void *data)
{
    ElfW(Phdr) phdr;
    ElfW(Half) i;

    (void) size;
    struct append_args *args = data;

    const char *filename = info->dlpi_name;
    bool vdso = false;
    if (!*filename) {
        filename = program_invocation_name;
    } else {
        vdso = strcmp(filename, VDSO_NAME) == 0;
    }

    for (i = 0; i < info->dlpi_phnum; i++) {
        phdr = info->dlpi_phdr[i];
        if ((phdr.p_type != PT_LOAD) || (!(phdr.p_flags & PF_X))) {
            continue;
        }

        uint64_t vaddr = info->dlpi_addr + phdr.p_vaddr;
        uint64_t offset;

        if (vdso) {
            // Since this is testing code, we don't worry too much about the
            // exact error, but we do have to pass down an error struct.
            struct perf_pt_cerror err;
            if (!dump_vdso(args->vdso_fd, vaddr, phdr.p_memsz, &err)) {
                return 1;
            }
            filename = args->vdso_filename;
            offset = 0;
        } else {
            offset = phdr.p_offset;
        }

        // Push arguments of the form: --raw <filename>:<start>-<end>:<vaddr>
        char *raw_arg = NULL;
        int rv = asprintf(&raw_arg, "%s:0x%" PRIx64 "-0x%" PRIx64 ":0x%" PRIx64,
                          filename, offset, phdr.p_offset + phdr.p_filesz, vaddr);
        if (rv < 0) {
            return 1;
        }

        // Call to Rust to do the stores.
        push_ptxed_arg(args->rs_args_vec, "--raw");
        push_ptxed_arg(args->rs_args_vec, raw_arg);

        free(raw_arg); // Safe because push_ptxed_arg() copies.
    }

    return 0;
}
