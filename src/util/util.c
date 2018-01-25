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

#include <sys/syscall.h>
#include <link.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>

// Private prototypes.
static int dl_iterate_phdr_cb(struct dl_phdr_info *, size_t, void *);

// Exposed prototypes.
int hwtracer_exec_base(uintptr_t *);
pid_t hwtracer_linux_gettid(void);

/*
 * Search program headers for the relocated start address of the current
 * program's executable code.
 *
 * Returns 1 if the address is found, or 0 otherwise. If found, the address is
 * written into `*addr`.
 */
int
hwtracer_exec_base(uintptr_t *addr)
{
    return dl_iterate_phdr(dl_iterate_phdr_cb, addr);
}

/*
 * The callback for `dl_iterate_phdr`, called once for each program header.
 *
 * We are looking for the start address of the program's executable
 * instructions.
 *
 * We must return 1 to stop iteration (i.e. we found what we needed), or 0 to
 * continue. See dl_iterate_phdr(3) for details.
 */
static int
dl_iterate_phdr_cb(struct dl_phdr_info *info, size_t size, void *data)
{
#ifdef __OpenBSD__
    Elf_Phdr phdr;
    Elf_Half i;
#else
    ElfW(Phdr) phdr;
    ElfW(Half) i;
#endif
    (void) size;

    // Check if this is the entry for the binary itself (not a shared object).
#if defined(__linux__)
    // On Linux the main binary has no name.
    if (info->dlpi_name[0] != '\0') {
        return 0;
    }
#elif defined(__OpenBSD__)
    // On OpenBSD the name is that of the binary.
    if (strcmp(basename(info->dlpi_name), getprogname()) != 0) {
        return 0;
    }
#else
#error "dl_iterate_phdr_cb(): unknown platform"
#endif

    // Now we have the header for the binary, search its segments for the code.
    for (i = 0; i < info->dlpi_phnum; i++) {
        phdr = info->dlpi_phdr[i];
        if ((phdr.p_type == PT_LOAD) && (phdr.p_flags == (PF_R | PF_X))) {
            *((uintptr_t *) data) = info->dlpi_addr + phdr.p_vaddr;
            return 1;
        }
    }
    return 0;
}

/*
 * Get the thread ID of the current thread.
 *
 * This is a Linux specific notion. The pid_t type is overloaded to also refer
 * to individual threads.
 *
 * At the time of writing, there is no glibc stub for this.
 */
#ifdef __linux__
pid_t
hwtracer_linux_gettid(void)
{
    return syscall(__NR_gettid);
}
#endif
