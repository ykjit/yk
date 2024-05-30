// A gdb plugin for debugging Yk JITted traces.
//
// To use this, put in your ~/.gdbinit:
//   jit-reader-load /path/to/yk/target/yk_gdb_plugin.so

#define _GNU_SOURCE

#include <assert.h>
#include <gdb/jit-reader.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gdb requires JIT plugins to be GPL licensed.
//
// Without this declaration the plugin will build, but refuse to load.
//
// As such, this individual C file is licensed under the GPL license.
GDB_DECLARE_GPL_COMPATIBLE_READER

enum gdb_status read_debug_info_cb(struct gdb_reader_funcs *self,
                                   struct gdb_symbol_callbacks *cb,
                                   void *memory, long memory_sz) {
  (void)self;
  (void)memory_sz;
  (void)cb;

  // Treat the payload as a byte array.
  char *payload = memory;

  // Start deserialising the payload.
  //
  // To avoid pointer aliasing issues, we memcpy anything that isn't
  // `char *`-typed (instead of creating aliased pointers).
  //
  // Read the symbol name.
  char *sym_name = payload;
  payload += strlen(sym_name) + 1; // +1 for null terminator.

  // Read out the jitted code virtual address.
  unsigned char *jitted_code_vaddr;
  memcpy(&jitted_code_vaddr, payload, sizeof(jitted_code_vaddr));
  payload += sizeof(jitted_code_vaddr);

  // Read out the jitted code size.
  size_t jitted_code_size;
  memcpy(&jitted_code_size, payload, sizeof(jitted_code_size));
  payload += sizeof(jitted_code_size);

  // Next is the source file path (as a null-terminated string).
  char *src_path = payload;
  payload += strlen(src_path) + 1; // +1 for null terminator.

  // Tell gdb about where the code lives and where the "source code" is.
  struct gdb_object *obj = cb->object_open(cb);
  struct gdb_symtab *symtab = cb->symtab_open(cb, obj, src_path);

  // Tell gdb the extent of the JITted code.
  //
  // FIXME: why can't we break on the symbol name we tell gdb here?
  //
  // NOTE: this returns a `struct gdb_block`. At the time of writing there's a
  // comment next to `gdb_block_open()` in <jit-reader.h> that says the return
  // value is unused, but must not be freed by the caller.
  unsigned char *jitted_code_end_vaddr = jitted_code_vaddr + jitted_code_size;
  assert(sizeof(GDB_CORE_ADDR) == sizeof(uintptr_t));
  cb->block_open(cb, symtab, NULL, (GDB_CORE_ADDR)jitted_code_vaddr,
                 (GDB_CORE_ADDR)jitted_code_end_vaddr, sym_name);

  // Read out the number of lineinfo pairs to expect.
  int num_lineinfos;
  memcpy(&num_lineinfos, payload, sizeof(num_lineinfos));
  payload += sizeof(num_lineinfos);

  // Read out the lineinfo records.
  //
  // Note that for gdb to reliably use the lineinfo, it appears the records
  // need to be ordered by line number, ascending.
  struct gdb_line_mapping *l_infos =
      calloc(num_lineinfos, sizeof(struct gdb_line_mapping));
  for (int i = 0; i < num_lineinfos; i++) {
    // Read out the virtual address.
    memcpy(&(l_infos[i].pc), payload, sizeof(l_infos[i].pc));
    payload += sizeof(l_infos[i].pc);

    // Read out the line number.
    memcpy(&(l_infos[i].line), payload, sizeof(l_infos[i].line));
    payload += sizeof(l_infos[i].line);
  }

  // Tell gdb about the lineinfo.
  cb->line_mapping_add(cb, symtab, num_lineinfos, l_infos);

  free(l_infos);

  // Tells gdb that no more blocks will be added to this symtab.
  cb->symtab_close(cb, symtab);
  // Commit everything to gdb's internal data structures.
  cb->object_close(cb, obj);

  return GDB_SUCCESS;
}

// Required by the plugin API, but unused by us.
void destory_reader_cb(struct gdb_reader_funcs *self) { (void)self; }

// Required by the plugin API, but unused by us.
enum gdb_status unwind_frame_cb(struct gdb_reader_funcs *self,
                                struct gdb_unwind_callbacks *cb) {
  (void)self;
  (void)cb;
  return GDB_FAIL;
}

// Required by the plugin API, but unused by us.
struct gdb_frame_id get_frame_id_cb(struct gdb_reader_funcs *self,
                                    struct gdb_unwind_callbacks *c) {
  (void)self;
  (void)c;
  struct gdb_frame_id ret = {0, 0};
  return ret;
}

struct gdb_reader_funcs reader_funcs = {
    .reader_version = GDB_READER_INTERFACE_VERSION,
    .priv_data = NULL,
    .read = read_debug_info_cb,
    .unwind = unwind_frame_cb,
    .get_frame_id = get_frame_id_cb,
    .destroy = destory_reader_cb,
};

struct gdb_reader_funcs *gdb_init_reader(void) {
  printf("Yk JIT support loaded.\n");
  return &reader_funcs;
}
