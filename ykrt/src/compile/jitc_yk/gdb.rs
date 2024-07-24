//! This implements support for the GDB JIT Compilation Interface described here:
//! https://sourceware.org/gdb/onlinedocs/gdb/JIT-Interface.html
//!
//! This allows gdb to recognise our JITted code, so that we can have higher-level information
//! (than just raw asm) displayed when debugging traces.

use super::CompilationError;
use deku::prelude::*;
use indexmap::IndexMap;
use std::{
    ffi::{c_char, c_int, CString},
    io::Write,
    ptr,
    sync::Mutex,
};
use tempfile::NamedTempFile;

const TRACE_SYM_PREFIX: &str = "__yk_compiled_trace";

/// JITted code actions.
///
/// This is a mirror of `jit_actions_t` from <jit-reader.h>.
#[allow(dead_code, clippy::enum_variant_names)]
#[repr(u32)]
pub enum JitActionsT {
    JitNoAction = 0u32,
    JitRegisterFn = 1u32,
    JitUnregisterFn = 2u32,
}

/// An entry in gdb's linked list of JITted code objects.
///
/// This is a mirror of `struct jit_code_entry` from <jit-reader.h>.
#[repr(C)]
#[derive(Debug)]
pub struct JitCodeEntry {
    next_entry: *mut Self,
    prev_entry: *mut Self,
    symfile_addr: *const c_char,
    symfile_size: u64,
}

impl Drop for JitCodeEntry {
    fn drop(&mut self) {
        JIT_DESCRIPTOR_ACCESSOR.with_mut(|desc| {
            let prev: *mut Self = self.prev_entry;
            let next: *mut Self = self.next_entry;

            // Take the entry out of gdb's linked list.
            //
            // If there's a previous entry, update its "next" link.
            if !prev.is_null() {
                unsafe { ptr::write(ptr::addr_of_mut!((*prev).next_entry), next) };
            } else {
                // The entry being deleted is the head of the list. Update the head pointer.
                unsafe { (*desc).first_entry = next };
            }
            // If there's a "next" entry, update its "prev" link.
            if !next.is_null() {
                unsafe { ptr::write(ptr::addr_of_mut!((*next).prev_entry), prev) };
            }
        });
    }
}

/// The top-level data structure for talking to gdb.
///
/// This is a mirror of `struct jit_descriptor` from <jit-reader.h>.
#[repr(C)]
pub struct JitDescriptor {
    version: u32,
    action_flag: u32,
    relevant_entry: *mut JitCodeEntry,
    first_entry: *mut JitCodeEntry,
}

/// An instance of [JitDescriptor] with a special symbol name that gdb recognises specially.
///
/// Only [JitDescriptorAccessor] should directly access this.
#[allow(non_upper_case_globals)]
#[no_mangle]
pub static mut __jit_debug_descriptor: JitDescriptor = JitDescriptor {
    version: 1,
    action_flag: 0,
    relevant_entry: ptr::null_mut(),
    first_entry: ptr::null_mut(),
};

/// Due to the way the gdb JIT API works, we are unable to have Rust handle the synchronisation of
/// `__jit_debug_descriptor` for us.
unsafe impl Send for JitDescriptor {}
unsafe impl Sync for JitDescriptor {}
unsafe impl Send for JitCodeEntry {}
unsafe impl Sync for JitCodeEntry {}

/// Wraps `__jit_debug_descriptor`, ensuring that accesses are synchronised.
struct JitDescriptorAccessor {
    mtx: Mutex<()>,
}

impl JitDescriptorAccessor {
    fn with_mut<F>(&self, mut f: F)
    where
        F: FnMut(*mut JitDescriptor),
    {
        let lock = self.mtx.lock().unwrap();
        f(ptr::addr_of_mut!(__jit_debug_descriptor));
        drop(lock);
    }
}

static JIT_DESCRIPTOR_ACCESSOR: JitDescriptorAccessor = JitDescriptorAccessor {
    mtx: Mutex::new(()),
};

/// The JITted code registration hook.
///
/// GDB regognises calls to this function to detect when JITted code is being loaded.
#[inline(never)]
#[no_mangle]
pub extern "C" fn __jit_debug_register_code() {}

/// Describes the mapping from a line in the source file to a virtual address.
#[derive(Debug)]
#[deku_derive(DekuWrite)]
struct LineInfo {
    vaddr: usize,
    line_num: c_int,
}

/// Our custom gdb "symbol file".
///
/// Instances of this get serialised for gdb to read. Note that gdb runs in a separate address
/// space, so we can't put pointers from this address space in here and expect them to be
/// dereferencable.
#[derive(Debug)]
#[deku_derive(DekuWrite)]
struct YkSymFile {
    sym_name: CString,
    jitted_code_vaddr: usize,
    jitted_code_size: usize,
    src_path: std::ffi::CString,
    num_lineinfos: c_int,
    #[deku(count = "num_lineinfos")]
    lineinfos: Vec<LineInfo>,
}

/// The gdb context for a trace.
///
/// This contains resources that need to be kept-alive for the trace to be safely debuggable.
#[derive(Debug)]
pub(crate) struct GdbCtx {
    /// The file containing the "source code" that we show in gdb.
    #[allow(dead_code)]
    src_file: tempfile::NamedTempFile,
    /// The entry into gdb's linked list.
    ///
    /// This is a ptr to a Rust `Box` that must be reconstituted to be freed.
    #[allow(dead_code)]
    jit_code_entry: *mut JitCodeEntry,
    /// The serialised payload of the symbol file.
    #[allow(dead_code)]
    payload: Vec<u8>,
}

unsafe impl Send for GdbCtx {}
unsafe impl Sync for GdbCtx {}

impl Drop for GdbCtx {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.jit_code_entry) });
    }
}

/// Inform gdb of newly-compiled JITted code.
pub(crate) fn register_jitted_code(
    id: u64,
    jitted_code: *const u8,
    jitted_code_size: usize,
    comments: &IndexMap<usize, Vec<String>>,
) -> Result<GdbCtx, CompilationError> {
    // Write the comment lines out to a "source code file" that we want gdb to show lines from. As
    // we do this, we also build a mapping from virtual addresses to line numbers.
    let mut src_file = NamedTempFile::new()
        .map_err(|_| CompilationError::InternalError("failed to create gdb src_file".into()))?;
    let mut lineinfos = Vec::new();
    let mut line_num = 1;
    let code_vaddr = jitted_code as usize;
    for (off, lines) in comments {
        lineinfos.push(LineInfo {
            line_num,
            vaddr: code_vaddr + off,
        });
        for line in lines {
            writeln!(src_file, "{}", line).map_err(|_| {
                CompilationError::InternalError("failed to write into gdb src_file".into())
            })?;
            line_num += 1;
        }
    }
    // Ensure the source file is fully-written before gdb can read it.
    src_file
        .flush()
        .map_err(|_| CompilationError::InternalError("failed to flush gdb src_file".into()))?;

    // Build the symbol file we are going to give to gdb.
    //
    // unwrap safe: string cannot contain internal zero bytes.
    let sym_name = CString::new(format!("{}{}", TRACE_SYM_PREFIX, id)).unwrap();
    // unwrap safe: path is valid UTF-8 and  cannot contain internal zero bytes.
    let src_path = CString::new(src_file.path().to_str().unwrap()).unwrap();
    // Support for more lineinfos could be added if required.
    let num_lineinfos = c_int::try_from(lineinfos.len())
        .map_err(|_| CompilationError::LimitExceeded("too many gdb lineinfos".into()))?;
    let sym_file = Box::new(YkSymFile {
        sym_name,
        jitted_code_vaddr: jitted_code as usize, // cast safe: ptr and usize the same size.
        jitted_code_size,
        src_path,
        num_lineinfos,
        lineinfos,
    });

    // And serialise it for gdb to read.
    //
    // The serialised payload is conceptually an owned copy, so it's ok to let the original drop.
    let payload = sym_file
        .to_bytes()
        .map_err(|_| CompilationError::InternalError("failed to serialise gdb payload".into()))?;

    // Create and insert the new entry into gdb's JITted code linked list.
    let mut jit_code_entry = None;
    JIT_DESCRIPTOR_ACCESSOR.with_mut(|desc| {
        // Cache the current head of the linked list.
        let old_first = unsafe { (*desc).first_entry };

        // Make a new linked list node.
        let new_ent = Box::new(JitCodeEntry {
            next_entry: old_first,
            prev_entry: ptr::null_mut(),
            symfile_addr: payload.as_ptr() as *const i8,
            // unwrap so unlikely to fail, it's not worth considering.
            symfile_size: u64::try_from(payload.len()).unwrap(),
        });
        let new_ent_ptr = Box::into_raw(new_ent);

        // Insert the new entry at the head position of gdb's linked list.
        //
        // If the linked list is non-empty, set the current head node's previous pointer to the entry
        // we are about to prepend.
        if !old_first.is_null() {
            unsafe { ptr::write(ptr::addr_of_mut!((*old_first).prev_entry), new_ent_ptr) };
        }
        // Update the head pointer.
        unsafe { __jit_debug_descriptor.first_entry = new_ent_ptr };

        // Inform gdb that new JITted code has arrived.
        unsafe { ptr::write(ptr::addr_of_mut!((*desc).relevant_entry), new_ent_ptr) };
        unsafe {
            ptr::write(
                ptr::addr_of_mut!((*desc).action_flag),
                JitActionsT::JitRegisterFn as u32,
            )
        };
        __jit_debug_register_code();

        jit_code_entry = Some(new_ent_ptr);
    });

    Ok(GdbCtx {
        src_file,
        jit_code_entry: jit_code_entry.unwrap(), // unwrap cannot fail. Populated by closure above.
        payload,
    })
}
