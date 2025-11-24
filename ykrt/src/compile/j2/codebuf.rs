//! Support for code buffers.
//!
//! When creating a code buffer, one starts with [CodeBufInProgress]: this allocates memory, but
//! does not require anything meaningful to have been written into it. When code has actually been
//! generated, [CodeBufInProgress::into_execodebuf] turns a [CodeBufInProgress] into a
//! [ExeCodeBuf]. An [ExeCodeBuf] is complete and executable as-is: although it might be patched,
//! the quantity of memory it contains will not change.

use crate::compile::j2::SyncSafePtr;
use libc::{
    MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect, munmap,
};
use std::ffi::c_void;

/// A code buffer that does backing memory allocated but no actual code stored in it.
#[derive(Debug)]
pub(super) struct CodeBufInProgress {
    /// A pointer to the beginning of the `mmap`ed buffer.
    buf: *mut u8,
    /// How many bytes have we allocated to the buffer?
    len: usize,
}

impl CodeBufInProgress {
    /// Create a new code buffer with a size at least `len` bytes big.
    pub fn new(len: usize) -> Self {
        let len = len.next_multiple_of(page_size::get());
        let buf = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_ANON | MAP_PRIVATE,
                -1,
                0,
            )
        };
        if buf == MAP_FAILED {
            todo!();
        }

        Self {
            buf: buf as *mut u8,
            len,
        }
    }

    /// Get a raw pointer to the start of the executable code buffer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.buf
    }

    /// Return the size of this buffer in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Copy `other_len` bytes from `other` into `self` and produce an [ExeCodeBuf].
    pub unsafe fn into_execodebuf(self, other: *const u8, other_len: usize) -> ExeCodeBuf {
        assert!(other_len <= self.len);
        unsafe {
            self.buf.copy_from_nonoverlapping(other, other_len);
        }

        // If there's more than a page's worth of unused space in the buffer, return it to the OS.
        let unused = self.len - other_len.next_multiple_of(page_size::get());
        if unused > 0 {
            let rtn =
                unsafe { munmap(self.buf.byte_add(self.len - unused) as *mut c_void, unused) };
            if rtn != 0 {
                todo!();
            }
        }

        let used = other_len.next_multiple_of(page_size::get());
        // FIXME: should be `WRITE`.
        unsafe {
            mprotect(
                self.buf as *mut c_void,
                used,
                PROT_EXEC | PROT_READ | PROT_WRITE,
            );
        }
        ExeCodeBuf {
            buf: SyncSafePtr(self.buf),
            len: used,
        }
    }
}

/// An executable code buffer.
#[derive(Debug)]
pub(super) struct ExeCodeBuf {
    /// A pointer to the beginning of the `mmap`ed buffer.
    buf: SyncSafePtr<*mut u8>,
    /// How many bytes have we allocated to the buffer?
    len: usize,
}

impl ExeCodeBuf {
    /// Get a raw pointer to the start of the executable code buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.0
    }

    /// Return the size of this buffer in bytes.
    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.len
    }
}
