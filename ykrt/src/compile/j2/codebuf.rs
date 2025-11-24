//! An executable code buffer.

use libc::{MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, munmap};
use std::ffi::c_void;

/// An executable code buffer.
#[derive(Debug)]
pub(super) struct CodeBuf {
    /// Where will this trace be stored in memory?
    buf: *mut u8,
    /// How many bytes have we allocated to the buffer?
    len: usize,
}

impl CodeBuf {
    /// Create a new code buffer with a size at least `len` bytes big.
    pub fn new(len: usize) -> Self {
        let len = len.next_multiple_of(page_size::get());
        let buf = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE | PROT_EXEC,
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

    /// Copy `other_len` bytes from `other` into `self`.
    pub unsafe fn copy_into(&mut self, other: *const u8, other_len: usize) {
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
    }
}
