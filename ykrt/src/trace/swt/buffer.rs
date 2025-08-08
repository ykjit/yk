//! # Temporary Buffer Management for Live Variable Transfers
//!
//! This module provides temporary storage for live variables during control point
//! transitions. It serves as an intermediate staging area for variables that need to be moved
//! between source and destination locations.
//!
//! ## Purpose
//!
//! During control point transitions, variables stored in stack locations (`Indirect` and `Direct`)
//! need temporary storage because source and destination locations might have conflicting values
//! that can override each other.
//! Example of live variables:
//!
//!  source: Indirect(6, -8, 8), Indirect(6, -16, 8)
//!  destination: Indirect(6, -16, 8), Indirect(6, -8, 8)
//!
//!  We need to copy rbp-8 to rbp-16 and rbp-16 to rbp-8
//!  Here we cannot copy the values from source to destination sequentially because
//!  source values will be overwritten by other source values.
//!  A temporary buffer solves this by serving as temporary storage for the values from
//!  source and then copying them to destination.
//!
//! ## Temporary Buffers
//!
//! The module implements a thread-local buffer pool with two separate buffers:
//! - **OPT_BUFFER**: Used for transitions from optimised variants.
//! - **UNOPT_BUFFER**: Used for transitions from unoptimised variants.
//!
//! ## Buffer Lifecycle
//!
//! ```text
//! 1. Calculate Size - Analyse stackmap record for Indirect/Direct variables.
//!                     
//! 2. Get/Create Buffer - Reuse existing buffer if already allocated.
//!    Or allocate new buffer if needed.
//!                     
//! 3. Populate Buffer - Copy variables from stack to buffer.
//!                     
//! 4. Transfer Variables - Move variables from buffer to destinations.
//!                     
//! 5. Buffer Retention - Keep buffer instance live for reuse.
//! ```

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::collections::HashMap;
use yksmp::Location::{Direct, Indirect};
use yksmp::Record;

use crate::trace::swt::cp::ControlPointStackMapId;

/// A safer wrapper around allocated buffer memory with proper RAII
#[derive(Debug)]
pub(crate) struct AlignedBuffer {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuffer {
    /// Create a new aligned buffer with the specified size
    pub(crate) unsafe fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 8)
            .expect("Failed to create layout for live vars buffer");
        let ptr = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            std::ptr::write_bytes(ptr, 0, size);
            ptr
        };

        AlignedBuffer { ptr, layout }
    }

    /// Get the raw pointer to the buffer
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get the size of the buffer
    pub(crate) fn size(&self) -> usize {
        self.layout.size()
    }

    /// Get the layout of the buffer  
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}

// SAFETY: AlignedBuffer is Send because we control access to the raw pointer
// and ensure proper ownership transfer
unsafe impl Send for AlignedBuffer {}

// NOTE: We do NOT implement Sync for AlignedBuffer because concurrent access
// to the same buffer would be unsafe. Each buffer should be used by only one
// thread at a time, or protected by explicit synchronization.
thread_local! {
    static OPT_BUFFER: std::cell::RefCell<Option<AlignedBuffer>> = std::cell::RefCell::new(None);
    static UNOPT_BUFFER: std::cell::RefCell<Option<AlignedBuffer>> = std::cell::RefCell::new(None);
}

/// A high-level interface for managing temporary live variable buffers.
///
/// This struct provides metadata and access to an allocated buffer used for storing
/// live variables during control point transitions. It tracks the buffer's memory
/// layout, size, and variable offsets for efficient access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiveVarsBuffer {
    pub ptr: *mut u8,
    pub layout: Layout,
    // TODO: variables are only used in tests - can be removed
    pub variables: HashMap<i32, i32>,
    pub size: i32,
}

impl LiveVarsBuffer {
    pub(crate) fn new(
        ptr: *mut u8,
        layout: Layout,
        size: usize,
        variables: HashMap<i32, i32>,
    ) -> Self {
        LiveVarsBuffer {
            ptr,
            layout,
            variables,
            size: size as i32,
        }
    }

    /// Calculates the size of the live vars buffer.
    pub(crate) fn calculate_size(src_rec: &Record) -> i32 {
        let mut src_val_buffer_size: i32 = 0;
        for (_, src_var) in src_rec.live_vals.iter().enumerate() {
            match src_var.get(0).unwrap() {
                Indirect(_, _, src_val_size) | Direct(_, _, src_val_size) => {
                    src_val_buffer_size += *src_val_size as i32;
                }
                _ => { /* DO NOTHING */ }
            }
        }
        // Align the buffer size to 8 bytes (only round up, never down)
        if src_val_buffer_size % 8 == 0 {
            src_val_buffer_size
        } else {
            ((src_val_buffer_size / 8) + 1) * 8
        }
    }

    /// Gets or creates a thread-local buffer for temporary live variable storage.
    pub(crate) fn get_or_create(
        src_rec: &Record,
        smid: ControlPointStackMapId,
    ) -> (*mut u8, Layout, usize) {
        let src_val_buffer_size = LiveVarsBuffer::calculate_size(src_rec);

        if src_val_buffer_size == 0 {
            return (std::ptr::null_mut(), Layout::new::<u8>(), 0);
        }

        let buffer_cell = match smid {
            ControlPointStackMapId::UnOpt => &UNOPT_BUFFER,
            ControlPointStackMapId::Opt => &OPT_BUFFER,
        };

        buffer_cell.with(|cell| {
            let mut buffer_opt = cell.borrow_mut();

            // Check if we need a new buffer (first use or size changed)
            let needs_new_buffer = buffer_opt
                .as_ref()
                .map_or(true, |buf| buf.size() < src_val_buffer_size as usize);

            if needs_new_buffer {
                // Create new buffer with required size
                let new_buffer = unsafe { AlignedBuffer::new(src_val_buffer_size as usize) };
                *buffer_opt = Some(new_buffer);
            }

            let buffer = buffer_opt.as_ref().unwrap();
            (buffer.as_ptr(), buffer.layout(), buffer.size())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::LiveVarsBuffer;
    use yksmp::{Location, Record};

    #[test]
    fn test_calculate_live_vars_buffer_size() {
        let mock_record = Record {
            offset: 0,
            size: 0,
            id: 0,
            live_vals: vec![
                vec![Location::Indirect(0, 0, 16)].into(),
                vec![Location::Indirect(0, 0, 8)].into(),
                vec![Location::Indirect(0, 0, 4)].into(),
                vec![Location::Indirect(0, 0, 8)].into(),
            ],
        };

        let buffer_size = LiveVarsBuffer::calculate_size(&mock_record);
        assert_eq!(
            // 4 is the padding (round up 36 to the next multiple of 8)
            16 + 8 + 4 + 8 + 4,
            buffer_size,
            "Buffer size should equal the sum of all live variable sizes + padding"
        );
    }

    #[test]
    fn test_calculate_live_vars_buffer_size_alignment() {
        // Test cases with different initial sizes
        let test_cases = vec![
            (0, 0),   // 0 should remain 0
            (1, 8),   // 1 should become 8
            (8, 8),   // 8 should remain 8
            (9, 16),  // 9 should become 16
            (15, 16), // 15 should become 16
            (16, 16), // 16 should remain 16
        ];
        for (val_size, expected_buffer_size) in test_cases {
            // Create a mock record with the given buffer size
            let mock_record = Record {
                offset: 0,
                size: 0,
                id: 0,
                live_vals: vec![vec![Location::Indirect(0, 0, val_size)].into()],
            };
            let buffer_size = LiveVarsBuffer::calculate_size(&mock_record);
            assert_eq!(
                buffer_size, expected_buffer_size,
                "Buffer size for input {} should be {}",
                val_size, expected_buffer_size
            );
        }
    }
}
