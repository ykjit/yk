#![allow(dead_code)]

/// Which direction does the CPU stack grow when more room is required?
pub(crate) enum StackDirection {
    /// The CPU stack grows "downwards" i.e. growth changes it to a lower address.
    GrowsToHigherAddress,
    /// The CPU stack grows "upwards" i.e. growth changes it to a higher address.
    GrowsToLowerAddress,
}

#[cfg(target_arch = "x86_64")]
pub(crate) const STACK_DIRECTION: StackDirection = StackDirection::GrowsToLowerAddress;
