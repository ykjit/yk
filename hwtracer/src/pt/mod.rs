use core::arch::x86_64::__cpuid_count;

/// Checks if the CPU supports Intel Processor Trace.
pub(crate) fn pt_supported() -> bool {
    let res = unsafe { __cpuid_count(0x7, 0x0) };
    (res.ebx & (1 << 25)) != 0
}
