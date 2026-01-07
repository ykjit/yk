pub(crate) mod c_errors;
pub(crate) mod ykpt;

use core::arch::x86_64::__cpuid_count;

/// Checks if the CPU supports Intel Processor Trace.
pub(crate) fn pt_supported() -> bool {
    // Check that the chip has PT capabilities.
    let cpuid1 = __cpuid_count(0x7, 0x0);
    let res1 = (cpuid1.ebx & (1 << 25)) != 0;

    // Check that the chip supports at least one PT IP filtering range.
    let cpuid2 = __cpuid_count(0x14, 0x0);
    let res2 = (cpuid2.ebx & (1 << 2)) != 0;

    res1 && res2
}
