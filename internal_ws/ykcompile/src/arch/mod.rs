//! Platform support for the Yorick TIR trace compiler.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("Currently only linux x86_64 is supported.");

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub(super) mod x86_64;
