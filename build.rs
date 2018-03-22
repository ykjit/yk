// Copyright (c) 2017-2018 King's College London
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

extern crate gcc;

#[cfg(target_os = "linux")]
use std::path::{PathBuf, Path};
use std::env;
use std::process::Command;

#[cfg(target_os = "linux")]
const FEATURE_CHECKS_PATH: &str = "feature_checks";

const C_DEPS_PATH: &str = "c_deps";

/// Simple feature check, returning `true` if we have the feature.
///
/// The checks themselves are in files under `FEATURE_CHECKS_PATH`.
#[cfg(target_os = "linux")]
fn feature_check(filename: &str) -> bool {
    let mut path = PathBuf::new();
    path.push(FEATURE_CHECKS_PATH);
    path.push(filename);

    let mut check_build = gcc::Build::new();
    check_build.file(path).try_compile("check_perf_pt").is_ok()
}

// XXX Currently no way to clean the c_deps dir at `cargo clean` time:
// https://github.com/rust-lang/cargo/issues/572

fn build_libipt() {
    eprintln!("Building libipt...");
    env::set_current_dir(&Path::new(C_DEPS_PATH)).unwrap();
    let res = Command::new("make")
        .arg("libipt")
        .output()
        .unwrap_or_else(|_| panic!("Fatal error when building libipt"));
    if !res.status.success() {
        eprintln!("libipt build failed\n>>> stdout");
        eprintln!("stdout: {}", String::from_utf8_lossy(&res.stdout));
        eprintln!("\n>>> stderr");
        eprintln!("stderr: {}", String::from_utf8_lossy(&res.stderr));
        panic!();
    }
    env::set_current_dir(&Path::new("..")).unwrap();
}

// We always fetch libipt regardless of if we will build our own libipt. This is becuase there are
// a couple of private CPU configuration files that we need to borrow from libipt.
fn fetch_libipt() {
    eprintln!("Fetch libipt...");
    env::set_current_dir(&Path::new(C_DEPS_PATH)).unwrap();
    let res = Command::new("make")
        .arg("processor-trace") // target just fetches the code.
        .output()
        .unwrap_or_else(|_| panic!("Fatal error when fetching libipt"));
    if !res.status.success() {
        eprintln!("libipt fetch failed\n>>> stdout");
        eprintln!("stdout: {}", String::from_utf8_lossy(&res.stdout));
        eprintln!("\n>>> stderr");
        eprintln!("stderr: {}", String::from_utf8_lossy(&res.stderr));
        panic!();
    }
    env::set_current_dir(&Path::new("..")).unwrap();
}

fn main() {
    let mut c_build = gcc::Build::new();

    // Check if we should build the perf_pt backend.
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        if feature_check("check_perf_pt.c") {
            c_build.file("src/backends/perf_pt/collect.c");
            c_build.file("src/backends/perf_pt/decode.c");

            // XXX At the time of writing you can't conditionally build C code for tests in a build
            // script: https://github.com/rust-lang/cargo/issues/1581
            c_build.file("src/backends/perf_pt/test_helpers.c");

            // Decide whether to build our own libipt.
            if let Ok(val) = env::var("IPT_PATH") {
                let mut inc_path = PathBuf::from(val.clone());
                inc_path.push("include");
                c_build.include(inc_path);
                c_build.flag(&format!("-L{}/lib", val));
                println!("cargo:rustc-link-search={}/lib", val);
                println!("cargo:rustc-env=PTXED={}/bin/ptxed", val);
            } else {
                build_libipt();
                c_build.include(&format!("{}/inst/include/", C_DEPS_PATH));
                c_build.flag(&format!("-L{}/inst/lib", C_DEPS_PATH));
                println!("cargo:rustc-link-search={}/inst/lib", C_DEPS_PATH);
                println!("cargo:rustc-env=PTXED={}/inst/bin/ptxed", C_DEPS_PATH);
            }

            // We borrow the CPU detection functions from libipt (they are not exposed publicly).
            // If we built our own libipt above, then the fetch is a no-op.
            fetch_libipt();
            c_build.include(&format!("{}/processor-trace/libipt/internal/include", C_DEPS_PATH));
            c_build.file(&format!("{}/processor-trace/libipt/src/pt_cpu.c", C_DEPS_PATH));
            c_build.file(&format!("{}/processor-trace/libipt/src/posix/pt_cpuid.c", C_DEPS_PATH));

            println!("cargo:rustc-cfg=perf_pt");
            println!("cargo:rustc-link-lib=ipt");

            // XXX Cargo bug: no way to encode an rpath, otherwise we would do that here:
            // https://github.com/rust-lang/cargo/issues/5077
            //
            // Until this is implemented, the user will need to add c_deps/inst/lib to
            // LD_LIBRARY_PATH if the build process compiles its own libipt.
        }
    }
    c_build.file("src/util/util.c");
    c_build.compile("hwtracer_c");

    // Additional circumstances under which to re-run this build.rs.
    println!("cargo:rerun-if-env-changed=IPT_PATH");
    println!("cargo:rerun-if-changed=src/util");
    println!("cargo:rerun-if-changed={}", C_DEPS_PATH);
    println!("cargo:rerun-if-changed=src/backends/perf_pt");
    println!("cargo:rerun-if-changed={}/processor-trace/libipt/src/pt_cpu.c", C_DEPS_PATH);
    println!("cargo:rerun-if-changed={}/processor-trace/libipt/src/posix/pt_cpuid.c", C_DEPS_PATH);
}
