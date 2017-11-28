// Copyright (c) 2017 King's College London
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
use std::env;
use std::process::Command;
use std::collections::HashMap;

fn run_lsb_release(args: Vec<&str>) -> HashMap<String, String> {
    let output = Command::new("lsb_release")
                         .args(args)
                         .output()
                         .unwrap();
    assert!(output.status.success());
    let output_s = String::from_utf8(output.stdout).unwrap();

    let mut res = HashMap::new();
    for line in output_s.lines() {
        let parts: Vec<&str> = line.split(":").collect();
        assert!(parts.len() == 2);
        res.insert(String::from(parts[0].trim()), String::from(parts[1].trim()));
    }
    res
}

fn main() {
    let mut c_build = gcc::Build::new();
    c_build.file("src/traceme.c").include("src");

    // Travis' Linux setup is too old for what we do with linux-perf.
    if env::var("TRAVIS").is_ok() {
        // Crash out when Travis updates to a newer Ubuntu or a different OS.
        // When this fires, travis may have a new enough linux-perf to test properly.
        let lsb_rel = run_lsb_release(vec!["-ir"]);
        assert!(lsb_rel["Distributor ID"] == "Ubuntu");
        assert!(lsb_rel["Release"] == "14.04");
        // Setting -DTRAVIS causes the C API to be stubbed.
        c_build.define("TRAVIS", None);
    }

    c_build.compile("traceme");
}
