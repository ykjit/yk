use std::env;
use std::process::Command;
use walkdir::{DirEntry, WalkDir};
use ykbuild::ykllvm_bin;

fn ignore_dir(entry: &DirEntry) -> bool {
    if entry.path().starts_with("./target")
        || entry.path().starts_with("./.cargo")
        || entry.path().starts_with("./.git")
        || entry.path().starts_with("./ykllvm")
    {
        return false;
    }
    true
}

fn clang_format() {
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_entry(ignore_dir)
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(ext) = entry.path().extension() {
            match ext.to_str().unwrap() {
                "h" | "c" | "cpp" | "cc" => {
                    let clang_format = ykllvm_bin("clang-format");
                    let res = Command::new(&clang_format)
                        .arg("-i")
                        .arg(entry.path())
                        .output()
                        .expect(&format!("Failed to execute {:?}", clang_format));
                    if !res.status.success() {
                        panic!(
                            "{:?} failed on {}",
                            clang_format,
                            entry.path().to_str().unwrap()
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

fn main() {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("cfmt") => clang_format(),
        _ => println!(
            "Please specify task to run:
cfmt       Formats C/C++ files with clang-format."
        ),
    }
}
