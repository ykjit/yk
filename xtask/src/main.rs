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
                    let mut err_msg = None;
                    match Command::new(&clang_format)
                        .arg("-i")
                        .arg(entry.path())
                        .output()
                    {
                        Ok(r) => {
                            let stderr =
                                std::str::from_utf8(&r.stderr).unwrap_or("<non UTF-8 stderr>");
                            match r.status.code() {
                                Some(0) => (),
                                Some(c) => {
                                    err_msg = Some(format!("returned exit code {c}:\n\n{stderr}"))
                                }
                                None => {
                                    err_msg = Some("terminated by signal:\n\n{stderr}".to_owned())
                                }
                            }
                        }
                        Err(e) => err_msg = Some(e.to_string()),
                    }
                    if let Some(m) = err_msg {
                        panic!(
                            "{} -i {}: {m}",
                            clang_format.as_path().to_str().unwrap(),
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
