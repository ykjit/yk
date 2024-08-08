use std::env;
use std::process::{exit, Command};
use walkdir::WalkDir;
use ykbuild::ykllvm_bin;

fn clang_format(check_only: bool) {
    let mut failed = false;
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_entry(|x| !(x.path().starts_with("./.") || x.path().starts_with("./ykllvm")))
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
                    if check_only {
                        match Command::new(&clang_format)
                            .args(["--dry-run", "-ferror-limit=1", "-Werror"])
                            .arg(entry.path())
                            .output()
                        {
                            Ok(r) => {
                                if !r.status.success() {
                                    eprintln!(
                                        "{}",
                                        std::str::from_utf8(&r.stderr)
                                            .unwrap_or("<non UTF-8 stderr>")
                                    );
                                    failed = true;
                                }
                            }
                            Err(e) => panic!("{e:?}"),
                        }
                    } else {
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
                                        err_msg =
                                            Some(format!("returned exit code {c}:\n\n{stderr}"))
                                    }
                                    None => {
                                        err_msg =
                                            Some("terminated by signal:\n\n{stderr}".to_owned())
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
                }
                _ => {}
            }
        }
    }

    if failed {
        exit(1);
    }
}

fn usage() -> ! {
    eprintln!(
        "Please specify task to run:
cfmt [--check]    Formats C/C++ files with clang-format."
    );
    exit(1)
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(|x| x.as_str()) {
        Some("cfmt") => {
            let check_only = match args.get(1).map(|x| x.as_str()) {
                Some("--check") => true,
                Some(_) => usage(),
                None => false,
            };
            clang_format(check_only);
        }
        _ => usage(),
    }
}
