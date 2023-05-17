//! Generate compilation command databases for use with clangd.

use crate::target_dir;
use glob::glob;
use std::{
    env,
    fs::{create_dir_all, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::TempDir;

/// Wrap C/C++ compiler commands to help LSP. At the moment this produces output suitable for
/// [clangd](https://clangd.llvm.org/) into (typically)
/// `target/<debug|release>/clangd/<crate-name>/compile_commands.json`.
///
/// It works by providing a compiler wrapper which is called in lieu of the compiler: it captures
/// full compiler command-line invocations, writes them to a temporary directory, then joins them
/// together.
///
/// Steps for use:
/// - Call `CompletionWrapper::new()` with the desired parameters.
/// - Compile your C/C++ code with the compiler wrapper returned by `CCGenerator::wrapper_path()`
///   with the environment returned by `CCGenerator::build_env()` applied. Note that the source
///   file to be compiled MUST be the last argument of the compiler invocation!
/// - Once all compilation is complete, the consumer can generate the JSON file by calling
///   `generate()`.
///
/// By default `clangd` will only search parent directories of source files for databases, so it
/// will not find the generated databases in the Rust `target` directory. You will need to
/// place a `.clangd` file in each yk crate:
///
/// ```ignore
/// CompileFlags:
///     CompilationDatabase: ../target/<debug|release>/clangd/<crate-name>/
/// ```
pub struct CompletionWrapper {
    db_subdir: String,
    dir_field: String,
    tmpdir: TempDir,
}

impl CompletionWrapper {
    /// Create a compiler commands database generator.
    ///
    ///  - `db_subdir` specifies the subdirectory of `target/compiler_commands/` to put the
    ///    generated JSON file into.
    ///
    ///  - `dir_field` specifies the string to use for the `directory` fields inside the generated
    ///    JSON file. See: https://clang.llvm.org/docs/JSONCompilationDatabase.html
    pub fn new(db_subdir: &str, dir_field: &str) -> Self {
        Self {
            db_subdir: db_subdir.to_owned(),
            dir_field: dir_field.to_owned(),
            tmpdir: TempDir::new().unwrap(),
        }
    }

    /// Return the path of the `completion-wrapper`.
    pub fn wrapper_path(&self) -> PathBuf {
        let mut p = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .parent()
            .unwrap()
            .to_path_buf();
        p.push("ykbuild");
        p.push("completion-wrapper");
        p
    }

    /// Returns the key and value that must be applied to the wrapped compiler's environment.
    pub fn build_env(&self) -> (&str, &str) {
        ("YK_COMPILER_TEMPDIR", self.tmpdir.path().to_str().unwrap())
    }

    /// Call when the build is done to generate the `compile_commands.json` file.
    pub fn generate(self) {
        let mut entries = Vec::new();

        for path in glob(&format!("{}/*", self.tmpdir.path().to_str().unwrap())).unwrap() {
            let mut infile = File::open(path.unwrap()).unwrap();
            let mut buf = String::new();
            infile.read_to_string(&mut buf).unwrap();
            let buf = buf.trim();

            // We assume (and assert) that the source file is the last argument.
            let ccfile = buf.split(' ').last().unwrap();
            assert!(["c", "cpp", "cxx", "cc"]
                .iter()
                .any(|e| e == &Path::new(ccfile).extension().unwrap().to_str().unwrap()));
            let mut entry = String::new();
            entry.push_str("  {\n");
            entry.push_str(&format!("    \"directory\": \"{}\",\n", self.dir_field));
            entry.push_str(&format!("    \"command\": \"{buf}\",\n"));
            entry.push_str(&format!("    \"file\": \"{ccfile}\",\n"));
            entry.push_str("  }");
            entries.push(entry);
        }

        // Write JSON to compile_commands.json.
        let mut p = target_dir();
        p.push("clangd");
        p.push(&self.db_subdir);
        create_dir_all(p.clone()).unwrap();
        p.push("compile_commands.json");
        let mut f = File::create(p).unwrap();
        writeln!(f, "[").unwrap();
        write!(f, "{}", entries.join(",\n")).unwrap();
        write!(f, "\n]\n").unwrap();
    }
}
