//! Generate compilation command databases for use with clangd.

use crate::manifest_dir;
use glob::glob;
use std::{
    fs::{create_dir_all, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::TempDir;

/// A language that clangd understands.
pub enum CCLang {
    C,
    CPP,
}

impl CCLang {
    fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::C => &["c"],
            Self::CPP => &["cpp", "cxx", "cc"],
        }
    }

    pub fn compiler_wrapper(&self) -> PathBuf {
        let script = match self {
            Self::C => "wrap-clang.sh",
            Self::CPP => "wrap-clang++.sh",
        };
        [&manifest_dir(), "..", "ykbuild", script]
            .iter()
            .collect::<PathBuf>()
    }
}

/// Generate a clangd compilation databases to give LSP support for Yk's C/C++ code.
///
/// This provides a compiler wrapper which captures the compilation commands and a way to write
/// them into a `compile_commands.json` file that can be used by clangd.
///
/// This only exists in part due to:
/// https://github.com/rust-lang/cc-rs/issues/497
///
/// but also so that we can use clangd on our lang_tester suites.
///
/// Steps for use:
///
/// - Call `CCGenerator::new()` with the desired parameters.
///
/// - Compile your C/C++ code with the compiler wrapper returned by `CCLang::compiler_wrapper()`
///   with the environment returned by `CCGenerator::build_env()` applied. Note that the source
///   file to be compiled MUST be the last argument of the compiler invocation!
///
/// - Once all compilation is complete, the consumer can generate the JSON file by calling
///   `generate()`.
///
/// By default `clangd` will only search parent directories of source files for databases, so it
/// will not find the generated databases in the Rust `target` directory. You will need to
/// strategically place `.clangd` configuration files like the following:
///
/// ```ignore
/// CompileFlags:
///     CompilationDatabase: ../target/compile_commands/<db_subdir>/
/// ```
pub struct CCGenerator {
    db_subdir: String,
    dir_field: String,
    tmpdir: TempDir,
}

impl CCGenerator {
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

    /// Returns the key and value that must be applied to the wrapped compiler's environment.
    pub fn build_env(&self) -> (&str, &str) {
        ("YK_CC_TEMPDIR", self.tmpdir.path().to_str().unwrap())
    }

    /// Call when the build is done to generate the `build_commands.json` file.
    pub fn generate(self) {
        let mut entries = Vec::new();

        for path in glob(&format!("{}/*", self.tmpdir.path().to_str().unwrap())).unwrap() {
            let mut infile = File::open(path.unwrap()).unwrap();
            let mut buf = String::new();
            infile.read_to_string(&mut buf).unwrap();
            let buf = buf.trim();

            // We assume (and assert) that the source file is the last argument.
            let ccfile = buf.split(' ').last().unwrap();
            assert!(CCLang::C
                .extensions()
                .iter()
                .chain(CCLang::CPP.extensions())
                .any(|e| e == &Path::new(ccfile).extension().unwrap().to_str().unwrap()));
            let mut entry = String::new();
            entry.push_str("  {\n");
            entry.push_str(&format!("    \"directory\": \"{}\",\n", self.dir_field));
            entry.push_str(&format!("    \"command\": \"{buf}\",\n"));
            entry.push_str(&format!("    \"file\": \"{ccfile}\",\n"));
            entry.push_str("  }");
            entries.push(entry);
        }

        let out_dir = [
            &manifest_dir(),
            "..",
            "target",
            "compile_commands",
            &self.db_subdir,
        ]
        .iter()
        .collect::<PathBuf>();
        create_dir_all(out_dir.clone()).unwrap();

        // Write JSON to Rust target dir.
        let outpath = [&out_dir, Path::new("compile_commands.json")]
            .iter()
            .collect::<PathBuf>();
        let mut outfile = File::create(outpath).unwrap();
        writeln!(outfile, "[").unwrap();
        write!(outfile, "{}", entries.join(",\n")).unwrap();
        write!(outfile, "\n]\n").unwrap();
    }
}
