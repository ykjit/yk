use std::{
    env,
    error::Error,
    path::PathBuf,
    process::{exit, Command},
};
use tempfile::TempDir;
use ykrt::compile::jitc_yk::aot_ir;

const OUTFILE: &str = "ykir";

#[cfg(target_family = "unix")]
fn extract_bytecode(tempdir: &TempDir, fname: &str) -> Result<PathBuf, Box<dyn Error>> {
    let mut out_file = tempdir.path().to_owned();
    out_file.push(OUTFILE);
    let output = Command::new("objcopy")
        .args(&[
            "--dump-section",
            &format!(".yk_ir={}", out_file.to_str().unwrap()),
            fname,
        ])
        .output()?;

    if !output.status.success() {
        return Err(String::from_utf8(output.stderr)?.into());
    }

    Ok(out_file)
}

fn inner() -> Result<(), Box<dyn Error>> {
    if let Some(exe) = env::args().skip(1).next() {
        let tempdir = TempDir::new()?;
        let bcfile = extract_bytecode(&tempdir, &exe)?;
        aot_ir::print_from_file(&bcfile)?;
        drop(tempdir); // hold it live until here so we can read from it.
        Ok(())
    } else {
        Err(String::from("Dumps Yk IR from an executable binary.\n\nusage: dump_ir <exe>").into())
    }
}

fn main() {
    if let Err(e) = inner() {
        eprintln!("{e}");
        exit(1);
    }
}
