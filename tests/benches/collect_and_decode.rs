//! Critereon benchmarks.

use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion, SamplingMode,
};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};
use tempfile::TempDir;
use tests::{check_output, mk_compiler};
use ykbuild::ykllvm_bin;

const SAMPLE_SIZE: usize = 50;
const MEASUREMENT_TIME: Duration = Duration::from_secs(30);

fn compile_runner(tempdir: &TempDir) -> PathBuf {
    let md = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = [&md, "benches", "collect_and_decode.c"]
        .iter()
        .collect::<PathBuf>();

    let mut exe = PathBuf::new();
    exe.push(tempdir);
    exe.push(src.file_stem().unwrap());

    let mut compiler = mk_compiler(&ykllvm_bin("clang"), &exe, &src, &[], false, None);
    compiler.arg("-ltests");
    let out = compiler.output().unwrap();
    check_output(&out);

    exe
}

fn collect_and_decode_trace(runner: &Path, benchmark: usize, param: usize) {
    let out = Command::new(runner)
        .arg(format!("{}", benchmark))
        .arg(format!("{}", param))
        .output()
        .unwrap();
    check_output(&out);
}

fn setup<'c, M: Measurement>(
    c: &'c mut Criterion<M>,
    group_name: &str,
) -> (TempDir, PathBuf, BenchmarkGroup<'c, M>) {
    let tempdir = TempDir::new().unwrap();
    let runner = compile_runner(&tempdir);

    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(MEASUREMENT_TIME);
    group.sampling_mode(SamplingMode::Flat);

    (tempdir, runner, group)
}

/// Benchmark decoding a trace where most of the code has blockmap entries (i.e. where ykpt can use
/// compiler-assisted decoding for the most part).
fn bench_native(c: &mut Criterion) {
    let (_tempdir, runner, mut group) = setup(c, "trace-decode-native");

    for param in [1000, 10000, 100000] {
        group.bench_function(BenchmarkId::new("YkPT", format!("{}", param)), |b| {
            b.iter(|| collect_and_decode_trace(&runner, 0, param))
        });
    }
}

/// Benchmark decoding a trace which does a lot of calls to foreign code (thus blocking ykpt from
/// doing compiler-assisted decoding).
fn bench_disasm(c: &mut Criterion) {
    let (_tempdir, runner, mut group) = setup(c, "trace-decode-disasm");

    for param in [10, 30, 50] {
        group.bench_function(BenchmarkId::new("YkPT", format!("{}", param)), |b| {
            b.iter(|| collect_and_decode_trace(&runner, 1, param))
        });
    }
}

criterion_group!(decoder_benchmarks, bench_native, bench_disasm);
criterion_main!(decoder_benchmarks);
