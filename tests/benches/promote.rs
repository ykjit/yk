//! Critereon benchmarks for measuring the impact of promotion.

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

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_TIME: Duration = Duration::from_secs(20);

fn compile_bench(tempdir: &TempDir, promote: bool) -> PathBuf {
    let md = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = [&md, "benches", "promote.c"].iter().collect::<PathBuf>();

    let mut exe = PathBuf::new();
    exe.push(tempdir);
    exe.push(src.file_stem().unwrap());

    let mut compiler = mk_compiler(&ykllvm_bin("clang"), &exe, &src, &[], true, None);
    compiler.arg("-ltests");
    if promote {
        compiler.arg("-DDO_PROMOTE");
    }
    let out = compiler.output().unwrap();
    check_output(&out);

    exe
}

fn setup<'c, M: Measurement>(
    c: &'c mut Criterion<M>,
    group_name: &str,
) -> (TempDir, PathBuf, PathBuf, BenchmarkGroup<'c, M>) {
    let tempdir = TempDir::new().unwrap();
    let bin_np = compile_bench(&tempdir, false);
    let bin_p = compile_bench(&tempdir, true);

    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(MEASUREMENT_TIME);
    group.sampling_mode(SamplingMode::Flat);

    (tempdir, bin_np, bin_p, group)
}

fn run_bench(bench_bin: &Path, param: usize) {
    assert!(bench_bin.exists());
    let out = Command::new(bench_bin)
        .arg(format!("{}", param))
        .env("YKD_SERIALISE_COMPILATION", "1")
        .env("YKD_LOG_JITSTATE", "1")
        .output()
        .unwrap();
    check_output(&out);
}

fn bench_promote(c: &mut Criterion) {
    let (_tempdir, bin_np, bin_p, mut group) = setup(c, "promote");

    let reps = 100000;
    // Benchmark a simple loop *without* promotion.
    group.bench_function(
        BenchmarkId::new("without promotes", format!("{}", reps)),
        |b| b.iter(|| run_bench(&bin_np, reps)),
    );
    // Benchmark a simple loop *with* promotion.
    group.bench_function(
        BenchmarkId::new("with promotes", format!("{}", reps)),
        |b| b.iter(|| run_bench(&bin_p, reps)),
    );
}

criterion_group!(promote_benchmarks, bench_promote);
criterion_main!(promote_benchmarks);
