/// Test that it's OK for the process being traced to change directory, even if it was invoked with
/// a relative path.
///
/// This may seem like a rather arbitrary thing to check, but this test was derived from a real
/// bug: the LibIPT decoder was trying to create a libipt image using a (stale) relative path for
/// the main binary's object.
use hwtracer::{
    collect::TraceCollectorBuilder,
    decode::{TraceDecoderBuilder, TraceDecoderKind},
};
use std::{env, ffi::CString, path::PathBuf, time::SystemTime};

#[inline(never)]
pub fn work_loop(iters: u64) -> u64 {
    let mut res = 0;
    for _ in 0..iters {
        // Computation which stops the compiler from eliminating the loop.
        res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
    }
    res
}

#[test]
fn pt_chdir_rel() {
    let arg0 = env::args().next().unwrap();
    if arg0.starts_with("/") {
        // Reinvoke ourself with a relative path.
        let path = PathBuf::from(arg0);

        let dir = path.parent().unwrap();
        env::set_current_dir(&dir.to_str().unwrap()).unwrap();

        let prog = path.file_name().unwrap().to_str().unwrap();
        let prog_c = CString::new(prog).unwrap();
        let prog_p = prog_c.as_ptr();

        let args = env::args().collect::<Vec<_>>();
        let mut args_p = args.iter().map(|a| a.as_ptr()).collect::<Vec<_>>();
        args_p[0] = prog_p as *const u8; // Replace absolute path.
        args_p.push(0 as *const u8); // NULL sentinel.

        // We don't use `std::process::Command` because it can't reliably handle a relative path.
        unsafe { libc::execv(prog_p as *const i8, args_p.as_ptr() as *const *const i8) };
        unreachable!();
    }

    // When we get here, we have a process that was invoked with a relative path.

    let tc = TraceCollectorBuilder::new().build().unwrap();
    tc.start_thread_collector().unwrap();
    println!("{}", work_loop(env::args().len() as u64));
    let trace = tc.stop_thread_collector().unwrap();

    // Now check that the trace decoder can still find its objects after we change dir.
    let tdec = TraceDecoderBuilder::new()
        .kind(TraceDecoderKind::LibIPT)
        .build()
        .unwrap();
    env::set_current_dir("/").unwrap();
    for b in tdec.iter_blocks(&*trace) {
        b.unwrap(); // this would error if the decoder was confused by changing dir.
    }
}
