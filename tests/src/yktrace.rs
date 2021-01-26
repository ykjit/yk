// FIXME hard-coded hardware testing. Use default mode, but requires more shimming.

use std::{hint::black_box, thread};
use ykshim_client::{start_tracing, TracingKind};

// Some work to trace.
#[interp_step]
fn work(io: &mut WorkIO) {
    let mut res = 0;
    for i in 0..(io.0) {
        if i % 2 == 0 {
            res += 5;
        } else {
            res += 10 / i;
        }
    }
    println!("{}", res); // prevents the above from being optimised out.
}

struct WorkIO(usize);

/// Test that basic tracing works.
#[test]
fn trace() {
    #[cfg(tracermode = "hw")]
    let th = start_tracing(TracingKind::HardwareTracing);
    black_box(work(&mut WorkIO(10)));
    let trace = th.stop_tracing().unwrap();
    assert!(trace.len() > 0);
}

/// Test that tracing twice sequentially in the same thread works.
#[test]
fn trace_twice() {
    #[cfg(tracermode = "hw")]
    let kind = TracingKind::HardwareTracing;

    let th1 = start_tracing(kind);
    black_box(work(&mut WorkIO(10)));
    let trace1 = th1.stop_tracing().unwrap();

    let th2 = start_tracing(kind);
    black_box(work(&mut WorkIO(20)));
    let trace2 = th2.stop_tracing().unwrap();

    assert!(trace1.len() < trace2.len());
}

/// Test that tracing in different threads works.
#[test]
pub(crate) fn trace_concurrent() {
    #[cfg(tracermode = "hw")]
    let kind = TracingKind::HardwareTracing;

    let thr = thread::spawn(move || {
        let th1 = start_tracing(kind);
        black_box(work(&mut WorkIO(10)));
        th1.stop_tracing().unwrap().len()
    });

    let th2 = start_tracing(kind);
    black_box(work(&mut WorkIO(20)));
    let len2 = th2.stop_tracing().unwrap().len();

    let len1 = thr.join().unwrap();
    assert!(len1 < len2);
}

mod tir {
    use super::black_box;
    use crate::helpers::assert_tir;
    use ykrt::trace_debug;
    use ykshim_client::{start_tracing, TirTrace, TracingKind};

    #[test]
    fn nonempty_tir_trace() {
        #[inline(never)]
        #[interp_step]
        fn work(io: &mut IO) {
            let mut res = 0;
            while res < io.1 {
                res += io.0;
            }
            io.2 = res
        }

        struct IO(usize, usize, usize);
        let mut io = IO(3, 13, 0);
        // FIXME TracingMode::Default.
        #[cfg(tracermode = "hw")]
        let tracer = start_tracing(TracingKind::HardwareTracing);
        black_box(work(&mut io));
        let sir_trace = tracer.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&sir_trace);
        assert_eq!(io.2, 15);
        assert!(tir_trace.len() > 0);
    }

    struct DebugTirIO(usize, usize);

    #[inline(never)]
    #[interp_step]
    fn debug_tir_work(io: &mut DebugTirIO) {
        match io.0 {
            0 => {
                trace_debug("Add 10");
                io.1 += 10;
            }
            1 => {
                trace_debug("Minus 2");
                io.1 -= 2;
            }
            2 => {
                trace_debug("Multiply 2");
                io.1 *= 2;
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn trace_debug_tir() {
        let mut io = DebugTirIO(0, 0);
        #[cfg(tracermode = "hw")]
        let tracer = start_tracing(TracingKind::HardwareTracing);
        black_box(debug_tir_work(&mut io)); // +10
        black_box(debug_tir_work(&mut io)); // +10
        io.0 = 2;
        black_box(debug_tir_work(&mut io)); // *2
        io.0 = 1;
        black_box(debug_tir_work(&mut io)); // -2
        let sir_trace = tracer.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&sir_trace);
        assert_eq!(io.1, 38);
        assert_tir(
            "...\n\
            ops:\n\
              ...
              // Add 10
              ...
              ... + 10usize (checked)
              ...
              // Add 10
              ...
              ... + 10usize (checked)
              ...
              // Multiply 2
              ...
              ... * 2usize (checked)
              ...
              // Minus 2
              ...
              ... - 2usize (checked)
              ...",
            &tir_trace,
        );
    }
}
