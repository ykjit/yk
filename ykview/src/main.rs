use memmap::Mmap;
use std::{fs::File, io::Read};

fn main() {
    let mut args = std::env::args();
    args.next().unwrap(); // command name
    let path = args.next().unwrap_or_else(|| {
        println!(
            "No executable provided\n\
            Usage: ykview <executable> [<trace file>]\n\
            \n\
            Trace format:\n\
            symbol_name1 bb_index1\n\
            symbol_name2 bb_index2\n\
            ..."
        );
        std::process::exit(1);
    });

    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let sir = yktrace::sir::Sir::new(&mmap).unwrap();

    if let Some(trace_file) = args.next() {
        let trace_text = if trace_file == "-" {
            let mut trace = Vec::new();
            std::io::stdin().read_to_end(&mut trace).unwrap();
            trace
        } else {
            std::fs::read(trace_file).unwrap()
        };
        let trace_text = String::from_utf8(trace_text).unwrap();
        // Leak the text here as the sir trace requires a 'static borrow of the symbol names parsed
        // from this text.
        let trace_text = Box::leak(trace_text.into_boxed_str());
        let mut trace = vec![];
        for line in trace_text.lines() {
            let mut parts = line.trim().split(' ');
            let symbol_name = parts.next().unwrap();
            let bb_idx = parts.next().unwrap().parse::<u32>().unwrap();
            assert!(parts.next().is_none());
            trace.push(yktrace::sir::SirLoc {
                symbol_name,
                bb_idx,
                addr: None,
            });
        }
        for loc in &trace {
            println!("{:?}", loc);
        }
        let tir = yktrace::tir::TirTrace::new(&sir, &yktrace::sir::SirTrace::new(trace)).unwrap();
        println!("{}", tir);
    } else {
        println!("{}", sir);
    }
}
