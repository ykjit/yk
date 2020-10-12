use std::io::Read;

fn main() {
    let mut args = std::env::args();
    args.next().unwrap(); // command name
    let sir = if let Some(file) = args.next() {
        yktrace::sir::Sir::read_file(file.as_ref()).unwrap()
    } else {
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
    };

    if let Some(trace_file) = args.next() {
        let trace_text = if trace_file == "-" {
            let mut trace = Vec::new();
            std::io::stdin().read_to_end(&mut trace).unwrap();
            trace
        } else {
            std::fs::read(trace_file).unwrap()
        };
        let trace_text = String::from_utf8(trace_text).unwrap();
        let mut trace = VecSirTrace(vec![], ykpack::Local(0) /*FIXME*/);
        for line in trace_text.lines() {
            let mut parts = line.trim().split(" ");
            let symbol_name = parts.next().unwrap().to_string();
            let bb_idx = parts.next().unwrap().parse::<u32>().unwrap();
            assert!(parts.next().is_none());
            trace.0.push(yktrace::sir::SirLoc {
                symbol_name,
                bb_idx,
                addr: None,
            });
        }
        for loc in yktrace::sir::SirTraceIterator::new(&trace) {
            println!("{:?}", loc);
        }
        let tir = yktrace::tir::TirTrace::new(&sir, &trace).unwrap();
        println!("{}", tir);
    } else {
        println!("SIR:");
        println!("{}", sir);
    }
}

#[derive(Debug)]
struct VecSirTrace(Vec<yktrace::sir::SirLoc>, ykpack::Local);

impl yktrace::sir::SirTrace for VecSirTrace {
    fn raw_len(&self) -> usize {
        self.0.len()
    }

    fn raw_loc(&self, idx: usize) -> &yktrace::sir::SirLoc {
        &self.0[idx]
    }
}
