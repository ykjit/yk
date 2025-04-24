use object::{Object, ObjectSection};
use std::sync::LazyLock;
#[cfg(not(test))]
use std::thread;
use ykaddr::obj::SELF_BIN_MMAP;
use yksmp::{PrologueInfo, Record, StackMapParser};

/// Parsed stackmap information of the AOT module.
pub(crate) struct AOTStackmapInfo {
    /// Prologue information for each function.
    pinfos: Vec<PrologueInfo>,
    /// All stackmap records of the module, and the index of the prologue info relevant for each
    /// record.
    records: Vec<(Record, usize)>,
}

impl AOTStackmapInfo {
    pub(crate) fn get(&self, stackmapid: usize) -> (&Record, &PrologueInfo) {
        let (rec, pid) = &self.records[stackmapid];
        let pinfo = &self.pinfos[*pid];
        (rec, pinfo)
    }
}

pub(crate) static AOT_STACKMAPS: LazyLock<Result<AOTStackmapInfo, String>> = LazyLock::new(|| {
    fn errstr(msg: &str) -> String {
        format!("failed to load stackmaps: {msg}")
    }

    // We use an inner function so that we can use the `?` operator for errors.
    fn load_stackmaps() -> Result<AOTStackmapInfo, String> {
        // Load the stackmap from the binary to parse in tthe stackmaps.
        let object = object::File::parse(&**SELF_BIN_MMAP).map_err(|e| errstr(&e.to_string()))?;
        let sec = object
            .section_by_name(".llvm_stackmaps")
            .ok_or_else(|| errstr("can't find section"))?;

        // Parse the stackmap.
        let data = sec.data().map_err(|e| errstr(&e.to_string()))?;
        let (pinfos, records) = StackMapParser::parse(data);
        Ok(AOTStackmapInfo { pinfos, records })
    }

    load_stackmaps()
});

pub(crate) fn load_aot_stackmaps() {
    // Rust unit test binaries will not contain stackmaps, so don't try to load them.
    #[cfg(not(test))]
    thread::spawn(|| LazyLock::force(&AOT_STACKMAPS));
}
