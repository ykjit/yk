//! Note that LLVM currently only supports stackmaps for 64 bit architectures. Once they support
//! others we will need to either make this parser more dynamic or create a new one for each
//! architecture.
#[cfg(not(target_arch = "x86_64"))]
compile_error!("The stackmap parser currently only supports x64.");

use std::collections::HashMap;
use std::error;

struct Function {
    addr: u64,
    record_count: u64,
    stack_size: u64,
}

pub struct Record {
    pub id: u64,
    pub offset: u64,
    pub live_vars: Vec<LiveVar>,
    pub size: u64,
}

impl Record {
    pub fn empty() -> Record {
        Record {
            id: 0,
            offset: 0,
            live_vars: Vec::new(),
            size: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Location {
    Register(u16, u16, i32, u16),
    Direct(u16, i32, u16),
    Indirect(u16, i32, u16),
    Constant(u32),
    LargeConstant(u64),
}

#[derive(Debug)]
pub struct LiveVar {
    locs: Vec<Location>,
}

impl LiveVar {
    pub fn len(&self) -> usize {
        self.locs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locs.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&Location> {
        self.locs.get(idx)
    }
}

pub struct PrologueInfo {
    pub hasfp: bool,
    pub csrs: Vec<(u16, i32)>,
}

pub struct SMEntry {
    pub pinfo: PrologueInfo,
    pub records: Vec<Record>,
}

/// Parses LLVM stackmaps version 3 from a given address. Provides a way to query relevant
/// locations given the return address of a `__llvm_deoptimize` function.
pub struct StackMapParser<'a> {
    data: &'a [u8],
    offset: usize,
}

impl StackMapParser<'_> {
    pub fn parse(data: &[u8]) -> Result<HashMap<u64, Vec<LiveVar>>, Box<dyn error::Error>> {
        let mut smp = StackMapParser { data, offset: 0 };
        let (entries, _) = smp.read()?;
        let mut map = HashMap::new();
        for sme in entries {
            for r in sme.records {
                map.insert(r.offset, r.live_vars);
            }
        }
        Ok(map)
    }

    pub fn get_entries(data: &[u8]) -> (Vec<SMEntry>, u32) {
        let mut smp = StackMapParser { data, offset: 0 };
        smp.read().unwrap()
    }

    fn read(&mut self) -> Result<(Vec<SMEntry>, u32), Box<dyn error::Error>> {
        // Read version number.
        if self.read_u8() != 3 {
            return Err("Only stackmap format version 3 is supported.".into());
        }

        // Reserved
        assert_eq!(self.read_u8(), 0);
        assert_eq!(self.read_u16(), 0);

        let num_funcs = self.read_u32();
        let num_consts = self.read_u32();
        let num_recs = self.read_u32();

        let funcs = self.read_functions(num_funcs);
        let consts = self.read_consts(num_consts);

        // Check that the records match the sum of the expected records per function.
        assert_eq!(
            funcs.iter().map(|f| f.record_count).sum::<u64>(),
            u64::from(num_recs)
        );

        let mut recs = Vec::new();

        // Parse records.
        for f in &funcs {
            let mut records = self.read_records(f.record_count, &consts);
            for r in &mut records {
                r.offset += f.addr;
                r.size = f.stack_size;
            }
            recs.push(records);
        }

        // Read prologue info.
        let mut ps = self.read_prologue(num_funcs);

        // Collect all the information into `SMEntry`s.
        let mut smentries = Vec::new();
        for records in recs.into_iter().rev() {
            let pinfo = ps.pop().unwrap();
            smentries.push(SMEntry { pinfo, records });
        }
        Ok((smentries, num_recs))
    }

    fn read_functions(&mut self, num: u32) -> Vec<Function> {
        let mut v = Vec::new();
        for _ in 0..num {
            let addr = self.read_u64();
            let stack_size = self.read_u64();
            let record_count = self.read_u64();
            v.push(Function {
                addr,
                record_count,
                stack_size,
            });
        }
        v
    }

    fn read_consts(&mut self, num: u32) -> Vec<u64> {
        let mut v = Vec::new();
        for _ in 0..num {
            v.push(self.read_u64());
        }
        v
    }

    fn read_records(&mut self, num: u64, consts: &[u64]) -> Vec<Record> {
        let mut v = Vec::new();
        for _ in 0..num {
            let id = self.read_u64();
            let offset = u64::from(self.read_u32());
            self.read_u16();
            let num_live_vars = self.read_u16();
            let live_vars = self.read_live_vars(num_live_vars, consts);
            // Padding
            self.align_8();
            self.read_u16();
            let num_liveouts = self.read_u16();
            self.read_liveouts(num_liveouts);
            self.align_8();
            v.push(Record {
                id,
                offset,
                live_vars,
                size: 0,
            });
        }
        v
    }

    fn read_live_vars(&mut self, num: u16, consts: &[u64]) -> Vec<LiveVar> {
        let mut v = Vec::new();
        for _ in 0..num {
            let num_locs = self.read_u8();
            v.push(LiveVar {
                locs: self.read_locations(num_locs, consts),
            });
        }
        v
    }

    fn read_locations(&mut self, num: u8, consts: &[u64]) -> Vec<Location> {
        let mut v = Vec::new();
        for _ in 0..num {
            let kind = self.read_u8();
            self.read_u8();
            let size = self.read_u16();
            let dwreg = self.read_u16();
            let extrareg = self.read_u16();

            let location = match kind {
                0x01 => {
                    let offset = self.read_i32();
                    Location::Register(dwreg, size, offset, extrareg)
                }
                0x02 => {
                    let offset = self.read_i32();
                    Location::Direct(dwreg, offset, size)
                }
                0x03 => {
                    let offset = self.read_i32();
                    Location::Indirect(dwreg, offset, size)
                }
                0x04 => {
                    let offset = self.read_u32();
                    Location::Constant(offset)
                }
                0x05 => {
                    let offset = self.read_i32();
                    Location::LargeConstant(consts[usize::try_from(offset).unwrap()])
                }
                _ => unreachable!(),
            };

            v.push(location)
        }
        v
    }

    fn read_liveouts(&mut self, num: u16) {
        for _ in 0..num {
            let _dwreg = self.read_u16();
            let _size = self.read_u8();
        }
    }

    fn read_prologue(&mut self, num_funcs: u32) -> Vec<PrologueInfo> {
        let mut pis = Vec::new();
        for _ in 0..num_funcs {
            let hasfptr = self.read_u8();
            assert!(hasfptr == 0 || hasfptr == 1);
            self.read_u8(); // Padding
            let numspills = self.read_u32();

            let mut v = Vec::new();
            for _ in 0..numspills {
                let reg = self.read_u16();
                self.read_u16(); // Padding
                let off = self.read_i32();
                v.push((reg, off));
            }
            let pi = PrologueInfo {
                hasfp: hasfptr != 0,
                csrs: v,
            };
            pis.push(pi);
        }
        pis
    }

    fn align_8(&mut self) {
        self.offset += (8 - (self.offset % 8)) % 8;
    }

    fn read_u8(&mut self) -> u8 {
        let d = u8::from_ne_bytes(self.data[self.offset..self.offset + 1].try_into().unwrap());
        self.offset += 1;
        d
    }

    fn read_u16(&mut self) -> u16 {
        let d = u16::from_ne_bytes(self.data[self.offset..self.offset + 2].try_into().unwrap());
        self.offset += 2;
        d
    }

    fn read_u32(&mut self) -> u32 {
        let d = u32::from_ne_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        d
    }

    fn read_i32(&mut self) -> i32 {
        let d = i32::from_ne_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        d
    }

    fn read_u64(&mut self) -> u64 {
        let d = u64::from_ne_bytes(self.data[self.offset..self.offset + 8].try_into().unwrap());
        self.offset += 8;
        d
    }
}
