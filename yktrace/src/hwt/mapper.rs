use object::Object;
use phdrs::objects;

use crate::SirLoc;
use hwtracer::Trace;
use std::{borrow, collections::HashMap, env, fs};

pub struct HWTMapper {
    phdr_offset: u64,
    labels: Option<HashMap<u64, (String, u32)>>
}

impl HWTMapper {
    pub fn new() -> HWTMapper {
        let phdr_offset = get_phdr_offset();
        let labels = match extract_labels() {
            Ok(l) => Some(l),
            Err(_) => None
        };
        HWTMapper {
            phdr_offset,
            labels
        }
    }

    /// Maps each entry of a hardware trace to the appropriate SirLoc.
    pub fn map(&self, trace: Box<dyn Trace>) -> Option<Vec<SirLoc>> {
        if !self.labels.is_some() {
            return None;
        }
        let labels = self.labels.as_ref().unwrap();
        let mut annotrace = Vec::new();
        for b in trace.iter_blocks() {
            match b {
                Ok(block) => {
                    let start_addr = block.start_vaddr() - self.phdr_offset;
                    let end_addr = start_addr + block.len();
                    // XXX check if there exists a label that is within addr and addr + block.len()
                    for (addr, (sym, bb_idx)) in labels {
                        if addr >= &start_addr && addr < &end_addr {
                            // found matching label
                            annotrace.push(SirLoc::new(sym.to_string(), *bb_idx));
                            break
                        }
                    }
                    //match labels.get(&start_addr) {
                    //    Some(l) => {
                    //        // FIXME Do the splitting at the time we load from DWARF.
                    //        let data: Vec<&str> = l.split(':').collect();
                    //        debug_assert!(data.len() == 3);
                    //        let sym = String::from(data[1]);
                    //        let bb_idx = data[2].parse::<u32>().unwrap();
                    //        annotrace.push(SirLoc::new(sym, bb_idx))
                    //    }
                    //    None => {}
                    //};
                }
                Err(_) => {}
            }
        }
        Some(annotrace)
    }
}

/// Extract the program header offset. This offset can be used to translate the address of a trace
/// block to a program address, allowing us to find the correct SIR location.
fn get_phdr_offset() -> u64 {
    (&objects()[0]).addr() as u64
}

/// Extracts YK debug labels and their addresses from the executable.
fn extract_labels() -> Result<HashMap<u64, (String, u32)>, gimli::Error> {
    // Load executable
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(&pathb.as_path()).unwrap();
    let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
    let object = object::File::parse(&*mmap).unwrap();
    let endian = if object.is_little_endian() {
        gimli::RunTimeEndian::Little
    } else {
        gimli::RunTimeEndian::Big
    };

    // Extract labels
    let mut labels = HashMap::new();

    let loader = |id: gimli::SectionId| -> Result<borrow::Cow<[u8]>, gimli::Error> {
        Ok(object
            .section_data_by_name(id.name())
            .unwrap_or(borrow::Cow::Borrowed(&[][..])))
    };
    let sup_loader = |_| Ok(borrow::Cow::Borrowed(&[][..]));
    let dwarf_cow = gimli::Dwarf::load(&loader, &sup_loader)?;
    let borrow_section: &dyn for<'a> Fn(
        &'a borrow::Cow<[u8]>
    ) -> gimli::EndianSlice<'a, gimli::RunTimeEndian> =
        &|section| gimli::EndianSlice::new(&*section, endian);
    let dwarf = dwarf_cow.borrow(&borrow_section);
    let mut iter = dwarf.units();
    let mut subaddr = None;
    while let Some(header) = iter.next()? {
        let unit = dwarf.unit(header)?;
        let mut entries = unit.entries();
        while let Some((_, entry)) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                if let Some(_name) = entry.attr_value(gimli::DW_AT_linkage_name)? {
                    if let Some(lowpc) = entry.attr_value(gimli::DW_AT_low_pc)? {
                        let addr = match lowpc {
                            gimli::AttributeValue::Addr(v) => v as u64,
                            _ => panic!("Error reading dwarf information. Expected type 'Addr'.")
                        };
                        // We can not accurately insert labels at the beginning of functions,
                        // because the label is offset by the function headers. We thus simply
                        // remember the subprogram's address so we can later assign it to the first
                        // block (ending with '_0') of this subprogram.
                        subaddr = Some(addr);
                    }
                }
            } else if entry.tag() == gimli::DW_TAG_label {
                if let Some(name) = entry.attr_value(gimli::DW_AT_name)? {
                    if let Some(es) = name.string_value(&dwarf.debug_str) {
                        let s = es.to_string()?;
                        if s.starts_with("__YK_") {
                            if let Some(lowpc) = entry.attr_value(gimli::DW_AT_low_pc)? {
                                let addr = match lowpc {
                                    gimli::AttributeValue::Addr(v) => v as u64,
                                    _ => panic!(
                                        "Error reading dwarf information. Expected type 'Addr'."
                                    )
                                };
                                if subaddr.is_some() && s.ends_with("_0") {
                                    // This is the first block of the subprogram. Assign its label
                                    // to the subprogram's address.
                                    labels.insert(subaddr.unwrap(), split_symbol(s));
                                    subaddr = None;
                                } else {
                                    labels.insert(addr, split_symbol(s));
                                }
                            } else {
                                // Ignore labels that have no address.
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(labels)
}

fn split_symbol(s: &str) -> (String, u32) {
    let data: Vec<&str> = s.split(':').collect();
    debug_assert!(data.len() == 3);
    let sym = String::from(data[1]);
    let bb_idx = data[2].parse::<u32>().unwrap();
    (sym, bb_idx)
}
