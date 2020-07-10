use object::Object;
use phdrs::objects;

use crate::SirLoc;
use hwtracer::Trace;
use lazy_static::lazy_static;
use std::{borrow, env, fs};

lazy_static! {
    /// Maps a label address to its symbol name and block index.
    ///
    /// We use a vector here since we never actually look up entries by address; we only iterate
    /// over the labels checking if each address is within the range of a block.
    ///
    /// The labels are the same for each trace, and they are immutable, so it makes sense for this
    /// to be a lazy static, loaded only once and shared.
    ///
    /// FIXME if we want to support dlopen(), we will have to rethink this.
    static ref LABELS: Vec<(u64, (String, u32))> = extract_labels().unwrap();
}

pub struct HWTMapper {
    phdr_offset: u64
}

impl HWTMapper {
    pub fn new() -> HWTMapper {
        let phdr_offset = get_phdr_offset();
        HWTMapper { phdr_offset }
    }

    /// Maps each entry of a hardware trace to the appropriate SirLoc.
    pub fn map(&self, trace: Box<dyn Trace>) -> Option<Vec<SirLoc>> {
        let mut annotrace = Vec::new();
        for b in trace.iter_blocks() {
            match b {
                Ok(block) => {
                    let start_addr = block.first_instr() - self.phdr_offset;
                    let end_addr = block.last_instr() - self.phdr_offset;
                    // Each block reported by the hardware tracer corresponds to one or more SIR
                    // blocks, so we collect them in a vector here. This is safe because:
                    //
                    // a) We know that the SIR blocks were compiled (by LLVM) to straight-line
                    // code, otherwise a control-flow instruction would have split the code into
                    // multiple PT blocks.
                    //
                    // b) `labels` is sorted, so the blocks will be appended to the trace in the
                    // correct order.
                    let mut locs = Vec::new();
                    for (addr, (sym, bb_idx)) in &*LABELS {
                        if *addr >= start_addr && *addr <= end_addr {
                            // Found matching label.
                            locs.push((addr, SirLoc::new(sym.to_string(), *bb_idx)));
                        } else if *addr > end_addr {
                            // `labels` is sorted by address, so once we see one with an address
                            // higher than `end_addr`, we know there can be no further hits.
                            break;
                        }
                    }
                    annotrace.extend(
                        locs.into_iter()
                            .map(|(_, loc)| loc)
                            .collect::<Vec<SirLoc>>()
                    );
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
///
/// The returned vector is sorted by label address ascending.
fn extract_labels() -> Result<Vec<(u64, (String, u32))>, gimli::Error> {
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
    let mut labels = Vec::new();

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
                                    labels.push((subaddr.unwrap(), split_symbol(s)));
                                    subaddr = None;
                                } else {
                                    labels.push((addr, split_symbol(s)));
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

    labels.sort_by_key(|k| k.0);
    Ok(labels)
}

fn split_symbol(s: &str) -> (String, u32) {
    let data: Vec<&str> = s.split(':').collect();
    debug_assert!(data.len() == 3);
    let sym = data[1].to_owned();
    let bb_idx = data[2].parse::<u32>().unwrap();
    (sym, bb_idx)
}
