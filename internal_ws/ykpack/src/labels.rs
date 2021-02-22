use object::{Object, ObjectSection};
use std::{
    convert::TryFrom,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{SirLabel, YKLABELS_SECTION};

/// Splits a Yorick mapping label name into its constituent fields.
fn split_label_name(s: &str) -> (String, u32) {
    let data: Vec<&str> = s.split(':').collect();
    debug_assert!(data.len() == 3);
    let sym = data[1].to_owned();
    let bb_idx = data[2].parse::<u32>().unwrap();
    (sym, bb_idx)
}

/// Add a Yorick label section to the specified executable.
pub fn add_yk_label_section(exe_path: &Path) {
    let labels = extract_dwarf_labels(exe_path).unwrap();
    let mut tempf = tempfile::NamedTempFile::new().unwrap();
    bincode::serialize_into(&mut tempf, &labels).unwrap();
    add_section(exe_path, tempf.path());
}

/// Copies the bytes in `sec_data_path` into a new Yorick label section of an executable.
fn add_section(exe_path: &Path, sec_data_path: &Path) {
    let mut out_path = PathBuf::from(exe_path);
    out_path.set_extension("with_labels");
    Command::new("objcopy")
        .args(&[
            "--add-section",
            &format!("{}={}", YKLABELS_SECTION, sec_data_path.to_str().unwrap()),
            "--set-section-flags",
            &format!("{}=contents,alloc,readonly", YKLABELS_SECTION),
            exe_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to insert labels section");
    std::fs::rename(out_path, exe_path).unwrap();
}

/// Walks the DWARF tree of the specified executable and extracts Yorick location mapping
/// labels. Returns an list of labels ordered by file offset (ascending).
fn extract_dwarf_labels(exe_filename: &Path) -> Result<Vec<SirLabel>, gimli::Error> {
    let file = fs::File::open(exe_filename).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let object = object::File::parse(&*mmap).unwrap();
    let endian = if object.is_little_endian() {
        gimli::RunTimeEndian::Little
    } else {
        gimli::RunTimeEndian::Big
    };
    let loader = |id: gimli::SectionId| -> Result<&[u8], gimli::Error> {
        Ok(object
            .section_by_name(id.name())
            .map(|sec| sec.data().expect("failed to decompress section"))
            .unwrap_or(&[] as &[u8]))
    };
    let sup_loader = |_| Ok(&[] as &[u8]);
    let dwarf_cow = gimli::Dwarf::load(&loader, &sup_loader)?;
    let borrow_section: &dyn for<'a> Fn(&&'a [u8]) -> gimli::EndianSlice<'a, gimli::RunTimeEndian> =
        &|section| gimli::EndianSlice::new(section, endian);
    let dwarf = dwarf_cow.borrow(&borrow_section);
    let mut iter = dwarf.units();
    let mut subaddr = None;
    let mut labels = Vec::new();
    while let Some(header) = iter.next()? {
        let unit = dwarf.unit(header)?;
        let mut entries = unit.entries();
        while let Some((_, entry)) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                if let Some(_name) = entry.attr_value(gimli::DW_AT_linkage_name)? {
                    if let Some(lowpc) = entry.attr_value(gimli::DW_AT_low_pc)? {
                        if let gimli::AttributeValue::Addr(v) = lowpc {
                            // We can not accurately insert labels at the beginning of
                            // functions, because the label is offset by the function headers.
                            // We thus simply remember the subprogram's address so we can later
                            // assign it to the first block (ending with '_0') of this
                            // subprogram.
                            subaddr = Some(u64::try_from(v).unwrap());
                        } else {
                            panic!("Error reading dwarf information. Expected type 'Addr'.")
                        }
                    }
                }
            } else if entry.tag() == gimli::DW_TAG_label {
                if let Some(name) = entry.attr_value(gimli::DW_AT_name)? {
                    if let Some(es) = name.string_value(&dwarf.debug_str) {
                        let s = es.to_string()?;
                        if s.starts_with("__YK_") {
                            if let Some(lowpc) = entry.attr_value(gimli::DW_AT_low_pc)? {
                                if subaddr.is_some() && s.ends_with("_0") {
                                    // This is the first block of the subprogram. Assign its
                                    // label to the subprogram's address.
                                    let (fsym, bb) = split_label_name(s);
                                    labels.push(SirLabel {
                                        off: usize::try_from(subaddr.unwrap()).unwrap(),
                                        symbol_name: fsym,
                                        bb,
                                    });
                                    subaddr = None;
                                } else {
                                    let (fsym, bb) = split_label_name(s);
                                    if let gimli::AttributeValue::Addr(v) = lowpc {
                                        labels.push(SirLabel {
                                            off: usize::try_from(u64::try_from(v).unwrap())
                                                .unwrap(),
                                            symbol_name: fsym,
                                            bb,
                                        });
                                    } else {
                                        panic!(
                                                "Error reading dwarf information. Expected type 'Addr'."
                                            );
                                    }
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

    labels.sort_by_key(|l| l.off);
    Ok(labels)
}
