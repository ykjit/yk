//! Tests for the code generator's register allocator.

use crate::helpers::TestTypes;
use std::{collections::HashMap, convert::TryFrom};
use ykshim_client::{reg_pool_size, Local, LocalDecl, LocalIndex, TraceCompiler};

// Repeatedly fetching the register for the same local should yield the same register and
// should not exhaust the allocator.
#[test]
fn reg_alloc_same_local() {
    let types = TestTypes::new();
    let mut local_decls = HashMap::new();
    local_decls.insert(Local(0), LocalDecl::new(types.t_u8, false));
    local_decls.insert(Local(1), LocalDecl::new(types.t_i64, false));
    local_decls.insert(Local(2), LocalDecl::new(types.t_string, false));

    let mut tc = TraceCompiler::new(local_decls);
    let u8_loc = tc.local_to_location_str(Local(0));
    let i64_loc = tc.local_to_location_str(Local(1));
    let string_loc = tc.local_to_location_str(Local(2));
    dbg!(&u8_loc);
    for _ in 0..32 {
        assert_eq!(tc.local_to_location_str(Local(0)), u8_loc);
        assert_eq!(tc.local_to_location_str(Local(1)), i64_loc);
        assert_eq!(tc.local_to_location_str(Local(2)), string_loc);
        assert_eq!(tc.local_to_location_str(Local(1)), i64_loc);
        assert_eq!(tc.local_to_location_str(Local(2)), string_loc);
        assert_eq!(tc.local_to_location_str(Local(0)), u8_loc);
    }
}

// Locals should be allocated to different registers.
#[test]
fn reg_alloc() {
    let types = TestTypes::new();
    let mut local_decls = HashMap::new();
    for i in (0..9).step_by(3) {
        local_decls.insert(Local(i + 0), LocalDecl::new(types.t_u8, false));
        local_decls.insert(Local(i + 1), LocalDecl::new(types.t_i64, false));
        local_decls.insert(Local(i + 2), LocalDecl::new(types.t_string, false));
    }

    let mut tc = TraceCompiler::new(local_decls);
    let mut seen: Vec<String> = Vec::new();
    for l in 0..7 {
        let reg = tc.local_to_location_str(Local(l));
        assert!(!seen.contains(&reg));
        seen.push(reg);
    }
}

// Once registers are full, the allocator should start spilling.
#[test]
fn reg_alloc_spills() {
    let types = TestTypes::new();
    let num_regs = reg_pool_size() + 1; // Plus one for ICTX_REG.
    let num_spills = 16;
    let num_decls = num_regs + num_spills;
    let mut local_decls = HashMap::new();
    for i in 0..num_decls {
        local_decls.insert(
            Local(u32::try_from(i).unwrap()),
            LocalDecl::new(types.t_u8, false),
        );
    }

    let mut tc = TraceCompiler::new(local_decls);
    for l in 0..num_regs {
        assert!(tc
            .local_to_location_str(Local(LocalIndex::try_from(l).unwrap()))
            .starts_with("Reg("));
    }

    for l in num_regs..num_decls {
        assert!(tc
            .local_to_location_str(Local(LocalIndex::try_from(l).unwrap()))
            .starts_with("Mem("));
    }
}

// Freeing registers should allow them to be re-allocated.
#[test]
fn reg_alloc_spills_and_frees() {
    let types = TestTypes::new();
    let num_regs = reg_pool_size() + 1; // Plus one for ICTX_REG.
    let num_decls = num_regs + 4;
    let mut local_decls = HashMap::new();
    for i in 0..num_decls {
        local_decls.insert(
            Local(u32::try_from(i).unwrap()),
            LocalDecl::new(types.t_u8, false),
        );
    }

    let mut tc = TraceCompiler::new(local_decls);

    // Fill registers.
    for l in 0..num_regs {
        assert!(tc
            .local_to_location_str(Local(LocalIndex::try_from(l).unwrap()))
            .starts_with("Reg("));
    }

    // Allocating one more local should spill.
    assert!(tc
        .local_to_location_str(Local(LocalIndex::try_from(num_regs).unwrap()))
        .starts_with("Mem("));

    // Now let's free two locals previously given a register.
    tc.local_dead(Local(3));
    tc.local_dead(Local(5));

    // Allocating two more locals should therefore yield register locations.
    assert!(tc
        .local_to_location_str(Local(LocalIndex::try_from(num_regs + 1).unwrap()))
        .starts_with("Reg("));
    assert!(tc
        .local_to_location_str(Local(LocalIndex::try_from(num_regs + 2).unwrap()))
        .starts_with("Reg("));

    // And one more should spill again.
    assert!(tc
        .local_to_location_str(Local(LocalIndex::try_from(num_regs + 3).unwrap()))
        .starts_with("Mem("));
}

// Test cases where a local is allocated on the stack even if registers are available.
#[test]
fn reg_alloc_always_on_stack() {
    let types = TestTypes::new();
    let mut local_decls = HashMap::new();

    // In a TIR trace, the first two decls are a unit and the interpreter context, which are
    // handled specially. We populate their slots so that we can acquire regular locals with no
    // special casing.
    assert!(reg_pool_size() >= 3); // Or we'd spill regardless.
    for i in 0..=1 {
        local_decls.insert(
            Local(u32::try_from(i).unwrap()),
            LocalDecl::new(types.t_u8, false),
        );
    }

    // These are the decls we will actually test.
    local_decls.insert(Local(2), LocalDecl::new(types.t_string, false));
    local_decls.insert(Local(3), LocalDecl::new(types.t_u8, true));
    local_decls.insert(Local(4), LocalDecl::new(types.t_tiny_struct, false));
    local_decls.insert(Local(5), LocalDecl::new(types.t_tiny_array, false));
    local_decls.insert(Local(6), LocalDecl::new(types.t_tiny_tuple, false));

    let mut tc = TraceCompiler::new(local_decls);

    // Things larger than a register shouldn't be allocated to a register. Here we are
    // allocating space for a `String`, which at the time of writing, is much larger than the
    // size of a register on any platform I can think of (e.g. String is 24 bytes on x86_64).
    dbg!(tc.local_to_location_str(Local(2)));
    assert!(tc.local_to_location_str(Local(2)).starts_with("Mem("));
    tc.local_dead(Local(2));

    // Small types, like `u8`, can easily fit in a register, but if a local decl is referenced
    // at some point in its live range then the allocator will put it directly onto the stack
    // (even if registers are available).
    assert!(tc.local_to_location_str(Local(3)).starts_with("Mem("));
    tc.local_dead(Local(3));

    // A one-byte struct/enum/array/tuple can easily fit in a register, but for simplicity the
    // code-gen currently allocates these types directly to the stack. This is not what we want
    // in the long-run, but it's an invariant that should be tested nonetheless. FIXME.
    //
    // FIXME enum case cannot yet be tested as its layout isn't yet lowered.
    for l in 4..=6 {
        assert!(tc.local_to_location_str(Local(l)).starts_with("Mem("));
        tc.local_dead(Local(l));
    }
}
