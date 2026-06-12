use std::ffi::c_void;

/// What is the relationship between `frame1` and `frame2`?
///
/// This assumes that you are passing in a stable address per frame (e.g. `rbp` on x64). If you
/// pass two addresses that are e.g. for two variables in the same frame, from frames in different
/// threads, etc, bad things will happen.
#[cfg(target_arch = "x86_64")]
pub(crate) fn frame_relationship(
    frame1: *const c_void,
    frame2: *const c_void,
) -> FrameRelationship {
    if frame1.addr() > frame2.addr() {
        FrameRelationship::Frame1IsCallerOfFrame2
    } else if frame2.addr() > frame1.addr() {
        FrameRelationship::Frame2IsCallerOfFrame1
    } else {
        FrameRelationship::SameFrame
    }
}

#[derive(PartialEq)]
pub(crate) enum FrameRelationship {
    Frame1IsCallerOfFrame2,
    Frame2IsCallerOfFrame1,
    SameFrame,
}
