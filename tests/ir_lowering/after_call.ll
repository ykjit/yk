; Dump:
;   stdout:
;     ...
;     func main($arg0: i32, $arg1: ptr) -> i32 {
;       bb0:
;         $0_0: i32 = call f($arg0)
;         $0_1: i1 = icmp $arg0, 1i32
;         condbr $0_1, bb2, bb1
;     ...


; Check that a instructions following a call are correctly lowered.

define i32 @f(i32 %0) noinline {
  ret i32 %0
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32 @f(i32 %argc)
  %1 = icmp eq i32 %argc, 1
  br i1 %1, label %bb1, label %bb2

bb1:
    ret i32 0

bb2:
    ret i32 1
}
