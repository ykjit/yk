; Dump:
;   stdout:
;     ...
;     func main($arg0: i32, $arg1: ptr) -> i32 {
;       bb0:
;         $0_0: ptr = alloca i32, 1i32
;         store 1i32, $0_0
;         $0_2: i1 = icmp $arg0, 1i32
;         condbr $0_2, bb2, bb1
;     ...


; Check that a instructions following a call are correctly lowered.

define i32 @f(i32 %0) noinline {
  ret i32 %0
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = alloca i32
  store i32 1, ptr %0
  %1 = icmp eq i32 %argc, 1
  br i1 %1, label %bb1, label %bb2

bb1:
    ret i32 0

bb2:
    ret i32 1
}
