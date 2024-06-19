; Dump:
;   stdout:
;     ...
;     func main(%arg0: i32, %arg1: ptr) -> i32 {
;       bb0:
;         %0_0: i32 = arg(0)
;         %0_1: ptr = arg(1)
;         %0_2: i32 = call f(%0_0) [safepoint: 1i64, ()]
;         %0_3: i1 = eq %0_0, 1i32
;         condbr %0_3, bb1, bb2 [safepoint: 2i64, ()]
;     ...

; Check that a instructions following a call are correctly lowered.

declare void @llvm.experimental.stackmap(i64, i32, ...);

define i32 @f(i32 %0) noinline {
  ret i32 %0
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32 @f(i32 %argc)
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0);
  %1 = icmp eq i32 %argc, 1
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 2, i32 0);
  br i1 %1, label %bb1, label %bb2

bb1:
    ret i32 0

bb2:
    ret i32 1
}
