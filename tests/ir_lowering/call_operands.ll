; Dump:
;   stdout:
;     ...
;     func main() -> i32 {
;       bb0:
;         call f(1i32, 2i32, 3i32) [safepoint: 1i64, ()]
;         ret 0i32
;     }
;     ...

; Check a call instruction lowers and prints correctly.
;
; The reason for this is that in LLVM's IR data structures the call target is
; the last operand, but in Yk's IR it's the first. A generic lowering would do
; the wrong thing.

declare void @llvm.experimental.stackmap(i64, i32, ...);

define void @f(i32 %0, i32 %1, i32 %2) { ret void }

define i32 @main() {
entry:
  call void @f(i32 1, i32 2, i32 3);
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0);
  ret i32 0
}
