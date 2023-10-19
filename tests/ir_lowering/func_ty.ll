; Dump:
;   stdout:
;     ...
;     func main($arg0: i32, $arg1: ptr) -> i32 {
;     ...

; Check function type lowering.

define i32 @main(i32 %argc, ptr %argv) {
    ret i32 1
}
