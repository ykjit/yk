; Dump:
;   stdout:
;     ...
;     func main() -> i1 {
;       bb0:
;         ret 0i1
;     }
;     ...

; Check that lowering non-byte-sized integer constants works.

define i1 @main() {
entry:
  ; Here we return a 1-bit constant integer.
  ret i1 0
}
