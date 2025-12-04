; Dump:
;   stdout:
;     ...
;     func main() -> ptr {
;       bb0:
;         ret 0x0
;     }
;     ...

; Check null pointer constants lower OK.

define ptr @main() {
  ret ptr null
}
