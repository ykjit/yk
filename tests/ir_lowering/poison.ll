; Dump:
;   stdout:
;     ...
;     func main(...
;       bb0:
;         %0_0: i32 = add poison<i32>, 1i32
;         ret
;     }
;     ...

; Check that poison lowers properly.

define void @main() optnone noinline {
entry:
    %0 = add i32 poison, 1
    ret void
}
