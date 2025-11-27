; Dump:
;   stdout:
;     ...
;     func main(%arg0: ptr) {
;       bb0:
;         %0_0: ptr = arg(0)
;         %0_1: i8 = load %0_0
;         %0_2: i8 = load %0_0, volatile
;         %0_3: i8 = load %0_0
;         %0_4: i8 = load %0_0
;         %0_5: i16 = load %0_0
;         *%0_0 = 0i8
;         *%0_0 = 0i8, volatile
;         *%0_0 = 0i8
;         *%0_0 = 0i8
;         unimplemented <<  store i16 0, ptr %0, align 1>>
;         ret
;     }
;     ...

; Check that loads and stores lower OK.

define void @main(ptr %p) optnone noinline {
entry:
  %l0 = load i8, ptr %p
  %l1 = load volatile i8, ptr %p
  %l2 = load i8, ptr %p, align 1
  %l3 = load i8, ptr %p, align 8
  %l4 = load i16, ptr %p, align 1 ; potentially misaligned
  store i8 0, ptr %p
  store volatile i8 0, ptr %p
  store i8 0, ptr %p, align 1
  store i8 0, ptr %p, align 8
  store i16 0, ptr %p, align 1 ; potentially misaligned
  ret void
}
