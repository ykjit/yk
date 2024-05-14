; Dump:
;   stdout:
;     ...
;     func main(...
;       bb0:
;         unimplemented <<  %{{4}} = getelementptr i32, <8 x ptr> %{{1}}, i32 1>>
;         ret
;     }

; This test ensures that as-yet unsupported variants of LLVM instructions are
; serialised as an unsupported instruction in the AOT IR. This prevents the JIT
; from silently miscompiling things we haven't yet thought about.

@arr = global [4 x i8] zeroinitializer

define void @main(ptr %ptr, <8 x ptr> %ptrs) {
geps:
  ; note `getelementptr inrange` cannot appear as a dedicated instruction, only
  ; as an inline expression. Hence no check for that in instruction form.
  %0 = getelementptr i32, <8 x ptr> %ptrs, i32 1
  ret void
}
