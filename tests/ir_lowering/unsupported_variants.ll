; Dump:
;   stdout:
;     ...
;     func main(...
;       bb0:
;         unimplemented <<  %{{4}} = getelementptr i32, <8 x ptr> %{{1}}, i32 1>>
;         br bb1
;       bb1:
;         unimplemented <<  %{{6}} = alloca inalloca i32, align 4>>
;         unimplemented <<  %{{7}} = alloca i32, align 4, addrspace(4)>>
;         unimplemented <<  %{{8}} = alloca i32, i32 %2, align 4>>
;         br bb2
;      bb2:
;         unimplemented <<  %{{13}} = fadd nnan float %{{3}}, %{{3}}>>
;         unimplemented <<  %{{14}} = udiv exact i32 %{{2}}, 1>>
;         unimplemented <<  %{{15}} = add <4 x i32> %{{44}}, %{{44}}>>
;         br bb3
;      bb3:
;         unimplemented <<  %{{17}} = call i32 @f(i32 swiftself 5)>>
;         unimplemented <<  %{{18}} = call inreg i32 @f(i32 5)>>
;         unimplemented <<  %{{19}} = call i32 @f(i32 5) #{{0}}>>
;         unimplemented <<  %{{20}} = call nnan float @g()>>
;         unimplemented <<  %{{21}} = call ghccc i32 @f(i32 5)>>
;         unimplemented <<  %{{22}} = call i32 @f(i32 5) [ "kcfi"(i32 1234) ]>>
;         unimplemented <<  %{{23}} = call addrspace(6) ptr @p()>>
;         br bb4
;      bb4:
;         unimplemented <<  %{{25}} = ptrtoint ptr %{{ptr}} to i8>>
;         unimplemented <<  %{{26}} = ptrtoint <8 x ptr> %{{ptrs}} to <8 x i8>>>
;         br bb5
;     bb5:
;         unimplemented <<  %27 = icmp ne ptr %0, null>>
;         unimplemented <<  %28 = icmp ne <4 x i32> %4, zeroinitializer>>
;         ret
;     }
;     ...

; This test ensures that as-yet unsupported variants of LLVM instructions are
; serialised as an unsupported instruction in the AOT IR. This prevents the JIT
; from silently miscompiling things we haven't yet thought about.

@arr = global [4 x i8] zeroinitializer

define i32 @f(i32 %num) {
    ret i32 5
}

define float @g() {
    ret float 5.5
}

define ptr @p() addrspace(6) {
    ret ptr null
}

declare void @llvm.experimental.stackmap(i64, i32, ...);

define void @main(ptr %ptr, <8 x ptr> %ptrs, i32 %num, float %flt, <4 x i32> %vecnums) {
geps:
  ; note `getelementptr inrange` cannot appear as a dedicated instruction, only
  ; as an inline expression. Hence no check for that in instruction form.
  %0 = getelementptr i32, <8 x ptr> %ptrs, i32 1
  br label %allocas
allocas:
  %1 = alloca inalloca i32
  %2 = alloca i32, addrspace(4)
  %3 = alloca i32, i32 %num
  ; Note that we don't test alloca's with number of elements not expressible in
  ; a `size_t`. At the time of writing using a type wider than i64 for the
  ; element count can crash selection dag.
  ; e.g.: `%blah = alloca i32, i66 36893488147419103232`
  br label %binops
binops:
  %4 = fadd nnan float %flt, %flt
  %5 = udiv exact i32 %num, 1
  %6 = add <4 x i32> %vecnums, %vecnums
  br label %calls
calls:
  ; FIXME: we are unable to test `musttail` because a tail call must be
  ; succeeded by either a `ret` or a `bitcast` and then a `ret`. But the JIT
  ; requires a stackmap after a call...
  ;
  ; param attrs
  %7 = call i32 @f(i32 swiftself 5)
  ; ret attrs
  %8 = call inreg i32 @f(i32 5)
  ; func attrs
  %9 = call i32 @f(i32 5) alignstack(8)
  ; fast math flags
  %10 = call nnan float @g()
  ; Non-C calling conventions.
  %11 = call ghccc i32 @f(i32 5)
  ; operand bundles
  %12 = call i32 @f(i32 5) ["kcfi"(i32 1234)]
  ; non-zero address spaces
  %13 = call addrspace(6) ptr @p()
  ; stackmap required (but irrelevant for the test) for all of the above calls.
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 7, i32 0);
  br label %casts
casts:
  %14 = ptrtoint ptr %ptr to i8
  %15 = ptrtoint <8 x ptr> %ptrs to <8 x i8>
  br label %icmps
icmps:
  ; pointer comparison
  %16 = icmp ne ptr %ptr, null
  ; vector of comparisons
  %17 = icmp ne <4 x i32> %vecnums, zeroinitializer
  ; stackmap stops icmps from being optimised out.
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 8, i32 0, i1 %16, <4 x i1> %17);
  ret void
}
