; Dump:
;   stdout:
;     ...
;     func main(...
;       bb0:
;         $0_0: ptr = PtrAdd @arr0, 3
;         $0_1: ptr = PtrAdd @arr1, 16
;         $0_2: ptr = PtrAdd @arr2, 32
;         $0_3: ptr = PtrAdd @arr3, 96
;         $0_4: ptr = PtrAdd @arr4, 12
;	    ...
;       bb1:
;         $1_0: ptr = PtrAdd @arr0, 1
;         $1_1: ptr = PtrAdd @arr1, 4
;         $1_2: ptr = PtrAdd @arr2, 8
;         $1_3: ptr = PtrAdd @arr3, 24
;         $1_4: ptr = PtrAdd @arr4, 3
;	    ...
;       bb2:
;         $2_0: ptr = PtrAdd @mdarr0, 3
;         $2_1: ptr = PtrAdd @mdarr1, 320
;       ...
;       bb3:
;         $3_0: ptr = PtrAdd @arr0, 0 + ($arg0 * 3)
;         $3_1: ptr = PtrAdd @arr0, 0 + ($arg0 * 3) + ($arg1 * 1)
;         $3_2: ptr = PtrAdd @mdarr0, 1 + ($arg0 * 2)
;       ...
;       bb4:
;         $4_0: ptr = PtrAdd @struct0, 8
;         $4_1: ptr = PtrAdd @struct0, 4
;         $4_2: ptr = PtrAdd @struct1, 24
;         $4_3: ptr = PtrAdd @struct1, 16
;       ...
;       bb5:
;         $5_0: ptr = PtrAdd @mixed0, 12
;         $5_1: ptr = PtrAdd @mixed0, 13
;         $5_2: ptr = PtrAdd @mixed0, 5 + ($arg0 * 8)
;         $5_3: ptr = PtrAdd @mixed1, 10
;         $5_4: ptr = PtrAdd @mixed1, 8 + ($arg0 * 8) + ($arg1 * 1)
;         ret
;     }
;     ...

; Check that GEP lowers properly.

; arrays.
@arr0 = global [3 x i8] zeroinitializer
@arr1 = global [4 x i16] zeroinitializer
@arr2 = global [4 x i32] zeroinitializer
@arr3 = global [4 x i64] zeroinitializer
@arr4 = global [4 x i7] zeroinitializer
@mdarr0 = global [3 x [2 x i8]] zeroinitializer
@mdarr1 = global [4 x [4 x [3 x i64]]] zeroinitializer

; structs
@struct0 = global {i8, i32, i32} zeroinitializer
@struct1 = global {i8, i8, i64, {i64, i8}} zeroinitializer

; mixed
@mixed0 = global [4 x {i32, i8, i8, i8, i8}] zeroinitializer
@mixed1 = global {[4 x i8], [2 x {i32, [4 x i8]}]} zeroinitializer

define void @main(i32 %arg0, i32 %arg1, ptr %argv) optnone noinline {
entry:
  ; One-index array indexing.
  ;
  ; The first array index steps in whole-array-sized chunks (past the end of
  ; the array if nonzero).
  %0 = getelementptr [3 x i8], ptr @arr0, i32 1
  %1 = getelementptr [4 x i16], ptr @arr1, i32 2
  %2 = getelementptr [4 x i32], ptr @arr2, i32 2
  %3 = getelementptr [4 x i64], ptr @arr3, i32 3
  %4 = getelementptr [4 x i7], ptr @arr4, i32 3
  br label %bb1
bb1:
  ; >1 index array indexing.
  ;
  ; When the first array index is 0, you index into the elements of the array.
  %5 = getelementptr [3 x i8], ptr @arr0, i32 0, i32 1
  %6 = getelementptr [4 x i16], ptr @arr1, i32 0, i32 2
  %7 = getelementptr [4 x i32], ptr @arr2, i32 0, i32 2
  %8 = getelementptr [4 x i64], ptr @arr3, i32 0, i32 3
  %9 = getelementptr [4 x i7], ptr @arr4, i32 0, i32 3
  br label %bb2
bb2:
  ; Indexing into multi-dimensional arrays.
  %10 = getelementptr  [3 x [2 x i8]], ptr @mdarr0, i32 0, i32 1, i32 1
  %11 = getelementptr  [4 x [4 x [3 x i64]]], ptr @mdarr1, i32 0, i32 3, i32 1, i32 1
  br label %bb3
bb3:
  ; dynamic array indexing.
  %12 = getelementptr [3 x i8], ptr @arr0, i32 %arg0
  %13 = getelementptr [3 x i8], ptr @arr0, i32 %arg0, i32 %arg1
  %14 = getelementptr [3 x [2 x i8]], ptr @mdarr0, i32 0, i32 %arg0, i32 1
  br label %bb4
bb4:
  ; struct field accesses
  %15 = getelementptr {i8, i32, i32}, ptr @struct0, i32 0, i32 2
  %16 = getelementptr {i8, i32, i32}, ptr @struct0, i32 0, i32 1
  %17 = getelementptr {i8, i8, i64, {i64,  i8}}, ptr @struct1, i32 0, i32 3, i32 1
  %18 = getelementptr {i8, i8, i64, {i64,  i8}}, ptr @struct1, i32 0, i32 3, i32 0
  br label %bb5
bb5:
  ; mixed array index and struct field accesses.
  %19 = getelementptr [4 x {i32, i8, i8, i8, i8}], ptr @mixed0, i32 0, i32 1, i32 1
  %20 = getelementptr [4 x {i32, i8, i8, i8, i8}], ptr @mixed0, i32 0, i32 1, i32 2
  %21 = getelementptr [4 x {i32, i8, i8, i8, i8}], ptr @mixed0, i32 0, i32 %arg0, i32 2
  %22 = getelementptr {[4 x i8], [2 x {i32, [4 x i8]}]}, ptr @mixed1, i32 0, i32 1, i32 0, i32 1, i32 2
  %23 = getelementptr {[4 x i8], [2 x {i32, [4 x i8]}]}, ptr @mixed1, i32 0, i32 1, i32 %arg0, i32 1, i32 %arg1
  ret void
}
