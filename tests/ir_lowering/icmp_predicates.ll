; Dump:
;   stdout:
;     ...
;       bb0:
;         $0_0: i1 = icmp $arg0, eq, $arg1
;         $0_1: i1 = icmp $0_0, ne, $arg1
;         $0_2: i1 = icmp $0_0, ult, $0_1
;         $0_3: i1 = icmp $0_1, ule, $0_2
;         $0_4: i1 = icmp $0_2, ugt, $0_3
;         $0_5: i1 = icmp $0_3, uge, $0_4
;         $0_6: i1 = icmp $0_4, slt, $0_5
;         $0_7: i1 = icmp $0_5, sle, $0_6
;         $0_8: i1 = icmp $0_6, sgt, $0_7
;         $0_9: i1 = icmp $0_7, sge, $0_8
;         ret $0_9
;     ...

; Check that icmp predicates lower correctly.

define i1 @main(i1 %op1, i1 %op2) {
entry:
  %0 = icmp eq i1 %op1, %op2
  %1 = icmp ne i1 %0, %op2
  %2 = icmp ult i1 %0, %1
  %3 = icmp ule i1 %1, %2
  %4 = icmp ugt i1 %2, %3
  %5 = icmp uge i1 %3, %4
  %6 = icmp slt i1 %4, %5
  %7 = icmp sle i1 %5, %6
  %8 = icmp sgt i1 %6, %7
  %9 = icmp sge i1 %7, %8
  ret i1 %9
}
