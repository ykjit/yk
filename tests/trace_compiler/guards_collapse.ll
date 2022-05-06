; Run-time-filtered:
;   env-var: YKD_PRINT_IR=jit-pre-opt,jit-post-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1,main:2
;   stdout:
;     --- Begin icmps ---
;     --- Begin jit-pre-opt ---
;       %{{pre-icmp1}} = icmp slt i32 %{{pre-v1}}, 999
;       %{{pre-icmp2}} = icmp slt i32 %{{pre-v2}}, 1000
;     --- End jit-pre-opt ---
;     --- Begin jit-post-opt ---
;       %{{post-icmp1}} = icmp slt i32 %{{post-v1}}, 999
;     --- End jit-post-opt ---
;     --- End icmps ---
;
;     --- Begin guard-fail-brs ---
;     --- Begin jit-pre-opt ---
;       br i1 %{{pre-icmp1}}, label %{{pre-true1}}, label %guardfail
;       br i1 %{{pre-icmp2}}, label %{{pre-true2}}, label %guardfail1
;     --- End jit-pre-opt ---
;     --- Begin jit-post-opt ---
;       br i1 %{{post-icmp1}}, label %{{post-true1}}, label %guardfail
;     --- End jit-post-opt ---
;     --- End guard-fail-brs ---

@g = global i32 5;

define void @main() {
entry:
    %v = load volatile i32 , i32 * @g
    %cond1 = icmp slt i32 %v, 999
    br i1 %cond1, label %true1, label %falses

true1:
    %cond2 = icmp slt i32 %v, 1000
    br i1 %cond2, label %true2, label %falses

true2:
    unreachable

falses:
    unreachable
}
