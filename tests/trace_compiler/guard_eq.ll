; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal {{type}} @__yk_compiled_trace_0(...
;        %{{0}} = alloca i32, align 4
;        %{{1}} = icmp eq i32 1, 1
;        br i1 %{{1}}, label %{{true}}, label %guardfail
;
;      guardfail:...
;        ...
;        %{{cprtn}} = call {{type}} (...) @llvm.experimental.deoptimize.{{type}}(...
;        ret {{type}} %{{cprtn}}
;
;      {{true}}:...
;        store i32 1, i32* %{{0}}, align 4
;        ret {{type}} 0
;      }
;
;      declare {{type}} @llvm.experimental.deoptimize.{{type}}(...)
;      --- End jit-pre-opt ---

define void @main() {
entry:
    %0 = alloca i32
    %1 = icmp eq i32 1, 1
    br i1 %1, label %true, label %false

true:
    store i32 1, i32 * %0
    unreachable

false:
    store i32 0, i32 * %0
    unreachable
}
