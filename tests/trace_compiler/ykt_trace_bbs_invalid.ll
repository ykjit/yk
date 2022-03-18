; Run-time:
;   env-var: YKT_TRACE_BBS=nonsense
;   status: error
;   stderr:
;     Error: "YKT_TRACE_BBS is malformed"

; Check the test harness gives an appropriate error message if the
; YKT_TRACE_BBS environment is malformed.

define void @main() {
entry:
    unreachable
}
