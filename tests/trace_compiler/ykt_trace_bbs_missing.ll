; Run-time:
;   status: error
;   stderr:
;     Error: "The test doesn't set the YKT_TRACE_BBS environment variable"

; Check the test harness gives an appropriate error message if the
; YKT_TRACE_BBS environment wasn't set by the test.

define void @main() {
entry:
    unreachable
}
