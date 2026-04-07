local x = 0
for _ = 0, 500 do
  x = x + 1
  for _ = 0, 500 do
    x = x + 1
  end
end
io.stderr:write(x)

-- Run-time:
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-tracing: start-tracing: nested_loops.lua:4: FORLOOP
--     yk-tracing: stop-tracing: nested_loops.lua:4: FORLOOP
--     --- Begin debugstrs: nested_loops.lua:4: FORLOOP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     nested_loops.lua:4: FORLOOP
--     nested_loops.lua:5: ADDI
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(0) ...
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(0) ...
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(0) ...
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(0) ...
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(0) ...
--     yk-tracing: start-side-tracing: nested_loops.lua:4: FORLOOP
--     yk-tracing: stop-tracing: nested_loops.lua:2: FORLOOP
--     yk-tracing: start-tracing: nested_loops.lua:2: FORLOOP
--     yk-tracing: stop-tracing: nested_loops.lua:4: FORLOOP
--     --- Begin debugstrs: nested_loops.lua:2: FORLOOP ---
--     ; {
--     ;   "trid": "2",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "0"
--     ;   }
--     ; }
--     nested_loops.lua:2: FORLOOP
--     nested_loops.lua:3: ADDI
--     nested_loops.lua:4: LOADI
--     nested_loops.lua:4: LOADI
--     nested_loops.lua:4: LOADI
--     nested_loops.lua:4: FORPREP
--     nested_loops.lua:5: ADDI
--     --- End debugstrs ---
--     --- Begin debugstrs: nested_loops.lua:4: FORLOOP ---
--     ; {
--     ;   "trid": "1",
--     ;   "start": {
--     ;     "kind": "Guard",
--     ;     "src_trid": "0",
--     ;     "gidx": "${{_}}"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "2"
--     ;   }
--     ; }
--
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:4: FORLOOP
--     yk-execution: deoptimise TraceId(2) ...
--     251502
