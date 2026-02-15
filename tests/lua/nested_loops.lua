-- Run-time:
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-tracing: start-tracing: nested_loops.lua:75: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:75: ADDI
--     --- Begin debugstrs: nested_loops.lua:75: ADDI ---
--     ; {
--     ;   "trid": "${{0}}",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     nested_loops.lua:75: ADDI
--     nested_loops.lua:74: FORLOOP
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     yk-tracing: start-side-tracing: nested_loops.lua:75: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:73: ADDI
--     yk-tracing: start-tracing: nested_loops.lua:73: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:75: ADDI
--     --- Begin debugstrs: nested_loops.lua:73: ADDI ---
--     ; {
--     ;   "trid": "${{2}}",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "${{0}}"
--     ;   }
--     ; }
--     nested_loops.lua:73: ADDI
--     nested_loops.lua:74: LOADI
--     nested_loops.lua:74: LOADI
--     nested_loops.lua:74: LOADI
--     nested_loops.lua:74: FORPREP
--     --- End debugstrs ---
--     --- Begin debugstrs: nested_loops.lua:75: ADDI ---
--     ; {
--     ;   "trid": "${{1}}",
--     ;   "start": {
--     ;     "kind": "Guard",
--     ;     "src_trid": "0",
--     ;     "gidx": "0"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "${{2}}"
--     ;   }
--     ; }
--     nested_loops.lua:72: FORLOOP
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:75: ADDI
--     yk-execution: deoptimise ...
--     251502

local x = 0
for _ = 0, 500 do
  x = x + 1
  for _ = 0, 500 do
    x = x + 1
  end
end
io.stderr:write(x)
