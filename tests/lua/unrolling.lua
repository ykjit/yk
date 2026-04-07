for i = 0, 5 do
  if i > 2 then
    for j = 0, 2 do
      io.stderr:write("j ", tostring(j), "\n")
    end
  end
  io.stderr:write("i ", tostring(i), "\n")
end
io.stderr:write("exit\n")

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     i 0
--     i 1
--     i 2
--     j 0
--     j 1
--     j 2
--     i 3
--     yk-tracing: start-tracing: unrolling.lua:1: FORLOOP
--     j 0
--     j 1
--     yk-warning: tracing-aborted: unrolled inner loop: unrolling.lua:3: FORLOOP
--     yk-tracing: start-tracing: unrolling.lua:3: FORLOOP
--     j 2
--     yk-tracing: stop-tracing: unrolling.lua:3: FORLOOP
--     --- Begin debugstrs: unrolling.lua:3: FORLOOP ---
--     ; {
--     ;   "trid": "1",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     unrolling.lua:3: FORLOOP
--     unrolling.lua:4: GETTABUP
--     unrolling.lua:4: GETFIELD
--     unrolling.lua:4: SELF
--     unrolling.lua:4: LOADK
--     unrolling.lua:4: GETTABUP
--     unrolling.lua:4: MOVE
--     unrolling.lua:4: CALL
--     unrolling.lua:4: LOADK
--     unrolling.lua:4: CALL
--     --- End debugstrs ---
--     i 4
--     yk-tracing: start-tracing: unrolling.lua:1: FORLOOP
--     j 0
--     yk-tracing: stop-tracing: unrolling.lua:3: FORLOOP
--     --- Begin debugstrs: unrolling.lua:1: FORLOOP ---
--     ; {
--     ;   "trid": "2",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "1"
--     ;   }
--     ; }
--     unrolling.lua:1: FORLOOP
--     unrolling.lua:2: GTI
--     unrolling.lua:3: LOADI
--     unrolling.lua:3: LOADI
--     unrolling.lua:3: LOADI
--     unrolling.lua:3: FORPREP
--     unrolling.lua:4: GETTABUP
--     unrolling.lua:4: GETFIELD
--     unrolling.lua:4: SELF
--     unrolling.lua:4: LOADK
--     unrolling.lua:4: GETTABUP
--     unrolling.lua:4: MOVE
--     unrolling.lua:4: CALL
--     unrolling.lua:4: LOADK
--     unrolling.lua:4: CALL
--     --- End debugstrs ---
--     j 1
--     yk-execution: enter-jit-code: unrolling.lua:3: FORLOOP
--     j 2
--     yk-execution: deoptimise TraceId(1) ...
--     i 5
--     yk-execution: enter-jit-code: unrolling.lua:1: FORLOOP
--     yk-execution: deoptimise TraceId(2) ...
--     exit
