-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     <0
--     <1
--     <2
--     yk-tracing: start-tracing: sidetrace.lua:79: LTI
--     <3
--     yk-tracing: stop-tracing: sidetrace.lua:79: LTI
--     --- Begin debugstrs: sidetrace.lua:79: LTI ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     sidetrace.lua:79: LTI
--     sidetrace.lua:80: GETTABUP
--     sidetrace.lua:80: GETFIELD
--     sidetrace.lua:80: SELF
--     sidetrace.lua:80: LOADK
--     sidetrace.lua:80: GETTABUP
--     sidetrace.lua:80: MOVE
--     sidetrace.lua:80: CALL
--     sidetrace.lua:80: LOADK
--     sidetrace.lua:80: CALL
--     sidetrace.lua:80: JMP
--     sidetrace.lua:78: FORLOOP
--     --- End debugstrs ---
--     <4
--     yk-execution: enter-jit-code: sidetrace.lua:79: LTI
--     yk-execution: deoptimise ...
--     >=5
--     yk-execution: enter-jit-code: sidetrace.lua:79: LTI
--     yk-execution: deoptimise ...
--     yk-tracing: start-side-tracing: sidetrace.lua:79: LTI
--     >=6
--     yk-tracing: stop-tracing: sidetrace.lua:79: LTI
--     --- Begin debugstrs: sidetrace.lua:79: LTI ---
--     ; {
--     ;   "trid": "1",
--     ;   "start": {
--     ;     "kind": "Guard",
--     ;     "src_trid": "0",
--     ;     "gidx": "72"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "0"
--     ;   }
--     ; }
--     sidetrace.lua:82: GETTABUP
--     sidetrace.lua:82: GETFIELD
--     sidetrace.lua:82: SELF
--     sidetrace.lua:82: LOADK
--     sidetrace.lua:82: GETTABUP
--     sidetrace.lua:82: MOVE
--     sidetrace.lua:82: CALL
--     sidetrace.lua:82: LOADK
--     sidetrace.lua:82: CALL
--     sidetrace.lua:78: FORLOOP
--     --- End debugstrs ---
--     >=7
--     yk-execution: enter-jit-code: sidetrace.lua:79: LTI
--     >=8
--     >=9
--     >=10
--     yk-execution: deoptimise ...
--     exit

for i = 0, 10 do
  if i < 5 then
    io.stderr:write("<", tostring(i), "\n")
  else
    io.stderr:write(">=", tostring(i), "\n")
  end
end
io.stderr:write("exit\n")
