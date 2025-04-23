-- ignore-if: test "$YKB_TRACER" = "swt"
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
--     yk-jit-event: start-tracing: sidetrace.lua:60: LTI
--     <3
--     yk-jit-event: stop-tracing: ...
--     --- Begin debugstrs: header: sidetrace.lua:60: LTI ---
--       sidetrace.lua:61: GETTABUP
--       sidetrace.lua:61: GETFIELD
--       sidetrace.lua:61: SELF
--       sidetrace.lua:61: LOADK
--       sidetrace.lua:61: GETTABUP
--       sidetrace.lua:61: MOVE
--       sidetrace.lua:61: CALL
--       sidetrace.lua:61: LOADK
--       sidetrace.lua:61: CALL
--       sidetrace.lua:61: JMP
--       sidetrace.lua:59: FORLOOP
--       sidetrace.lua:60: LTI
--     --- End debugstrs ---
--     <4
--     yk-jit-event: enter-jit-code: sidetrace.lua:60: LTI
--     yk-jit-event: deoptimise
--     >=5
--     yk-jit-event: enter-jit-code: sidetrace.lua:60: LTI
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing: sidetrace.lua:60: LTI
--     >=6
--     yk-jit-event: stop-tracing: ...
--     --- Begin debugstrs: side-trace: sidetrace.lua:60: LTI ---
--       sidetrace.lua:63: GETTABUP
--       sidetrace.lua:63: GETFIELD
--       sidetrace.lua:63: SELF
--       sidetrace.lua:63: LOADK
--       sidetrace.lua:63: GETTABUP
--       sidetrace.lua:63: MOVE
--       sidetrace.lua:63: CALL
--       sidetrace.lua:63: LOADK
--       sidetrace.lua:63: CALL
--       sidetrace.lua:59: FORLOOP
--       sidetrace.lua:60: LTI
--     --- End debugstrs ---
--     >=7
--     yk-jit-event: enter-jit-code: sidetrace.lua:60: LTI
--     >=8
--     >=9
--     >=10
--     yk-jit-event: deoptimise
--     exit

for i = 0, 10 do
  if i < 5 then
    io.stderr:write("<", tostring(i), "\n")
  else
    io.stderr:write(">=", tostring(i), "\n")
  end
end
io.stderr:write("exit\n")
