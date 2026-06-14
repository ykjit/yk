-- Test that the `longjmp` inside `pcall` is treated correctly.
function b()
	while 0 do
		a()
	end
end
pcall(b)
for a = 0, 0 do
end

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=0
--   env-var: YK_SIDETRACE_THRESHOLD=0
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-tracing: start-tracing: pcall.lua:4: GETTABUP
--     yk-warning: tracing-aborted: longjmp encountered: pcall.lua:4: GETTABUP
--     yk-tracing: start-tracing: pcall.lua:8: FORLOOP
