-- Run-time:
--   env-var: YKD_SERIALISE_COMPILATION=1

-- This program segfaulted before we correctly restored the shadow stack
-- pointer after a longjmp().
--
-- This test case was reduced out of calls.lua from the Lua test suite.

do
	function A()
		B(pcall(A))
	end
	xpcall(A, A)
end
