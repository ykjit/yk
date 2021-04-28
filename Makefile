all:
	${MAKE} -C c_tests test-debug
	${MAKE} -C c_tests test-release
