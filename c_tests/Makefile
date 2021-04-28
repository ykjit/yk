LDFLAGS = -Wl,--plugin-opt=-lto-embed-bitcode=optimized -lyktrace
CFLAGS = -fuse-ld=lld -flto

DEBUG_TARGET_DIR = ../../target/debug
RELEASE_TARGET_DIR = ../../target/release

# We can't use an rpath for the rust sysroot shared objects as the system rust is found first.
SYSROOT_LIBDIR = $(shell rustc --print sysroot)/lib

all:
	@echo "valid targets are:"
	@echo "  test-release"
	@echo "  test-debug"
	exit 1

test-debug:
	cargo build --features c_testing
	@for i in `ls -d */`; do \
		echo "==> Running C test $$i" && ${MAKE} -C $$i \
		LDFLAGS="${LDFLAGS} -L${DEBUG_TARGET_DIR} -Wl,-rpath=${DEBUG_TARGET_DIR}" \
		CFLAGS="${CFLAGS}" SYSROOT_LIBDIR=${SYSROOT_LIBDIR}; done

test-release:
	cargo build --release --features c_testing
	@for i in `ls -d */`; do \
		echo "==> Running C test $$i" && ${MAKE} -C $$i \
		LDFLAGS="${LDFLAGS} -L${RELEASE_TARGET_DIR} -Wl,-rpath=${RELEASE_TARGET_DIR}" \
		CFLAGS="${CFLAGS}" SYSROOT_LIBDIR=${SYSROOT_LIBDIR}; done

