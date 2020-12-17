# This Makefile deals with building libipt. It is symlinked into OUT_DIR at
# build time by build.rs.

DIR != pwd
INST_DIR = ${DIR}/inst
PYTHON=python3

PROCESSOR_TRACE_REPO = https://github.com/intel/libipt
PROCESSOR_TRACE_V = ffe1631be3dad2dc286529e3e05d552043d626f0
PROCESSOR_TRACE_SOURCE = libipt

XED_REPO = https://github.com/intelxed/xed
XED_V = f7191e268c3ee17fc8c9b8d9bd3eee7159f29556
XED_SOURCE = xed

MBUILD_REPO = https://github.com/intelxed/mbuild
MBUILD_V = 3e8eb33aada4153c21c4261b35e5f51f6e2019e8
MBUILD_SOURCE = mbuild

.PHONY: libipt

all: ${INST_DIR}/bin/ptdump

${INST_DIR}:
	install -d ${INST_DIR}/bin

# Fetch targets
.PHONY: ${PROCESSOR_TRACE_SOURCE}
${PROCESSOR_TRACE_SOURCE}:
	if ! [ -d ${PROCESSOR_TRACE_SOURCE} ]; then \
		git clone ${PROCESSOR_TRACE_REPO}; \
	else \
		cd ${PROCESSOR_TRACE_SOURCE} && git fetch; \
	fi
	cd ${PROCESSOR_TRACE_SOURCE} && git checkout ${PROCESSOR_TRACE_V}

.PHONY: ${XED_SOURCE}
${XED_SOURCE}:
	if ! [ -d ${XED_SOURCE} ]; then \
		git clone ${XED_REPO}; \
	else \
		cd ${XED_SOURCE} && git fetch; \
	fi
	cd ${XED_SOURCE} && git checkout ${XED_V}

.PHONY: ${MBUILD_SOURCE}
${MBUILD_SOURCE}:
	if ! [ -d ${MBUILD_SOURCE} ]; then \
		git clone ${MBUILD_REPO}; \
	else \
		cd ${MBUILD_SOURCE} && git fetch; \
	fi
	cd ${MBUILD_SOURCE} && git checkout ${MBUILD_V}

# Build targets
${PROCESSOR_TRACE_SOURCE}/bin/ptdump: ${PROCESSOR_TRACE_SOURCE} ${XED_SOURCE}/obj/libxed.so
	cd ${PROCESSOR_TRACE_SOURCE} && \
		env CFLAGS"=-I${DIR}/${XED_SOURCE}/include/public/xed -I${DIR}/${XED_SOURCE}/obj -Wno-error -g" \
		LDFLAGS="-L${DIR}/${XED_SOURCE}/obj -Wl,-rpath=${DIR}/${XED_SOURCE}/obj" \
		cmake -DCMAKE_INSTALL_PREFIX:PATH=${INST_DIR} \
		-DPTDUMP=ON -DPTXED=ON -DSIDEBAND=ON -DFEATURE_ELF=ON -DDEVBUILD=ON \
		-DBUILD_SHARED_LIBS=OFF . && ${MAKE}

${XED_SOURCE}/obj/libxed.so: ${XED_SOURCE} ${MBUILD_SOURCE}
	cd ${XED_SOURCE} && ${PYTHON} mfile.py --shared

# Install targets
${INST_DIR}/bin/ptdump: ${INST_DIR} ${PROCESSOR_TRACE_SOURCE}/bin/ptdump
	cd ${PROCESSOR_TRACE_SOURCE} && ${MAKE} install
	install ${PROCESSOR_TRACE_SOURCE}/bin/ptdump ${INST_DIR}/bin/
	install ${PROCESSOR_TRACE_SOURCE}/bin/ptxed ${INST_DIR}/bin/

clean:
	rm -rf ${INST_DIR} ${PROCESSOR_TRACE_SOURCE} ${XED_SOURCE} ${MBUILD_SOURCE}
