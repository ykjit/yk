DIR != pwd
INST_DIR = ${DIR}/inst
PYTHON=python3

PROCESSOR_TRACE_REPO = https://github.com/01org/processor-trace.git
PROCESSOR_TRACE_V = 892e12c5a27bda5806d1e63269986bb4171b5a8b
PROCESSOR_TRACE_SOURCE = processor-trace

XED_REPO = https://github.com/intelxed/xed
XED_V = b4109c0823ff35df7416a157178c87f77421994b
XED_SOURCE = xed

MBUILD_REPO = https://github.com/intelxed/mbuild
MBUILD_V = f2de5e93ac318e223f6ec61c41f541171db0ada7
MBUILD_SOURCE = mbuild

.PHONY: libipt

all: libipt

libipt: ${INST_DIR}/bin/ptdump

${INST_DIR}:
	install -d ${INST_DIR}/bin

# Fetch targets
${PROCESSOR_TRACE_SOURCE}:
	git clone ${PROCESSOR_TRACE_REPO}
	cd ${PROCESSOR_TRACE_SOURCE} && git checkout ${PROCESSOR_TRACE_V}

${XED_SOURCE}:
	git clone ${XED_REPO}
	cd ${XED_SOURCE} && git checkout ${XED_V}

${MBUILD_SOURCE}:
	git clone ${MBUILD_REPO}
	cd ${MBUILD_SOURCE} && git checkout ${MBUILD_V}

# Build targets
${PROCESSOR_TRACE_SOURCE}/bin/ptdump: ${PROCESSOR_TRACE_SOURCE} ${XED_SOURCE}/obj/libxed.so
	cd ${PROCESSOR_TRACE_SOURCE} && \
		env CFLAGS"=-I${DIR}/${XED_SOURCE}/include/public/xed -I${DIR}/${XED_SOURCE}/obj -Wno-error -g" \
		LDFLAGS="-L${DIR}/${XED_SOURCE}/obj -Wl,-rpath=${DIR}/${XED_SOURCE}/obj" \
		cmake -DCMAKE_INSTALL_PREFIX:PATH=${INST_DIR} \
		-DPTDUMP=ON -DPTXED=ON -DSIDEBAND=ON -DFEATURE_ELF=ON -DDEVBUILD=ON . && ${MAKE}

${XED_SOURCE}/obj/libxed.so: ${XED_SOURCE} ${MBUILD_SOURCE}
	cd ${XED_SOURCE} && ${PYTHON} mfile.py --shared

# Install targets
${INST_DIR}/bin/ptdump: ${INST_DIR} ${PROCESSOR_TRACE_SOURCE}/bin/ptdump
	cd ${PROCESSOR_TRACE_SOURCE} && ${MAKE} install
	install ${PROCESSOR_TRACE_SOURCE}/bin/ptdump ${INST_DIR}/bin/
	install ${PROCESSOR_TRACE_SOURCE}/bin/ptxed ${INST_DIR}/bin/

clean:
	rm -rf ${INST_DIR} ${PROCESSOR_TRACE_SOURCE} ${XED_SOURCE} ${MBUILD_SOURCE}
