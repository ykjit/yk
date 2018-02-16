DIR != pwd
INST_DIR = ${DIR}/inst
PYTHON=python2.7

PROCESSOR_TRACE_REPO = https://github.com/01org/processor-trace.git
PROCESSOR_TRACE_V = 0ff8b29b2fd2ebfcc47a747862e948e8b638a020
PROCESSOR_TRACE_SOURCE = processor-trace

XED_REPO = https://github.com/intelxed/xed
XED_V = bd73bdb2eefac1896b886e6a39e1794ae5267f7f
XED_SOURCE = xed

MBUILD_REPO = https://github.com/intelxed/mbuild
MBUILD_V = bb9123152a330c7fa1ff1a502950dc199c83e177
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
