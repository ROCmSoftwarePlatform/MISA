
#!/bin/sh
if [ $# -ge 1 ] ; then
    DIR=$1
else
    DIR=bwd
fi

if [ $# -ge 2 ] ; then
    LAYOUT=$2
else
    LAYOUT="nchw"
fi

if [ $# -ge 3 ] ; then
    PREC=$3
else
    PREC="fp32"
fi

if [ $# -ge 4 ] ; then
    ARCH=$4
else
    ARCH="gfx908"
fi

if [ "${LAYOUT}" = "nchw" ] ; then
    LAYOUT_HSACO=""
    LAYOUT_ARG=""
elif [ "${LAYOUT}" = "nhwc" ] ; then
    LAYOUT_HSACO="_nhwc"
    LAYOUT_ARG="--in_layout NHWC --fil_layout NHWC --out_layout NHWC"
else
    echo "wrong layout: ${LAYOUT}"
    exit 1
fi

if [ "${PREC}" = "fp32" ] ; then
    PREC_HSACO=""
    CONV="conv"
elif [ "${PREC}" = "fp16" ] ; then
    PREC_HSACO="_fp16"
    CONV="convfp16"
elif [ "${PREC}" = "int8" ] ; then
    PREC_HSACO="_int8"
    CONV="convint8"
else
    echo "wrong precision: ${PREC}"
    exit 1
fi

echo IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
#export IGEMM_SCLK_MHZ=1283
if [ "${ARCH}" = "gfx90a" ]; then
    export IGEMM_SCLK_MHZ=1700
elif [ "${ARCH}" = "gfx908" ]; then
    export IGEMM_SCLK_MHZ=1502
elif [ "${ARCH}" = "gfx1030" ] ; then
    export IGEMM_SCLK_MHZ=2450
fi
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=117
export PER_PIXEL_CHECK=0
export IGEMM_RAND_INT=1
export WRW_DBG=0
export IGEMM_BENCH_CSV=1

#HIP_VISIBLE_DEVICES=1 # 2

rm bench_model.csv

# Flag enables fwd, bwd, wrw convolutions
if [ "${DIR}" = "fwd" ] ; then
    FORW=1
elif [ "${DIR}" = "bwd" ] ; then
    FORW=2
elif [ "${DIR}" = "wrw" ] ; then
    FORW=4
else
    echo "wrong direction"
    exit 1
fi

EXE=./out/conv_driver.exe

group_size=( 1 2 )

for g  in "${group_size[@]}"; do
echo "=============================================================== gemm big size group:$g"
if [ "${DIR}" = "fwd" -o "${DIR}" = "wrw" ] ; then
$EXE $CONV -n 4 -c 1024 -H 16 -W 16 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 8 -c 2048 -H 16 -W 16 -k 1920  -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 16 -c 4096 -H 16 -W 16 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 32 -c 8192 -H 16 -W 16 -k 7680 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
elif [ "${DIR}" = "bwd" ] ; then
$EXE $CONV -n 15 -c 1024 -H 8 -W 8 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 15 -c 2048 -H 8 -W 16 -k 2048  -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 15 -c 4096 -H 16 -W 16 -k 4096 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
$EXE $CONV -n 30 -c 8192 -H 16 -W 16 -k 8192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g $g -t 1 -F $FORW ${LAYOUT_ARG} -V 1
fi
done
exit 1