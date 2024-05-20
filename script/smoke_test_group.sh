#!/bin/bash
# need use posix compatible bash

if [[ $# -ge 1 ]] ; then
    DIR=$1
else
    DIR=bwd
fi

if [[ $# -ge 2 ]] ; then
    LAYOUT=$2
else
    LAYOUT="nchw"
fi

if [[ $# -ge 3 ]] ; then
    PREC=$3
else
    PREC="fp32"
fi

if [[  $# -ge 4 ]] ; then
    ARCH=$4
else
    ARCH="gfx908"
fi

if [[  "${LAYOUT}" = "nchw" ]] ; then
    LAYOUT_HSACO=""
    LAYOUT_ARG=""
elif [[  "${LAYOUT}" = "nhwc" ]] ; then
    LAYOUT_HSACO="_nhwc"
    LAYOUT_ARG="--in_layout NHWC --fil_layout NHWC --out_layout NHWC"
elif [[  "${LAYOUT}" = "nchwc_kcyxc" ]] ; then
    LAYOUT_HSACO="_nchwc"
    LAYOUT_ARG="--in_layout NCHWC --fil_layout NCHWC --out_layout NCHWC"
elif [[  "${LAYOUT}" = "nchwc_cyxkc" ]] ; then
    LAYOUT_HSACO="_nchwc"
    LAYOUT_ARG="--in_layout NCHWC --fil_layout CHWNC --out_layout NCHWC"
else
    echo "wrong layout: ${LAYOUT}"
    exit 1
fi

if [[ "${PREC}" = "fp32" ]] ; then
    PREC_HSACO=""
    CONV="conv"
elif [[ "${PREC}" = "int4"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[  "${PREC}" = "fp16"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[  "${PREC}" = "int8"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="conv${PREC}"
elif [[ "${PREC}" = "bf16"* ]] ; then
    PREC_HSACO="_${PREC}"
    CONV="convbfp16${PREC:4}"
else
    echo "wrong precision: ${PREC}"
    exit 1
fi

if [ "${ARCH}" != "gfx90a" ] && [ "${ARCH}" != "gfx908" ] && [ "${ARCH}" != "gfx1030" ] ; then
    echo "wrong arch: ${ARCH}"
    exit 1
fi

export IGEMM_HSACO=out/igemm_${DIR}_gtc_${ARCH}${LAYOUT_HSACO}${PREC_HSACO}.hsaco
echo IGEMM_HSACO:$IGEMM_HSACO
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
#export IGEMM_SCLK_MHZ=1283
if [ "${ARCH}" = "gfx90a" ]; then
    export IGEMM_SCLK_MHZ=1700
elif [ "${ARCH}" = "gfx908" ]; then
    export IGEMM_SCLK_MHZ=1502
elif [ "${ARCH}" = "gfx1030" ] ; then
    export IGEMM_SCLK_MHZ=2450
fi

export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_SLEEP_MS=11
export PER_PIXEL_CHECK=0
export IGEMM_BENCH_CSV=1

export IGEMM_RAND_INT=1
export IGEMM_ASSERT_WHEN_INVALID=1
export IGEMM_WARMUP=1
export IGEMM_REPEAT=4

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

#group_size=( 1 2 16 )
group_size=( 8 )
batch_size=( 1 2 )
channel_size=( 16 32 64 )
#channel_size=( 32 64 )
image_size=( 32 55 )
filter_size=( 1 3 5 7 )
pad_size=( 0 1 )    # 0: no padding; 1: skip if fs=1, p=1 if fs=3, p=2 if fs=5, p=3 if fs=7
stride_size=( 1 2 )
dilation_size=( 1 2 )
#dilation_size=( 1 )

for g  in "${group_size[@]}"; do
for n  in "${batch_size[@]}"; do
for c  in "${channel_size[@]}"; do
for k  in "${channel_size[@]}"; do
for hi in "${image_size[@]}"; do
wi=$(( $hi ))    # skip for wi != hi for now
for fy in "${filter_size[@]}"; do
fx=$(( $fy ))    # skip for fx != fy for now
for py in "${pad_size[@]}"; do
if (( $py == 1 )); then
    if (( $fy == 1 )); then
        #echo "skip if filter size less than or equal to padding size!"
        continue
    elif (( $fy == 5 )); then
        py=$(( $py + 1 ))
    elif (( $fy == 7 )); then
        py=$(( $py + 2 ))
    fi
fi
px=$(( $py ))    # skip for px != py for now
for sy in "${stride_size[@]}"; do
sx=$(( $sy ))    # skip for sx != sy for now
for dy in "${dilation_size[@]}"; do
dx=$(( $dy ))    # skip for dx != dy for now
if (( $dy > $fy || $fy > $hi || $c % $g != 0 || $k % $g != 0 )); then
    #echo "skip if dilation size greater than filter size!"
    continue
fi
if (( $dy > 1 )); then
    if (( $py != 0 || $sy != 1 )); then
        #echo "skip non-default padding or stride for now!"
        continue
    fi
fi
if (( ( $hi + 2 * $py - $dy * ( $fy - 1 ) - 1 ) < 0 )); then
    #echo "skip if output size could be less than zero!"
    continue
fi
if (( ( $hi + 2 * $py - $dy * ( $fy - 1 ) - 1 ) % $sy != 0 )); then
    #echo "skip if not divisible by stride size!"
    continue
fi

# (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
ho=$(( ( $hi + 2 * $py - $dy * ( $fy - 1 ) - 1 ) / $sy + 1  ))
wo=$(( ( $wi + 2 * $px - $dx * ( $fx - 1 ) - 1 ) / $sx + 1  ))
if (( $ho <= 0 || $wo <= 0 )); then
    #echo "skip if output size less than or equal to zero!"
    continue
fi

echo "${CONV} -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW ${LAYOUT_ARG} (ho:$ho, wo:$wo)"
$EXE $CONV -n $n -c $c -H $hi -W $wi -k $k -y $fy -x $fx -p $py -q $px -u $sy -v $sx -l $dy -j $dx -g $g -F $FORW ${LAYOUT_ARG} || echo -e "\n\n!!!ERROR!!!\n\n"

done
done
done
done
done
done
done
done
done

