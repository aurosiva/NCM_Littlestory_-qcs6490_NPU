#!/usr/bin/env bash
# ONNX -> DLC -> quantized DLC -> HTP context binary for QCS6490 (v68, soc 35, 2 MB VTCM).
# Usage: bash npu/build.sh [act_bits=16] [weight_bits=8]
set -eo pipefail
cd "$(dirname "$0")/build"
source ../qenv.sh; set -u
A=${1:-16}; Wb=${2:-8}
TAG=a${A}w${Wb}; [ "$A" = 16 ] && TAG=${TAG}mix   # 16-bit, attention matmuls 8-bit
for m in gen dec; do
  OV=""; [ "$A" = 16 ] && OV="--quantization_overrides ${m}_overrides.json"
  qairt-converter --input_network $m.onnx --output_path ${m}_fp32.dlc $OV > ${m}_convert.log 2>&1
  qairt-quantizer --input_dlc ${m}_fp32.dlc --input_list calib_$m.txt --output_dlc ${m}_${TAG}.dlc \
      --act_bitwidth $A --weights_bitwidth $Wb --bias_bitwidth 32 --use_per_channel_quantization \
      > ${m}_quant.log 2>&1
  cat > ${m}_backend.json <<J
{ "graphs":  [ { "graph_names": ["${m}_fp32"], "vtcm_mb": 2, "O": 3 } ],
  "devices": [ { "dsp_arch": "v68", "soc_id": 35 } ] }
J
  echo "{ \"backend_extensions\": { \"shared_library_path\": \"libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"${m}_backend.json\" } }" > ${m}_config.json
  qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
      --model $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so --dlc_path ${m}_${TAG}.dlc \
      --config_file ${m}_config.json --binary_file ${m}_${TAG} --output_dir . > ${m}_ctx.log 2>&1
  ls -la ${m}_${TAG}.bin
done
