# NCM Littlestory on the QCS6490 NPU

A minimal **Next-Concept Modeling (NCM)** language model, written and trained from scratch on
TinyStories, and deployed to the **Qualcomm Hexagon NPU** of a Radxa Dragon Q6A (QCS6490).

NCM ([arXiv 2610.14162](https://arxiv.org/abs/2610.14162)) does not write text one token at a
time. It first predicts a coarse plan of the whole text, then refines it level by level. Here a
128-token story is produced in **12 generation steps instead of 128**, and every code inside a
step is predicted in parallel, which suits an NPU well.

| | |
|---|---|
| Model | ~20M parameters (8.0M tokenizer + 11.9M generator), `v2` preset |
| Training | laptop RTX 5060 (8 GB), TinyStories, about 6.5 hours total |
| Target | Radxa Dragon Q6A, Qualcomm QCS6490, Hexagon v68 NPU, Ubuntu 24.04 |
| NPU toolchain | Qualcomm QAIRT 2.50 (converter, quantizer, HTP context binaries) |
| Result | **143 ms per story on the NPU, 1.9× faster than a tuned CPU** |

Open [`report/ncm_npu_report.html`](report/ncm_npu_report.html) in a browser for the full
comparison with charts (speed, sysMon NPU profiles, accuracy per level).

---

## Contents

1. [How NCM works](#how-ncm-works)
2. [Results](#results)
3. [Repository layout](#repository-layout)
4. [Train on a laptop](#1-train-on-a-laptop)
5. [Run on the board's CPU](#2-run-on-the-boards-cpu)
6. [Run on the Hexagon NPU](#3-run-on-the-hexagon-npu)
7. [Profile the NPU with sysMon](#4-profile-the-npu-with-sysmon)
8. [Lessons and pitfalls](#lessons-and-pitfalls)
9. [Limitations](#limitations)
10. [Credits](#credits)

---

## How NCM works

**Stage 1: multiscale residual VQ tokenizer.** A story of L = 128 BPE tokens is encoded to a
latent sequence, then quantized into K = 12 levels of discrete codes, from coarse to fine. Level
1 has a single code for the whole story; level 12 has 128 codes. Each level encodes what the
previous levels missed (a residual), and a small decoder turns the summed codes back into tokens.

**Stage 2: level-parallel generator.** A transformer predicts all codes of level k at once from
a downsampled view of levels 1..k-1. Attention is block-structured, so generation takes exactly
K = 12 passes, followed by one decoder pass.

```
level  1  ●                                   1 code   (the whole-story "concept")
level  2  ● ● ● ● ● ● ● ●                     8 codes
  ...                                         ...
level 12  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●   128 codes (token-level detail)
          └─ decoder ─► 128 BPE tokens ─► text
```

Training follows the paper: nested dropout, code corruption (75% at level 1 falling to 25% at
level 12) and per-level loss weights w_k ∝ l_k^-α.

Two presets are included:

| preset | what it is |
|---|---|
| `base` | first version: level sizes 1,2,4,8,…,128, fixed resizing, separate output heads, learned positions |
| `v2` (deployed) | adds the choices of the official code: its level sizes (1,8,16,…,128), learned level-1 resize, output heads tied to the codebooks, RoPE + QK-norm |

The official repository ([HazyResearch/ncm](https://github.com/HazyResearch/ncm)) contains
evaluation code and checkpoints only. All training code here is written from scratch;
the official code was used as a reference for the `v2` design.

---

## Results

All numbers were measured on the board or laptop, as described in each table.

### Speed (Radxa Q6A, warm, 48 stories)

| setup | ms per story |
|---|---|
| CPU, onnxruntime INT8, 8 threads | 808 |
| CPU, default thread settings | 586 |
| CPU, tuned (4 onnxruntime threads, 1 numpy thread) | 276 |
| CPU, tuned, 4 stories per batch | 196 |
| **NPU, 16-bit mixed precision (recommended)** | **143** |
| NPU, 8-bit | 107 |

Of the NPU's 143 ms, 98 ms is inside the 13 NPU calls (including data transfer) and 45 ms is
Python work on the CPU between calls.

### Does the NPU do the work? (Qualcomm sysMon, 10 s captures)

| capture | NPU utilisation | NPU clock |
|---|---|---|
| nothing running | 0% | 365 MHz |
| NCM on the CPU | 0% | 365 MHz |
| NCM on the NPU | 23% | 941 MHz 94% of the time, peaks of 1306 MHz |

The NPU works in short bursts and waits for the CPU in between, so the Python side is now the
bottleneck.

### Accuracy of the quantized models

Given the same input, how often the quantized model picks the same code as the full-precision
model (16 stories; level 1 always matches):

| build | levels 2–12 | decoder tokens |
|---|---|---|
| NPU 16-bit mixed | 91–98% | 99.7% |
| NPU 8-bit | 59–92% | 99.4% |
| CPU INT8 (laptop check, 8 stories) | 92–100% | – |

### Story quality (GPT-2 Large, 256 stories × 2 seeds, CPU INT8)

| text | GPT-2 perplexity | word entropy | doubled words |
|---|---|---|---|
| real TinyStories | 9.1 | 3.96 | 0.0% |
| `base` preset | 54.9 / 58.6 | 4.00 | 4.8% |
| `v2` preset | 53.1 / 49.6 | 3.95 | 2.7% |

Sample from the NPU (unedited):

> Once there was a little boy who lived to explore. He loved to always always exploring. One
> day, he was looking looking for something to eat. Suddenly, he saw a big rainbow in the ground.

The stories start well, then drift and repeat words. That is a limit of model size and training
time, not of the NPU.

---

## Repository layout

```
config.py            model, tokenizer and training settings; the base and v2 presets
prepare_data.py      download TinyStories, train a 4096-token BPE, write data/train.bin, val.bin
model.py             MultiscaleTokenizer (stage 1) and NCMGenerator (stage 2)
train_tokenizer.py   train stage 1
train_generator.py   train stage 2
run_v2.sh            queue: export base, then train and export v2
export_edge.py       ONNX export + INT8 + numpy weights for the CPU runner, with a parity check
edge_run_ncm.py      CPU runner for the board (copied as run_ncm.py)
bench_cpu.py         fair CPU timing on the board
compare.py           GPT-2 Large scoring (perplexity, entropy, repeats)
npu/
  export_npu.py      fixed-shape ONNX graphs for the NPU + calibration data from real generation
  make_overrides.py  8-bit overrides for the attention matmuls (v68 limit)
  build.sh           qairt-converter → qairt-quantizer → qnn-context-binary-generator
  qenv.sh            QAIRT environment setup
  qnn_serve.c        small C server that keeps the NPU models loaded (QNN C API)
  run_ncm_npu.py     NPU runner and accuracy check for the board
report/
  ncm_npu_report.html            the comparison page (self-contained)
  ncm_npu_report.template.html   the same page before the data is filled in
```

Not committed (see `.gitignore`): datasets, checkpoints, ONNX and NPU binaries, logs, and any
Qualcomm SDK files. Everything can be rebuilt with the steps below.

---

## 1. Train on a laptop

Tested on Ubuntu 22.04, Python 3.10, RTX 5060 8 GB.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python prepare_data.py                                   # data/ (1M stories, 128 tokens each)
python train_tokenizer.py --preset v2 --out ckpt_v2      # ~20 min
python train_generator.py --preset v2 --out ckpt_v2 --batch 64 --accum 2   # ~6 h
```

`--batch 64 --accum 2` gives an effective batch of 128 within 8 GB. On a larger GPU use
`--batch 128`.

Score the result against real stories (needs GPT-2 Large, about 3 GB download):

```bash
python export_edge.py --ckpt ckpt_v2 --out edge_v2 --int8
python compare.py edge_v2 --n 256 --seeds 0,1
```

---

## 2. Run on the board's CPU

`export_edge.py` writes `edge_v2/` with the ONNX models (fp32 and INT8), the tokenizer weights,
the BPE files and the runner. It also checks that ONNX and numpy match PyTorch.

```bash
# laptop
scp -r edge_v2 radxa@<board-ip>:~/ncm_v2

# board
python3 -m venv ~/ncm-venv
~/ncm-venv/bin/pip install numpy onnxruntime tokenizers
cd ~/ncm_v2
~/ncm-venv/bin/python run_ncm.py -n 4 --int8
```

For a fair timing (warm, averaged), copy `bench_cpu.py` into `~/ncm_v2` and run
`OPENBLAS_NUM_THREADS=1 ~/ncm-venv/bin/python bench_cpu.py --ort_threads 4 -n 48`.

---

## 3. Run on the Hexagon NPU

You need the **Qualcomm AI Runtime (QAIRT) SDK 2.50** from Qualcomm Software Center (free
account). Its files are not included here. The host steps run in a separate Python 3.10
environment with `numpy==1.26.4` and `onnx==1.16.1`, which the SDK's converters require.

### 3.1 Build the NPU models (laptop)

```bash
python npu/export_npu.py --ckpt ckpt_v2 --out npu/build   # gen.onnx, dec.onnx, calibration, cpu_head.npz

export QAIRT_SDK_ROOT=/path/to/qairt/2.50.0.260828
export QAIRT_PY_ENV=/path/to/miniconda3/envs/qairt        # the Python 3.10 environment
python npu/make_overrides.py npu/build/gen.onnx > npu/build/gen_overrides.json
python npu/make_overrides.py npu/build/dec.onnx > npu/build/dec_overrides.json
bash npu/build.sh 16 8      # → npu/build/gen_a16w8mix.bin, dec_a16w8mix.bin  (recommended)
bash npu/build.sh 8 8       # → npu/build/gen_a8w8.bin, dec_a8w8.bin          (faster, less accurate)
```

What the export does differently from the CPU export:

- **Fixed shapes.** Every level is padded to 128 positions and a mask hides the padding, so one
  compiled graph serves all 12 levels.
- **2-D inputs and outputs**, so the converter never reorders axes.
- **The output head stays on the CPU.** v2 scores codes by their distance to the codebook,
  a difference of large numbers that does not survive 8- or 16-bit rounding. The NPU returns
  hidden states and numpy computes the head in fp32.
- **Calibration data comes from real generation**, recorded level by level.

`build.sh` targets the QCS6490: `dsp_arch v68`, `soc_id 35`, 2 MB of VTCM.

### 3.2 Deploy to the board

The board needs the **same QAIRT version** as the one that compiled the models. Copy the 2.50
runtime from your SDK next to the models:

```bash
S=$QAIRT_SDK_ROOT
B=radxa@<board-ip>
ssh $B 'mkdir -p ~/ncm_npu/qnn ~/ncm_npu/inc'
scp $S/lib/aarch64-ubuntu-gcc9.4/{libQnnHtp.so,libQnnHtpV68Stub.so,libQnnSystem.so,libQnnHtpPrepare.so} \
    $S/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so  $B:~/ncm_npu/qnn/
scp -r $S/include/QNN $B:~/ncm_npu/inc/
scp npu/qnn_serve.c npu/run_ncm_npu.py npu/build/*_a16w8mix.bin npu/build/*_a8w8.bin \
    npu/build/cpu_head.npz $B:~/ncm_npu/
ssh $B 'cd ~/ncm_npu && cp ~/ncm_v2/{tok_weights.npz,run_ncm.py} . && cp -r ~/ncm_v2/bpe . \
        && gcc -O2 -Iinc/QNN qnn_serve.c -ldl -o qnn_serve'
```

### 3.3 Generate

```bash
cd ~/ncm_npu
~/ncm-venv/bin/python run_ncm_npu.py -n 4 --bits a16w8mix            # 4 stories
~/ncm-venv/bin/python run_ncm_npu.py -n 4 --bits a16w8mix --seed 0   # repeatable
~/ncm-venv/bin/python run_ncm_npu.py -n 100 --bits a16w8mix --out stories.txt
~/ncm-venv/bin/python run_ncm_npu.py --check 16 --bits a16w8mix      # agreement with fp32
```

The `rpcmem_init` line at start-up comes from Qualcomm's library and is harmless. To confirm
the NPU is in use, run this in a second terminal during a long run:

```bash
ls -l /proc/$(pgrep -x qnn_serve)/fd | grep fastrpc     # shows /dev/fastrpc-cdsp-secure
```

### How the runtime works

```
run_ncm_npu.py (Python, CPU)                         qnn_serve (C)          Hexagon NPU
for each of the 12 levels:
  zin = Down_k(zhat), padded to 128  ─ quantize ─►  stdin  ─► graph "gen" ─► transformer
  h  ◄─ dequantize ──────────────────────────────  stdout ◄─
  logits = head(h)        CPU, fp32
  codes  = sample(logits) CPU
  zhat  += contrib(codes) CPU
zhat ─ quantize ──────────────────────────────────►  stdin  ─► graph "dec" ─► decoder
tokens ◄─ dequantize, argmax ──────────────────────  stdout ◄─
```

`qnn_serve` loads both context binaries once, prints each tensor's name, shape and quantization
parameters, then runs requests from a pipe. Python reads those parameters, so a rebuilt model
works without code changes.

---

## 4. Profile the NPU with sysMon

`sysMonAppLE` (Hexagon SDK) records the NPU's activity. Run it while stories are generating:

```bash
# board
cd ~/ncm_npu
~/ncm-venv/bin/python run_ncm_npu.py --bits a16w8mix -n 200 --out /tmp/s.txt &
sleep 2
sudo ~/sysMonAppLE profiler --q6 cdsp --duration 10
wait
sudo chmod 644 /tmp/sysmon_cdsp.bin

# laptop
scp radxa@<board-ip>:/tmp/sysmon_cdsp.bin .
$HEXAGON_SDK_ROOT/tools/utils/sysmon/parser_linux_v2/HTML_Parser/sysmon_parser sysmon_cdsp.bin
```

The parser writes `Summary.csv`, `dcvs_metric.txt`, `post_processed_metrics.csv` and an HTML
report. Notes for this board:

- The output falls back to `/tmp/sysmon_cdsp.bin`. The `setprop`, `logcat` and `ERROR 0x14`
  messages are Android-only steps and can be ignored.
- When the NPU is idle the parser's HTML report comes out empty. Use `dcvs_metric.txt` or
  `post_processed_metrics.csv` instead. In the CSV, column 3 is the time and column 2 the
  sample length in ms (idle samples are about 200 ms long, busy ones 1 ms).
- Utilisation = effective DSP frequency ÷ clock.
- Profiling slows NCM from 143 to about 177 ms per story.

---

## Lessons and pitfalls

1. **INT8 export silently broke v2.** `onnxruntime.quantization.quantize_dynamic` also
   quantizes `Gather` tables by default, including the codebook. v2 then wrote "to, but, to,
   but…" while scoring a *better* GPT-2 perplexity (10.4), because repetitive text is easy to
   predict. Fix: `op_types_to_quantize=["MatMul"]`. Always check entropy next to perplexity.
2. **v68 cannot run float ops.** Graph inputs and outputs must be quantized, so
   `--preserve_io_datatype` fails ("SocModel doesn't support FP16"). Python quantizes and
   dequantizes using each tensor's scale and offset.
3. **v68 cannot multiply two 16-bit activation tensors** ("expected >= 73" for MatMul). The
   mixed build keeps 16-bit everywhere except the attention multiplies.
4. **Overrides on data-movement ops are ignored.** Setting Transpose or Squeeze outputs to 8 bits
   does nothing; set the compute op that produces the tensor. `make_overrides.py` walks back to it.
5. **numpy's thread pool can cost more than the NPU work.** With one spinning thread per core,
   all 8 cores sat at 96% and the NPU run took 225 ms. With `OPENBLAS_NUM_THREADS=4` it takes 143 ms.
6. **Compare like with like.** A cold, untuned CPU run first suggested the NPU was 5× faster.
   Warm and tuned on both sides, it is 1.9×.
7. **Use the static ONNX exporter** (`dynamo=False`, no dynamic axes) for QAIRT.
8. **Match versions.** Context binaries from QAIRT 2.50 need the 2.50 runtime on the board; the
   board's stock 2.42 runtime rejects them.
9. **Training stability.** The tokenizer diverged twice before switching to VAR-style residual
   quantization (detached residuals, one straight-through gradient) and a normalized encoder output.

---

## Limitations

- About half of each NPU story is Python between NPU calls. Moving that work into C, batching
  several stories per call, and requesting a higher NPU clock should help; none is done yet.
- Agreement with fp32 is measured per level with identical inputs. NPU-generated stories have
  not been scored by GPT-2 separately.
- Each preset was trained once; the base vs v2 difference rests on single runs.
- The NPU build was calibrated with 96 samples using plain post-training quantization.
- This implementation has not been compared with the official pretrained NCM checkpoint, and its
  perplexity is not comparable with the paper's numbers (different tokenizer and settings).

---

## Credits

- **Next-Concept Modeling**: arXiv [2610.14162](https://arxiv.org/abs/2610.14162).
- **[HazyResearch/ncm](https://github.com/HazyResearch/ncm)** (MIT): reference for the `v2` design choices.
- **TinyStories** dataset (Eldan & Li), via Hugging Face `roneneldan/TinyStories`.
- **Qualcomm AI Runtime (QAIRT)** and **Hexagon SDK**: required to build and profile the NPU
  version; download them from Qualcomm. No Qualcomm files are included in this repository.

Qualcomm, Hexagon and QAIRT are trademarks of Qualcomm Technologies, Inc. This is an independent
project, not affiliated with or endorsed by Qualcomm.
