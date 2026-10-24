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
| Result | **66 ms per story on the NPU, 2.9× faster than the best CPU setting** (191 ms) |

**[View the interactive report online](https://aurosiva.github.io/NCM_Littlestory_-qcs6490_NPU/)**: the full
comparison with charts (speed, where the time goes, sysMon NPU profiles, accuracy per level). The same page is in
[`report/ncm_npu_report.html`](report/ncm_npu_report.html).

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
11. [License](#license)

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

### Speed (Radxa Q6A, warm, 96 stories, 2 runs each, each run starting below 60 °C)

| setup | ms per story |
|---|---|
| CPU, tuned (4 onnxruntime threads, 1 numpy thread), 1 story at a time | 276 |
| CPU, tuned, 8 stories per batch | 191 |
| NPU, first version: `qnn_serve` pipe, 1 story per call | 125.6 |
| NPU, shared buffers | 105.3 |
| NPU, shared buffers, 4 stories per call | 99.5 |
| NPU, shared buffers, 2 batches of 4 in flight | 75.2 |
| **NPU, shared buffers, 2 batches of 8 in flight (fastest)** | **66.1** |

All NPU rows use the 16-bit mixed build and are pinned to the four fast cores
(`taskset -c 4-7`). The CPU rows are unpinned, which is their best case: pinned, the CPU took 305 ms
at one story per batch. An earlier unpinned round measured the first NPU version at 143 ms and the
8-bit build at 107 ms; the CPU at 808 ms with 8 onnxruntime threads and 586 ms with default settings.

### Where the time went, and what fixed it

Profiling the first version (`qnn-net-run --profiling_level basic` for the NPU calls, plus
`npu/profile_story.py` for the Python side) split its ~146 ms per story into:

| part | ms | |
|---|---|---|
| NPU computing (12 × 4.15 ms generator + 0.85 ms decoder) | ~51 | 35% |
| QNN call overhead (12 × 0.84 + 2.2 ms) | ~12 | 8% |
| Pipe to `qnn_serve` + 8/16-bit conversion in Python | ~37 | 25% |
| CPU math between levels (sampling, codebook head, adding levels) | ~46 | 32% |

The fixed per-call costs (~49 ms) were as large as the NPU's compute, and the NPU sat idle while
the CPU sampled. Three changes:

1. **Shared buffers** (126 → 105 ms). `npu/qnn_lib.c` is built as `libncm_qnn.so` and loaded by
   Python (`npu/qnn_fast.py`), so there is no server process and no pipe. Every input and output
   is allocated with rpcmem (memory the DSP maps directly), registered with QNN, and exposed to
   numpy as an array. The batched decoder also does its argmax on the NPU and returns 1,024 token
   ids instead of a 1 MB logits table: a decode call went from 6–8 ms to about 1 ms.
2. **Batching** (105 → 100 ms). `export_npu.py --batch B` stacks B stories on the first axis
   (`zin [B·128, d]`); all stories are at the same level, so they share the level embedding and
   the mask. On its own this saves little: the NPU needs about 4.75 ms per story per level, so 8
   stories take 8× as long, and only the small leftover call overhead is shared.
3. **Two batches in flight** (100 → 66 ms). `--lanes 2` loads the graphs twice (separate buffers)
   and runs one thread per batch. While one batch samples on the CPU, the NPU runs the other; the
   GIL is released during the NPU call and inside numpy. The CPU work (~45 ms per story) now hides
   behind the NPU work (~57 ms), which is the new limit. Running the lanes in parallel gives
   exactly the same tokens as running them one after the other (checked on 48 stories).

The trade-off is latency: a batch of 8 finishes after about 0.5 s. For one story as fast as
possible, use shared buffers with batch 1 (105 ms).

### Does the NPU do the work? (Qualcomm sysMon, 10 s captures)

| capture | NPU utilisation | NPU clock |
|---|---|---|
| nothing running | 0% | 365 MHz |
| NCM on the CPU | 0% | 365 MHz |
| NCM on the NPU (first version) | 23% | 941 MHz 94% of the time, peaks of 1306 MHz |

In the first version the NPU worked in short bursts and waited for the CPU in between. That idle
time is what the shared buffers and the two batches in flight removed (next table). These
captures were not pinned to the fast cores; pinned, the first version kept the NPU 45% busy.

### First version vs new version, inside the NPU

Second round, same session, pinned to the fast cores. NPU compute per story comes from
`qnn-net-run` profiling (48 calls per graph, real inputs); "NPU busy" is that compute divided by
the time per story.

| | first version | shared buffers | batch 8, 2 in flight |
|---|---|---|---|
| ms per story | 125.6 | 105.3 | 66.1 |
| NPU compute per story | 58 ms | 58 ms | 60 ms |
| **NPU busy** | **46%** | **55%** | **91%** |
| sysMon utilisation | 45% | 48% | 30%* |
| sysMon average NPU clock | 949 MHz | 950 MHz | 1246 MHz (up to 1440) |
| DDR traffic per story (sysMon) | 9 MB | 10 MB | 152 MB |

\* sysMon slowed this setup from 66 to about 200 ms per story, so its utilisation is not
representative. Its clock and DDR numbers are: a batch of 8 no longer fits in the 2 MB VTCM and
spills to DDR, and the NPU raises its clock to compensate. sysMon captures of an idle board and
of NCM on the CPU both stay at 0% and 364 MHz.

NPU time per call by batch size: generator 4.8 / 21.8 / 39.5 ms for 1 / 4 / 8 stories (4.8 / 5.5 /
4.9 ms per story), decoder 1.0 / 6.8 / 6.1 ms. The NPU's work per story does not shrink with
batching; only the time around it does. Every size uses 2 HVX threads.

### Accuracy of the quantized models

Given the same input, how often the quantized model picks the same code as the full-precision
model (16 stories; level 1 always matches):

| build | levels 2–12 | decoder tokens |
|---|---|---|
| NPU 16-bit mixed | 91–98% | 99.7% |
| NPU 16-bit mixed, batched rebuilds (batch 1, 4, 8) | 88–98% | 99.2–99.7% |
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
  qnn_serve.c        first version: C server that keeps the NPU models loaded, talks over a pipe
  qnn_lib.c          shared-buffer version: libncm_qnn.so, loaded in-process, rpcmem buffers
  qnn_fast.py        Python driver for libncm_qnn.so (numpy views of the NPU buffers)
  run_ncm_npu.py     NPU runner (--engine pipe|shm, --batch, --lanes) and accuracy check
  profile_story.py   per-stage timing of the NPU pipeline
  demo_ncm.py        live coarse-to-fine demo in the terminal; can record to JSON lines
  render_demo.py     turns a demo recording into a GIF and MP4
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

# batched graphs (B stories per call, decoder returns token ids); overrides are made automatically
python npu/export_npu.py --ckpt ckpt_v2 --out npu/build --batch 8
bash npu/build.sh 16 8 8    # → npu/build/gen_b8_a16w8mix.bin, dec_b8_a16w8mix.bin
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

# shared-buffer version
scp npu/qnn_lib.c npu/qnn_fast.py npu/profile_story.py npu/build/*_b8_a16w8mix.bin $B:~/ncm_npu/
ssh $B 'cd ~/ncm_npu && gcc -O2 -shared -fPIC -Iinc/QNN qnn_lib.c -ldl -o libncm_qnn.so'
```

### 3.3 Generate

```bash
cd ~/ncm_npu
~/ncm-venv/bin/python run_ncm_npu.py -n 4 --bits a16w8mix            # 4 stories
~/ncm-venv/bin/python run_ncm_npu.py -n 4 --bits a16w8mix --seed 0   # repeatable
~/ncm-venv/bin/python run_ncm_npu.py -n 100 --bits a16w8mix --out stories.txt
~/ncm-venv/bin/python run_ncm_npu.py --check 16 --bits a16w8mix      # agreement with fp32

# fastest: shared buffers, 8 stories per call, 2 batches in flight, fast cores only
taskset -c 4-7 ~/ncm-venv/bin/python run_ncm_npu.py -n 16 --engine shm --batch 8 --lanes 2 --bits a16w8mix

# where the time goes, per stage
taskset -c 4-7 ~/ncm-venv/bin/python profile_story.py -n 96                                   # first version
taskset -c 4-7 ~/ncm-venv/bin/python profile_story.py -n 96 --engine shm --batch 8 --lanes 2
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

With `--engine shm` the same loop runs without the pipe: `libncm_qnn.so` lives inside the Python
process, each tensor's buffer is rpcmem memory shared with the DSP, and Python quantizes straight
into it. The level embedding and mask for each level are quantized once at start-up.

To profile the NPU calls themselves, copy `qnn-net-run` and `qnn-profile-viewer` from
`$QAIRT_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4/` (plus `libQnnHtpNetRunExtensions.so` and
`libQnnHtpProfilingReader.so` from `lib/aarch64-ubuntu-gcc9.4/`) to the board and run a context
binary on recorded inputs with `--profiling_level basic`. Each run writes
`qnn-profiling-data_N.log`; open the numbered file (the unnumbered link it creates is broken).

---

## 4. Profile the NPU with sysMon

`sysMonAppLE` (Hexagon SDK) records the NPU's activity. Run it while stories are generating:

```bash
# board
cd ~/ncm_npu
taskset -c 4-7 ~/ncm-venv/bin/python run_ncm_npu.py --bits a16w8mix -n 150 --out /tmp/s.txt &
# new version instead: add  --engine shm --batch 1  (and -n 200)
sleep 5                                   # let it load the models and start generating
sudo ~/sysMonAppLE profiler --q6 cdsp --duration 10
wait
sudo chmod 644 /tmp/sysmon_cdsp.bin
ls -l /tmp/sysmon_cdsp.bin                # ~900 KB when the NPU was busy; ~9 KB means it captured nothing

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
- Profiling slowed the first version from 143 to about 177 ms per story.

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
   all 8 cores sat at 96% and the NPU run took 225 ms. With `OPENBLAS_NUM_THREADS=4` it took 143 ms.
6. **Compare like with like.** A cold, untuned CPU run first suggested the NPU was 5× faster.
   Warm and tuned on both sides, the first version was 1.9×.
7. **Pin benchmarks to the fast cores.** The QCS6490 has 4 slow and 4 fast CPU cores. The same
   NPU run took 120 ms on cores 4–7, 235 ms on cores 0–3 and anywhere in between unpinned. With
   `taskset -c 4-7` and a cool start (the board reaches 78–86 °C), runs repeat within 1%.
8. **Profile before optimizing.** The NPU computed for only a third of each story. The rest was the
   pipe, conversions and CPU math, so the gains came from the host code, not the model.
9. **Batching only pays when something overlaps.** Once call overhead is small, the NPU's time
   grows with the batch. Batching helped because it made each NPU call long enough to hide the
   CPU work of a second batch behind it.
10. **The DSP can hang.** Twice during testing every NPU call failed and the kernel logged
   `glink-edge: intent request timed out` every 10 s. The cDSP is in "attached" state, so Linux
   cannot restart it; only a reboot helped. The second time followed a batch-8, 2-lane run recorded
   by sysMon, whose NPU session took 11 s to close instead of 3–7 ms. Plain runs of the same
   setup close normally; the root cause is unknown. Avoid sysMon on the multi-lane setup.
11. **sysMon changes what it measures.** Sampling every 1 ms slowed the first version by ~10% but
   the batch-8, 2-lane setup by 3×. Cross-check with `qnn-net-run` profiling, which does not.
12. **Use the static ONNX exporter** (`dynamo=False`, no dynamic axes) for QAIRT.
13. **Match versions.** Context binaries from QAIRT 2.50 need the 2.50 runtime on the board; the
   board's stock 2.42 runtime rejects them.
14. **Training stability.** The tokenizer diverged twice before switching to VAR-style residual
   quantization (detached residuals, one straight-through gradient) and a normalized encoder output.

---

## Limitations

- The NPU calls (~57 ms per story, ~51 ms of it compute) are now the limit. The sampling and math between
  levels (~45 ms per story) still run in numpy; with fewer lanes or a faster NPU it would show
  again. Moving it into C, or the codebook head onto the NPU, has not been tried.
- No NPU performance mode is requested (`--perf_profile burst` changed the profiled call time by
  only ~5%).
- Batch 8 spills out of VTCM (15× the DDR traffic per story). A graph tiled to fit VTCM, or
  more HVX threads, might lower the ~58 ms of NPU compute per story; not tried.
- The 8-bit build was not rebuilt for batching or re-measured with pinning.
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

## License

The code in this repository is released under the [MIT License](LICENSE). It covers only the
files here: the Qualcomm SDKs, the TinyStories dataset and any trained weights keep their own terms.
