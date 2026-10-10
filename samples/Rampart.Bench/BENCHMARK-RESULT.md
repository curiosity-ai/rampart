# Benchmark results — Rampart C# vs ONNX Runtime

The question this answers: on the same model, how does the C# engine's forward pass compare with
ONNX Runtime, across sequence lengths, on one thread and on every core — and what does it allocate?

## Environment

Every run of the bench prints this block first. A result is only comparable with another whose
calibration is in the same band, so read it before the numbers.

```
  CPU            Intel(R) Xeon(R) Processor @ 2.80GHz  (4 logical cores)
  Memory         15.7 GiB total, 15.0 GiB available
  OS / runtime   Ubuntu 24.04.5 LTS / .NET 10.0.12 (X64)
  ONNX Runtime   1.23.2
  SIMD           AVX2, FMA, AVX-512F, AVX-512BW, Vector512.IsHardwareAccelerated=False, Vector<T>=256-bit
  GEMM kernel    512-bit (12x32)
== Calibration (single thread) ==
  GEMM 96x32x384, cache-resident     ~100–118 GFLOPS best of 15  (~95–104 median)
  Memory read, 256 MB                  ~9–10 GB/s
  Memory copy, 256 MB                  ~9.5–10.5 GB/s   (bytes read + written)
  CPU steal during calibration         0.0–1.0%
```

A cloud VM: 4 vCPUs with no SMT siblings, one socket. `Vector512.IsHardwareAccelerated` is false on
this Xeon (the runtime avoids 512-bit vectors where it fears downclocking), but `Avx512F.IsSupported`
is true and the engine's GEMM uses the 512-bit kernel on that basis (see the A/B below).

How to read the calibration:

- **GEMM** is the engine's own micro-kernel on a cache-resident shape: pure FMA throughput. The *best*
  sample is what a core can do. The *median* also picks up the scheduler: an unpinned thread that
  gets moved between cores mid-sample reads ~30% low now and then (median ~70). Pinned to any one
  core it never does, and the dip does not show in the model latencies, so treat a low median
  with a normal best as noise. A low *best* means a slower or throttled host.
- **Memory** is single-thread streaming over 256 MB, far past the last-level cache. The model's
  weights are ~5 MB, so this matters less than GEMM, but it tells a different host apart.
- **Steal** above a few percent means the hypervisor is taking CPU away from this VM. Discard the run.

## Method

- **One process, interleaved.** All four engines are loaded into the same process and measured in
  5 rounds. Each round runs every engine once more, rotating who goes first, with a 50 ms idle gap
  before each engine so the previous one's spinning threads have parked. A VM that slows down or
  gets rescheduled mid-run then slows every engine alike instead of whichever was running.
- **Warm-up** is time-based (1.5 s per engine) because the tiered JIT keeps recompiling hot methods
  in the background for a while after start-up. A fixed number of warm-up calls was not enough.
- **Input**: `[CLS]` + N−2 content tokens from the fixture corpus + `[SEP]`, N = 32 … 512 (512 is one
  full window). Calls per engine per round: 40 / 30 / 16 / 8 / 4, i.e. n = 200 / 150 / 80 / 40 / 20.
- **Engines**: Rampart C# with `MaxDegreeOfParallelism = 1`, and with the default pool (4 threads).
  ONNX Runtime 1.23.2 with full graph optimization, `intra_op_num_threads = 1`, and with 4 threads
  and its spinning thread pool (its default, and its fastest configuration).
- **Allocations**: `GC.GetTotalAllocatedBytes(precise: true)` around each engine's batch. The timer is
  `Stopwatch.GetTimestamp`, so nothing the bench does is charged to an engine.

## Results

Two independent runs of the default mode, each a single process. Median ms per forward pass, with the
minimum in parentheses.

**Run A** (calibration GEMM 102.8 GFLOPS median, read 10.1 GB/s, steal 0.3%)

| tokens | C# 1 thread | ORT 1 thread | C# 4 threads | ORT 4 threads (spinning) |
|---:|---:|---:|---:|---:|
| 32 | 9.92 (8.79) | 13.32 (12.37) | 4.55 (3.18) | 6.37 (5.29) |
| 64 | 19.26 (16.92) | 20.62 (19.66) | 8.43 (6.15) | 9.41 (7.24) |
| 128 | 37.93 (32.78) | 37.23 (35.20) | 14.52 (11.15) | 17.11 (15.35) |
| 256 | 86.50 (77.90) | 79.83 (74.92) | 30.83 (26.70) | 29.24 (23.24) |
| 512 | 196.46 (188.05) | 207.51 (188.02) | 65.38 (51.34) | 80.34 (70.88) |

**Run B** (calibration GEMM 71.6 GFLOPS median — the scheduler dip described above, latencies normal;
read 9.6 GB/s, steal 0.9%)

| tokens | C# 1 thread | ORT 1 thread | C# 4 threads | ORT 4 threads (spinning) |
|---:|---:|---:|---:|---:|
| 32 | 10.23 (8.65) | 14.14 (12.74) | 5.12 (3.47) | 7.26 (5.08) |
| 64 | 19.73 (17.40) | 21.74 (19.49) | 8.52 (6.27) | 11.79 (7.96) |
| 128 | 40.43 (34.96) | 39.34 (36.33) | 15.44 (11.20) | 17.63 (13.17) |
| 256 | 87.28 (72.40) | 84.80 (75.70) | 29.06 (27.17) | 31.35 (22.65) |
| 512 | 209.02 (176.70) | 204.91 (184.14) | 66.73 (52.97) | 77.38 (68.88) |

**C# ÷ ORT at the median** (below 1.00 = C# faster)

| tokens | 1 thread, A | 1 thread, B | 4 threads, A | 4 threads, B |
|---:|---:|---:|---:|---:|
| 32 | 0.74 | 0.72 | 0.71 | 0.70 |
| 64 | 0.93 | 0.91 | 0.90 | 0.72 |
| 128 | 1.02 | 1.03 | 0.85 | 0.88 |
| 256 | 1.08 | 1.03 | 1.05 | 0.93 |
| 512 | 0.95 | 1.02 | 0.81 | 0.86 |

**Throughput** (Run B, tokens/s at the median): C# 1 thread 2,450–3,250, ORT 1 thread 2,260–3,250;
C# 4 threads 6,250–8,810, ORT 4 threads 4,400–8,170.

**Memory per call**

| Engine | Allocated per forward pass |
|---|---:|
| Rampart C#, 1 or 4 threads (`ComputeLogits` into a span) | 0 B |
| ONNX Runtime via its C# API, 1 or 4 threads | ~17.7 KB (input/output tensors and result arrays) |

Native memory is not in those figures. The working set (whole bench process, both engines and the
test fixture loaded) went from 786 MB to 965 MB over the sweep.

**End to end** — `ChatGuard.Protect` on a 30-token chat message, i.e. heuristics, tokenizer, model,
BIO merge, policy and placeholders: 10.7 ms on one thread, 4.4–4.8 ms on four. Loading the model from
the embedded resource: 0.3–0.5 s.

## AVX-512 A/B

`--avx512-ab` runs the C# sweep in two child processes started from one parent, first with
`DOTNET_EnableAVX512=0` (the 256-bit 6×16 kernel), then with the default (the 512-bit 12×32 kernel).
On .NET 10 that one variable covers the whole AVX-512 family. The older per-ISA
`DOTNET_EnableAVX512F=0` is silently ignored, which the bench's SIMD line makes visible.

| tokens | 1 thread, AVX2 | 1 thread, AVX-512 | speed-up | 4 threads, AVX2 | 4 threads, AVX-512 | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 17.68 | 9.91 | 1.78× | 6.61 | 4.85 | 1.36× |
| 64 | 28.74 | 17.59 | 1.63× | 11.88 | 8.61 | 1.38× |
| 128 | 59.55 | 36.55 | 1.63× | 21.59 | 14.79 | 1.46× |
| 256 | 126.01 | 81.90 | 1.54× | 44.02 | 31.73 | 1.39× |
| 512 | 317.70 | 191.48 | 1.66× | 104.16 | 68.13 | 1.53× |

Calibration GEMM: 61.7 GFLOPS with AVX2, 101.0 with AVX-512. `ChatGuard.Protect`: 17.2 → 11.6 ms on
one thread, 6.4 → 5.4 ms on four. AVX-512 is worth having even where the runtime does not call it
hardware-accelerated. Without it the engine runs about 1.5× slower than ORT on one thread: ORT has
its own AVX-512 kernels, and these numbers say nothing about how ORT would fare without them.

## Assessment

- **Short inputs, where chat traffic lives (32–64 tokens): C# is faster**, ~0.72× on one thread at
  32 tokens and 0.70–0.90× on four. Fixed per-call overhead dominates there, and the C# engine has
  almost none: no tensor marshalling, no session machinery, no allocation.
- **128–512 tokens on one thread: parity, not a win.** The ratio sits between 0.95 and 1.08 and
  changes sign between runs, which is within this VM's noise. The time there is GEMM, and both
  engines run AVX-512 FMA kernels near the core's throughput.
- **Four threads: C# is faster at every length but 256**, where the two runs disagree (1.05 and 0.93).
  ORT's spinning pool keeps its threads hot. Rampart's pool spins for ~0.5 ms and then parks, so it
  does not burn idle cores between requests the way ORT's does.
- **Allocation: 0 B per forward pass** against ~17.7 KB for ORT's C# API, so a server running the C#
  engine puts no load on the GC from inference.
- Not measured here: other CPUs (AMD, ARM/NEON, Xeons without AVX-512, where the 256-bit kernel
  runs), batched inputs (neither engine batches, as the reference does not), ORT with a different
  execution provider.

## Reproduce

```bash
dotnet build -c Release
B=samples/Rampart.Bench/bin/Release/net10.0/Rampart.Bench.dll
dotnet $B                          # the sweep above: 4 engines, one process, interleaved
dotnet $B --single | --multi       # only the 1-thread / only the 4-thread pair
dotnet $B --avx512-ab              # the A/B above
dotnet $B --env                    # environment + calibration only (check the box before a long run)
```

Run the comparison in one shell session and don't start anything else while it runs. Before
comparing with an earlier result, check that the calibration block is in the same band.
