using System.Diagnostics;
using System.Text.RegularExpressions;
using Rampart;
using Rampart.Model;
using Rampart.Tests;

// Latency of the C# engine against ONNX Runtime (CPU, full graph optimization) on the same model,
// plus the end-to-end guard.
//
//   dotnet run -c Release --project samples/Rampart.Bench                    # latency table, all cores
//   dotnet run -c Release --project samples/Rampart.Bench -- --single        # both engines on one thread
//   dotnet run -c Release --project samples/Rampart.Bench -- --ort-spinning  # ONNX Runtime alone, spinning pool
//   dotnet run -c Release --project samples/Rampart.Bench -- --profile [--short] [--single]  # time per stage
//   dotnet run -c Release --project samples/Rampart.Bench -- --alloc [--single]  # bytes allocated per call
//   dotnet run -c Release --project samples/Rampart.Bench -- --alloc-parts   # bytes per kernel
//   dotnet run -c Release --project samples/Rampart.Bench -- --gemm | --attention  # kernel throughput
bool single = args.Contains("--single");
var model = single ? RampartModel.Load(new RampartModelOptions { MaxDegreeOfParallelism = 1 }) : RampartModel.Default;
var tokenizer = model.Tokenizer;
const string message = "My name is Alex Rivera and I live at 12 Main Street, Springfield. Call me at (202) 555-0188.";
var longIds = tokenizer.Encode(string.Join(" ", Enumerable.Repeat(message, 40)), addSpecialTokens: false);
int[] Window(int content) => [tokenizer.ClsId, .. longIds.AsSpan(0, content), tokenizer.SepId];

if (args.Contains("--ort-spinning"))
{
    // ONNX Runtime alone, with its default (spinning) intra-op pool: the fair multi-threaded baseline.
    foreach (var (name, ids) in new[] { ("message", tokenizer.Encode(message)), ("128 tokens", Window(126)), ("512 tokens", Window(510)) })
        Console.WriteLine($"  {name,-11} ({ids.Length,3} tokens)  ONNX Runtime (spinning): {Measure(() => OnnxReference.Shared.RunLogitsSpinning(ids)),7:F2} ms");
    return;
}

if (args.Contains("--gemm"))
{
    // Single-thread GEMM throughput on the encoder's shapes.
    var rnd = new Random(1);
    float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() - 0.5f; return a; }
    foreach (var (name, m, n, k, lda, ldb, ldc) in new[]
    {
        ("linear 512x384 <- K=384, 32-col panel", 512, 32, 384, 384, 32, 384),
        ("linear 512x1536 <- K=384, 32-col panel", 512, 32, 384, 384, 32, 1536),
        ("linear 512x384 <- K=1536, 32-col panel", 512, 32, 1536, 1536, 32, 384),
        ("scores 48x512, K=64 (head view)", 48, 512, 64, 384, 512, 512),
        ("scores 96x512, K=32, ldc 512", 96, 512, 32, 384, 512, 512),
        ("scores 96x512, K=32, ldc 528", 96, 512, 32, 384, 512, 528),
        ("scores 96x512, K=32, ldb 528, ldc 528", 96, 512, 32, 384, 528, 528),
        ("scores 96x512, K=32, lda 32 (packed Q)", 96, 512, 32, 32, 512, 512),
        ("scores 96x512, K=64, ldc 512", 96, 512, 64, 384, 512, 512),
        ("context 48x64, K=512 (V in place)", 48, 64, 512, 512, 384, 384),
    })
    {
        var a = R(m * lda); var b = R(k * ldb); var c = new float[m * ldc]; var bias = R(n);
        double flops = 2.0 * m * n * k;
        var t = Measure(() => { for (int r = 0; r < 10; r++) Rampart.Model.Gemm.Multiply(a, lda, b, ldb, c, ldc, m, n, k, bias); }, 30) / 10;
        Console.WriteLine($"  {name,-42} {t * 1000,8:F1} us  {flops / t / 1e6,6:F1} GFLOPS");
    }
    var lin = RampartModel.Default.Weights.Layers[0].Output; // K = 1536, N = 384
    var dq = new float[lin.K * Rampart.Model.Q4Linear.Tile];
    var td = Measure(() => { for (int t = 0; t < lin.Tiles; t++) lin.DequantizeTile(t, dq); }, 30);
    Console.WriteLine($"  {"dequantize 1536 x 384 (12 panels)",-42} {td * 1000,8:F1} us  {td * 1e6 / (lin.K * lin.N),6:F3} ns/weight");
    var rows = R(48 * 512);
    var ts = Measure(() => { for (int r = 0; r < 48; r++) Rampart.Model.Kernels.SoftmaxInPlace(rows.AsSpan(r * 512, 512)); }, 30);
    Console.WriteLine($"  {"softmax 48 rows x 512",-42} {ts * 1000,8:F1} us  {ts * 1e6 / (48 * 512),6:F2} ns/element");
    var tm = Measure(() => { for (int r = 0; r < 48; r++) System.Numerics.Tensors.TensorPrimitives.Max(rows.AsSpan(r * 512, 512)); }, 30);
    Console.WriteLine($"  {"  of which TensorPrimitives.Max",-42} {tm * 1000,8:F1} us");
    var tx = Measure(() => { for (int r = 0; r < 48; r++) System.Numerics.Tensors.TensorPrimitives.Multiply(rows.AsSpan(r * 512, 512), 1.0000001f, rows.AsSpan(r * 512, 512)); }, 30);
    Console.WriteLine($"  {"  of which TensorPrimitives.Multiply",-42} {tx * 1000,8:F1} us");
    var g = R(512 * 1536);
    var g0 = (float[])g.Clone();
    var tc = Measure(() => g0.CopyTo(g), 30);
    var tg = Measure(() => { g0.CopyTo(g); Rampart.Model.Kernels.Gelu(g); }, 30) - tc;
    Console.WriteLine($"  {"gelu 512 x 1536",-42} {tg * 1000,8:F1} us  {tg * 1e6 / g.Length,6:F2} ns/element");
    return;
}

if (args.Contains("--attention"))
{
    // One layer of attention at 512 tokens, single thread, timed step by step.
    const int T = 512, W = 384, H = 12, D = 32, B = 96;
    var rnd = new Random(1);
    float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() - 0.5f; return a; }
    var q = R(T * W); var k = R(T * W); var v = R(T * W);
    var kt = new float[H * T * D]; var vh = new float[H * T * D]; var probs = new float[H * T * T]; var ctx = new float[T * W];
    double pack = Measure(() =>
    {
        for (int j = 0; j < T; j++)
            for (int h = 0; h < H; h++)
            {
                for (int d = 0; d < D; d++) kt[h * T * D + d * T + j] = k[j * W + h * D + d] * 0.35f;
                v.AsSpan(j * W + h * D, D).CopyTo(vh.AsSpan(h * T * D + j * D, D));
            }
    }, 20);
    double scores = Measure(() => { for (int h = 0; h < H; h++) for (int i0 = 0; i0 < T; i0 += B) Rampart.Model.Gemm.Multiply(q.AsSpan(i0 * W + h * D), W, kt.AsSpan(h * T * D, T * D), T, probs.AsSpan((h * T + i0) * T, Math.Min(B, T - i0) * T), T, Math.Min(B, T - i0), T, D); }, 20);
    double soft = Measure(() => { for (int r = 0; r < H * T; r++) Rampart.Model.Kernels.SoftmaxInPlace(probs.AsSpan(r * T, T)); }, 20);
    double context = Measure(() => { for (int h = 0; h < H; h++) for (int i0 = 0; i0 < T; i0 += B) Rampart.Model.Gemm.Multiply(probs.AsSpan((h * T + i0) * T, Math.Min(B, T - i0) * T), T, vh.AsSpan(h * T * D, T * D), D, ctx.AsSpan(i0 * W + h * D), W, Math.Min(B, T - i0), D, T); }, 20);
    double flops = 2.0 * T * T * D * H;
    // Kᵀ packed in 32-key panels: kp[h][jb][d][32], so each column block of B is one contiguous 4 KB panel.
    var kp = new float[H * T * D];
    double packPanels = Measure(() =>
    {
        for (int j = 0; j < T; j++)
            for (int h = 0; h < H; h++)
                for (int d = 0; d < D; d++) kp[h * T * D + (j / 32) * D * 32 + d * 32 + j % 32] = k[j * W + h * D + d] * 0.35f;
    }, 20);
    double scoresPanels = Measure(() =>
    {
        for (int h = 0; h < H; h++)
            for (int i0 = 0; i0 < T; i0 += B)
                for (int jb = 0; jb < T / 32; jb++)
                    Rampart.Model.Gemm.Multiply(q.AsSpan(i0 * W + h * D), W, kp.AsSpan(h * T * D + jb * D * 32, D * 32), 32, probs.AsSpan((h * T + i0) * T + jb * 32), T, Math.Min(B, T - i0), 32, D);
    }, 20);
    Console.WriteLine($"  pack K/V  {pack,7:F2} ms   (panels: {packPanels:F2} ms)");
    Console.WriteLine($"  scores, panel-packed Kt {scoresPanels,7:F2} ms  {flops / scoresPanels / 1e6,6:F1} GFLOPS");
    Console.WriteLine($"  scores    {scores,7:F2} ms  {flops / scores / 1e6,6:F1} GFLOPS");
    Console.WriteLine($"  softmax   {soft,7:F2} ms");
    Console.WriteLine($"  context   {context,7:F2} ms  {flops / context / 1e6,6:F1} GFLOPS");
    return;
}

if (args.Contains("--alloc-parts"))
{
    var w = model.Weights;
    var x = new float[30 * 1536]; var y = new float[30 * 1536];
    void Probe(string name, Action a)
    {
        for (int i = 0; i < 20; i++) a();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) a();
        Console.WriteLine($"  {name,-30} {(GC.GetAllocatedBytesForCurrentThread() - before) / 100,8} bytes/call");
    }
    Probe("Q4Linear.Forward", () => w.Layers[0].Query.Forward(x, 30, y, 1));
    Probe("Classifier.Forward (remainder)", () => w.Classifier.Forward(x, 30, y, 1));
    Probe("AddLayerNorm", () => Rampart.Model.Kernels.AddLayerNorm(x, x, y, 30, w.Layers[0].AttentionNorm.Gamma, w.Layers[0].AttentionNorm.Beta, 1e-12f));
    Probe("Gelu", () => Rampart.Model.Kernels.Gelu(y.AsSpan(0, 30 * 1536)));
    Probe("Softmax", () => Rampart.Model.Kernels.SoftmaxInPlace(y.AsSpan(0, 30)));
    Probe("Heuristics.Detect", () => Rampart.Detection.Heuristics.Detect(message));
    var ids30 = tokenizer.Encode(message);
    var lg = new float[512 * 35];
    Probe("ComputeLogits(30, span)", () => model.ComputeLogits(ids30, lg));
    return;
}

if (args.Contains("--alloc"))
{
    // Bytes allocated per call on all threads, after warm-up.
    var analyzerA = new RampartAnalyzer(new RampartOptions { Model = model });
    var guardA = new ChatGuard(analyzerA);
    var ids30 = tokenizer.Encode(message);
    var ids512 = Window(510);
    var logits = new float[512 * model.Labels.Count];
    var ner = new Rampart.Ner.NerDetector(model);
    foreach (var (name, action) in new (string, Action)[]
    {
        ("ComputeLogits(30, span)", () => model.ComputeLogits(ids30, logits)),
        ("ComputeLogits(512, span)", () => model.ComputeLogits(ids512, logits)),
        ("CountTokens(message)", () => tokenizer.CountTokens(message)),
        ("Encode(message)", () => tokenizer.Encode(message)),
        ("NerDetector.Detect(message)", () => ner.Detect(message)),
        ("Heuristics(message)", () => Rampart.Detection.Heuristics.Detect(message)),
        ("Analyzer.Detect(message)", () => analyzerA.Detect(message)),
        ("Guard.Protect(message)", () => guardA.Protect(message)),
    })
    {
        for (int i = 0; i < 50; i++) action();
        const int n = 20;
        long before = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < n; i++) action();
        long per = (GC.GetTotalAllocatedBytes(precise: true) - before) / n;
        Console.WriteLine($"  {name,-26} {per,12:N0} bytes/call");
    }
    return;
}

if (args.Contains("--profile"))
{
    var ids = args.Contains("--short") ? tokenizer.Encode(message) : Window(510);
    var timer = new StageTimer();
    Measure(() => model.ComputeLogits(ids));
    int runs = ids.Length < 100 ? 200 : 20;
    for (int i = 0; i < runs; i++)
    {
        timer.Restart();
        model.ComputeLogits(ids, timer);
    }
    Console.WriteLine($"Time per stage, {ids.Length} tokens, mean of {runs} runs (a stage includes everything since the previous trace point):");
    foreach (var (stage, total) in timer.Totals.OrderByDescending(kv => kv.Value)) Console.WriteLine($"  {total / runs,8:F2} ms  {stage}");
    return;
}

Console.WriteLine($"{(single ? "1 thread (both engines)" : $"{Environment.ProcessorCount} cores")}, AVX-512: {System.Runtime.Intrinsics.X86.Avx512F.IsSupported}; median latency");
var ort = OnnxReference.Shared;
foreach (var (name, ids) in new[] { ("message", tokenizer.Encode(message)), ("128 tokens", Window(126)), ("512 tokens", Window(510)) })
{
    var csharp = Measure(() => model.ComputeLogits(ids));
    var onnx = single ? Measure(() => ort.RunLogitsSingleThread(ids)) : Measure(() => ort.RunLogits(ids));
    Console.WriteLine($"  {name,-11} ({ids.Length,3} tokens)  Rampart C#: {csharp,7:F2} ms   ONNX Runtime: {onnx,7:F2} ms");
}

var analyzer = new RampartAnalyzer(new RampartOptions { Model = model });
Console.WriteLine($"  ChatGuard.Protect (message, heuristics + model + policy): {Measure(() => new ChatGuard(analyzer).Protect(message)):F2} ms");

var sw = Stopwatch.StartNew();
GC.KeepAlive(RampartModel.Load());
Console.WriteLine($"  Load from the embedded resource: {sw.Elapsed.TotalMilliseconds:F0} ms");

static double Measure(Action action, int iterations = 50)
{
    // Warm up past tiered JIT promotion (tier-0 to tier-1 needs ~30 calls plus a background delay).
    var warm = Stopwatch.StartNew();
    while (warm.ElapsedMilliseconds < 1500) action();
    var times = new double[iterations];
    for (int i = 0; i < iterations; i++)
    {
        var sw = Stopwatch.StartNew();
        action();
        times[i] = sw.Elapsed.TotalMilliseconds;
    }
    Array.Sort(times);
    return times[iterations / 2];
}

sealed partial class StageTimer : IForwardTrace
{
    private readonly Stopwatch _sw = Stopwatch.StartNew();

    public Dictionary<string, double> Totals { get; } = [];

    public void Restart() => _sw.Restart();

    public bool CapturesTensors => false;

    public void Record(string onnxName, ReadOnlySpan<float> data)
    {
        var stage = LayerIndex().Replace(onnxName, "layer.*");
        Totals[stage] = Totals.GetValueOrDefault(stage) + _sw.Elapsed.TotalMilliseconds;
        _sw.Restart();
    }

    [GeneratedRegex(@"layer\.\d+")]
    private static partial Regex LayerIndex();
}
