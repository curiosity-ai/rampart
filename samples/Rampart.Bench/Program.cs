using System.Diagnostics;
using System.Text.RegularExpressions;
using Rampart;
using Rampart.Model;
using Rampart.Tests;

// Latency of the C# engine against ONNX Runtime (CPU, full graph optimization) on the same model,
// plus the end-to-end guard.
//
//   dotnet run -c Release --project samples/Rampart.Bench               # latency table
//   dotnet run -c Release --project samples/Rampart.Bench -- --profile  # time per encoder stage (512 tokens)
var model = RampartModel.Default;
var tokenizer = model.Tokenizer;
const string message = "My name is Alex Rivera and I live at 12 Main Street, Springfield. Call me at (202) 555-0188.";
var longIds = tokenizer.Encode(string.Join(" ", Enumerable.Repeat(message, 40)), addSpecialTokens: false);
int[] Window(int content) => [tokenizer.ClsId, .. longIds.AsSpan(0, content), tokenizer.SepId];

if (args.Contains("--profile"))
{
    var ids = Window(510);
    var timer = new StageTimer();
    Measure(() => model.ComputeLogits(ids));
    const int runs = 20;
    for (int i = 0; i < runs; i++)
    {
        timer.Restart();
        model.ComputeLogits(ids, timer);
    }
    Console.WriteLine($"Time per stage, 512 tokens, mean of {runs} runs (a stage includes everything since the previous trace point):");
    foreach (var (stage, total) in timer.Totals.OrderByDescending(kv => kv.Value)) Console.WriteLine($"  {total / runs,8:F2} ms  {stage}");
    return;
}

Console.WriteLine($"{Environment.ProcessorCount} cores, AVX-512: {System.Runtime.Intrinsics.X86.Avx512F.IsSupported}; median latency");
var ort = OnnxReference.Shared;
foreach (var (name, ids) in new[] { ("message", tokenizer.Encode(message)), ("128 tokens", Window(126)), ("512 tokens", Window(510)) })
{
    var csharp = Measure(() => model.ComputeLogits(ids));
    var onnx = Measure(() => ort.RunLogits(ids));
    Console.WriteLine($"  {name,-11} ({ids.Length,3} tokens)  Rampart C#: {csharp,7:F2} ms   ONNX Runtime: {onnx,7:F2} ms");
}

var analyzer = new RampartAnalyzer();
Console.WriteLine($"  ChatGuard.Protect (message, heuristics + model + policy): {Measure(() => new ChatGuard(analyzer).Protect(message)):F2} ms");

var sw = Stopwatch.StartNew();
GC.KeepAlive(RampartModel.Load());
Console.WriteLine($"  Load from the embedded resource, weights expanded: {sw.Elapsed.TotalMilliseconds:F0} ms");
sw.Restart();
GC.KeepAlive(RampartModel.Load(new RampartModelOptions { KeepWeightsQuantized = true }));
Console.WriteLine($"  Load from the embedded resource, weights kept 4-bit: {sw.Elapsed.TotalMilliseconds:F0} ms");

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

    public void Record(string onnxName, ReadOnlySpan<float> data)
    {
        var stage = LayerIndex().Replace(onnxName, "layer.*");
        Totals[stage] = Totals.GetValueOrDefault(stage) + _sw.Elapsed.TotalMilliseconds;
        _sw.Restart();
    }

    [GeneratedRegex(@"layer\.\d+")]
    private static partial Regex LayerIndex();
}
