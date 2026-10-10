using System.Diagnostics;
using Rampart;
using Rampart.Model;
using Rampart.Tests;
using Rampart.Tokenization;

/// <summary>
/// The engine comparison: a token-count sweep with every engine measured in this one process, in
/// interleaved rounds — each round runs every engine once more, in a rotating order, with an idle gap
/// before each engine so the previous one's spinning threads have parked. A VM that slows down or gets
/// rescheduled mid-run then slows every engine alike instead of whichever happened to run at that moment.
/// </summary>
internal static class Compare
{
    private sealed record Engine(string Name, bool Ort, Action<int[]> Run);

    private static readonly int[] Lengths = [32, 64, 128, 256, 512];
    private const int Rounds = 5;
    private static readonly TimeSpan Gap = TimeSpan.FromMilliseconds(50);

    /// <summary>Calls per engine per round: fewer as the sequence (and each call) grows.</summary>
    private static int CallsPerRound(int tokens) => tokens switch { <= 32 => 40, <= 64 => 30, <= 128 => 16, <= 256 => 8, _ => 4 };

    public static void Run(string[] args, int[] corpusIds, WordPieceTokenizer tokenizer)
    {
        bool onlySingle = args.Contains("--single"), onlyMulti = args.Contains("--multi"), csharpOnly = args.Contains("--csharp-only");
        var single = RampartModel.Load(new RampartModelOptions { MaxDegreeOfParallelism = 1 });
        var multi = RampartModel.Default;
        var logits = new float[512 * single.Labels.Count];
        var ort = OnnxReference.Shared;
        int cores = Environment.ProcessorCount;

        var engines = new List<Engine>();
        if (!onlyMulti)
        {
            engines.Add(new("Rampart C#, 1 thread", false, ids => single.ComputeLogits(ids, logits)));
            if (!csharpOnly) engines.Add(new("ONNX Runtime, 1 thread", true, ids => ort.RunLogitsSingleThread(ids)));
        }
        if (!onlySingle)
        {
            engines.Add(new($"Rampart C#, {cores} threads", false, ids => multi.ComputeLogits(ids, logits)));
            if (!csharpOnly) engines.Add(new($"ONNX Runtime, {cores} threads (spinning)", true, ids => ort.RunLogitsSpinning(ids)));
        }

        Console.WriteLine($"== Latency sweep: {engines.Count} engines in this process, {Rounds} interleaved rounds, {Gap.TotalMilliseconds:F0} ms gap between engines ==");
        Console.WriteLine("   (ms per forward pass over [CLS] + content + [SEP]; median, then min; n = calls measured)");
        Console.WriteLine();

        long workingSetBefore = Environment.WorkingSet;
        // Warm-up: tier every engine's hot path up before anything is measured.
        var warmIds = Ids(corpusIds, tokenizer, 128);
        foreach (var e in engines)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < 1500) e.Run(warmIds);
        }

        var header = $"  {"tokens",6} | " + string.Join(" | ", engines.Select(e => $"{e.Name,-34}"));
        Console.WriteLine(header);
        Console.WriteLine("  " + new string('-', header.Length - 2));

        var allocated = new long[engines.Count];
        var calls = new long[engines.Count];
        var medians = new Dictionary<(int, int), double>();
        foreach (int tokens in Lengths)
        {
            var ids = Ids(corpusIds, tokenizer, tokens);
            int perRound = CallsPerRound(tokens);
            var samples = engines.Select(_ => new List<double>(Rounds * perRound)).ToArray();
            foreach (var e in engines) e.Run(ids); // first touch at this length

            for (int round = 0; round < Rounds; round++)
            {
                for (int step = 0; step < engines.Count; step++)
                {
                    int ei = (step + round) % engines.Count; // rotate who goes first
                    var engine = engines[ei];
                    Thread.Sleep(Gap);
                    long before = GC.GetTotalAllocatedBytes(precise: true);
                    for (int c = 0; c < perRound; c++)
                    {
                        long t0 = Stopwatch.GetTimestamp(); // not StartNew: a Stopwatch is 40 B, charged to the engine
                        engine.Run(ids);
                        samples[ei].Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                    }
                    allocated[ei] += GC.GetTotalAllocatedBytes(precise: true) - before;
                    calls[ei] += perRound;
                }
            }

            var cells = new List<string>();
            for (int ei = 0; ei < engines.Count; ei++)
            {
                var s = samples[ei];
                s.Sort();
                double median = s[s.Count / 2];
                medians[(tokens, ei)] = median;
                cells.Add($"{median,8:F2} ({s[0],7:F2}) n={s.Count,-4}       ");
            }
            Console.WriteLine($"  {tokens,6} | " + string.Join(" | ", cells.Select(c => $"{c,-34}")));
        }

        Console.WriteLine();
        Console.WriteLine("  Throughput at the median, tokens per second:");
        foreach (int tokens in Lengths)
        {
            Console.WriteLine($"  {tokens,6} | " + string.Join(" | ", engines.Select((e, ei) => $"{tokens / medians[(tokens, ei)] * 1000,34:N0}")));
        }

        Console.WriteLine();
        Console.WriteLine("  C# / ONNX Runtime (median; below 1.00 = C# faster):");
        for (int ei = 0; ei + 1 < engines.Count; ei++)
        {
            if (engines[ei].Ort || !engines[ei + 1].Ort) continue;
            var ratios = Lengths.Select(t => $"{t}: {medians[(t, ei)] / medians[(t, ei + 1)]:F2}");
            Console.WriteLine($"    {engines[ei].Name,-24} vs {engines[ei + 1].Name,-34} {string.Join("  ", ratios)}");
        }

        Console.WriteLine();
        Console.WriteLine("  Memory:");
        for (int ei = 0; ei < engines.Count; ei++)
        {
            string note = engines[ei].Ort ? "  (includes the ORT C# wrapper's tensors and result arrays)" : "";
            Console.WriteLine($"    {engines[ei].Name,-40} {allocated[ei] / Math.Max(1, calls[ei]),10:N0} bytes allocated per call{note}");
        }
        Console.WriteLine($"    Working set {workingSetBefore / 1048576.0:F0} MB before warm-up, {Environment.WorkingSet / 1048576.0:F0} MB after the sweep; managed heap {GC.GetTotalMemory(false) / 1048576.0:F0} MB");

        Console.WriteLine();
        Console.WriteLine("  End to end (chat message, heuristics + model + policy + placeholders):");
        const string message = "My name is Alex Rivera and I live at 12 Main Street, Springfield. Call me at (202) 555-0188.";
        foreach (var (name, model) in new[] { ("1 thread", single), ($"{cores} threads", multi) })
        {
            if ((onlySingle && model == multi) || (onlyMulti && model == single)) continue;
            var analyzer = new RampartAnalyzer(new RampartOptions { Model = model });
            Thread.Sleep(Gap);
            // What a call to RampartCloaker.CloakAsync sends: analyze, then swap in the placeholders.
            Console.WriteLine($"    Analyze + cloak, {name,-10} {Median(() => new RampartCloaker().Cloak(message, analyzer.Analyze(message))),7:F2} ms");
        }
        var load = Stopwatch.StartNew();
        GC.KeepAlive(RampartModel.Load());
        Console.WriteLine($"    Model load from the embedded resource: {load.Elapsed.TotalMilliseconds:F0} ms");
    }

    private static double Median(Action action)
    {
        var warm = Stopwatch.StartNew();
        while (warm.ElapsedMilliseconds < 1500) action();
        var times = new double[50];
        for (int i = 0; i < times.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            action();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[times.Length / 2];
    }

    private static int[] Ids(int[] corpus, WordPieceTokenizer tokenizer, int tokens) => [tokenizer.ClsId, .. corpus.AsSpan(0, tokens - 2), tokenizer.SepId];

    /// <summary>
    /// Runs the C# sweep twice in child processes — AVX-512 disabled, then enabled — one after the
    /// other from this process. (.NET 10 has one knob for the whole AVX-512 family; the older
    /// per-ISA DOTNET_EnableAVX512F is ignored.) Both halves
    /// see the same machine state.
    /// </summary>
    public static void Avx512AB()
    {
        var dll = Environment.GetCommandLineArgs()[0];
        foreach (var (label, disable) in new[] { ("AVX-512 disabled (DOTNET_EnableAVX512=0)", true), ("AVX-512 enabled (default)", false) })
        {
            Console.WriteLine($"#### {label}");
            var psi = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
            if (dll.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)) psi.ArgumentList.Add(dll);
            psi.ArgumentList.Add("--csharp-only");
            if (disable) psi.Environment["DOTNET_EnableAVX512"] = "0";
            using var child = Process.Start(psi)!;
            child.WaitForExit();
            Console.WriteLine();
        }
    }
}
