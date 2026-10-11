using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Catalyst.Rampart.Model;

/// <summary>
/// Printed before every benchmark: what machine this is, and how fast it is right now. On a shared
/// VM the same code can run 10–30% slower from one hour to the next, so a result is only comparable
/// with another one whose calibration matches — a GEMM or bandwidth figure well below the usual one,
/// or CPU steal above a few percent, means the box is contended and the numbers are not a regression.
/// </summary>
internal static class BenchEnvironment
{
    public static void Print()
    {
        Console.WriteLine("== Environment =================================================================");
        Console.WriteLine($"  CPU            {CpuModel()}  ({Environment.ProcessorCount} logical cores)");
        Console.WriteLine($"  Memory         {Memory()}");
        Console.WriteLine($"  OS / runtime   {RuntimeInformation.OSDescription.Trim()} / .NET {Environment.Version} ({RuntimeInformation.ProcessArchitecture})");
        Console.WriteLine($"  ONNX Runtime   {OrtVersion()}");
        Console.WriteLine($"  SIMD           {Simd()}");
        Console.WriteLine($"  GEMM kernel    {(Gemm.UseVector512 ? "512-bit (12x32)" : Vector256.IsHardwareAccelerated ? "256-bit (6x16)" : "scalar")}");
        Console.WriteLine($"  Load average   {LoadAverage()}");

        var steal0 = CpuTimes();
        var gemm = CalibrateGemm();
        var (read, copy) = CalibrateBandwidth();
        var steal1 = CpuTimes();
        Console.WriteLine("== Calibration (single thread; compare with earlier runs before reading the results) ==");
        Console.WriteLine($"  GEMM 96x32x384, cache-resident   {gemm.Best,7:F1} GFLOPS  best of 15 ({gemm.Median:F1} median)");
        Console.WriteLine($"  Memory read, 256 MB              {read,7:F1} GB/s");
        Console.WriteLine($"  Memory copy, 256 MB              {copy,7:F1} GB/s   (bytes read + written)");
        Console.WriteLine($"  CPU steal during calibration     {StealPercent(steal0, steal1)}");
        Console.WriteLine("================================================================================");
        Console.WriteLine();
    }

    private static string CpuModel()
    {
        try
        {
            var line = File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal));
            if (line is not null) return line[(line.IndexOf(':') + 1)..].Trim();
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
        return RuntimeInformation.ProcessArchitecture.ToString();
    }

    private static string Memory()
    {
        try
        {
            var info = File.ReadLines("/proc/meminfo").Take(3).ToDictionary(l => l[..l.IndexOf(':')], l => long.Parse(l[(l.IndexOf(':') + 1)..].Replace("kB", "").Trim()) / 1024.0 / 1024.0);
            return $"{info["MemTotal"]:F1} GiB total, {info["MemAvailable"]:F1} GiB available";
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or KeyNotFoundException or FormatException)
        {
            return $"{GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1024.0 / 1024 / 1024:F1} GiB available to the GC";
        }
    }

    private static string OrtVersion()
    {
        try
        {
            return Microsoft.ML.OnnxRuntime.OrtEnv.Instance().GetVersionString();
        }
        catch (Exception e) when (e is DllNotFoundException or TypeInitializationException)
        {
            return "not available";
        }
    }

    private static string Simd()
    {
        var flags = new List<string>();
        if (Avx2.IsSupported) flags.Add("AVX2");
        if (Fma.IsSupported) flags.Add("FMA");
        if (Avx512F.IsSupported) flags.Add("AVX-512F");
        if (Avx512BW.IsSupported) flags.Add("AVX-512BW");
        if (AvxVnni.IsSupported) flags.Add("AVX-VNNI");
        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported) flags.Add("NEON");
        flags.Add($"Vector512.IsHardwareAccelerated={Vector512.IsHardwareAccelerated}");
        flags.Add($"Vector<T>={System.Numerics.Vector<byte>.Count * 8}-bit");
        return string.Join(", ", flags);
    }

    private static string LoadAverage()
    {
        try
        {
            return string.Join(" ", File.ReadAllText("/proc/loadavg").Split(' ').Take(3)) + "  (1/5/15 min)";
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return "n/a";
        }
    }

    /// <summary>The aggregate "cpu" line of /proc/stat: (total jiffies, steal jiffies).</summary>
    private static (long Total, long Steal)? CpuTimes()
    {
        try
        {
            var fields = File.ReadLines("/proc/stat").First().Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).Select(long.Parse).ToArray();
            return (fields.Sum(), fields.Length > 7 ? fields[7] : 0);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    private static string StealPercent((long Total, long Steal)? a, (long Total, long Steal)? b)
    {
        if (a is not { } x || b is not { } y || y.Total == x.Total) return "n/a";
        double pct = 100.0 * (y.Steal - x.Steal) / (y.Total - x.Total);
        return $"{pct:F1}%" + (pct > 5 ? "   <-- the hypervisor is taking CPU from this VM: expect noisy numbers" : "");
    }

    /// <summary>
    /// The engine's own micro-kernel on a shape that lives in L1/L2: a pure SIMD throughput figure.
    /// The best sample is what the core can do; the median also catches the scheduler. An unpinned
    /// thread that is moved between cores mid-sample reads ~30% low at times (pinned to any one core it
    /// never does), and that dip does not show in the model's own latencies, so read the best first.
    /// </summary>
    private static (double Best, double Median) CalibrateGemm()
    {
        const int m = 96, n = 32, k = 384;
        var rnd = new Random(1);
        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)rnd.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rnd.NextDouble();
        const int reps = 200;
        var samples = new double[15];
        // Time-based warm-up: at start-up the tiered JIT is still compiling in the background.
        var warm = Stopwatch.StartNew();
        while (warm.ElapsedMilliseconds < 500) Gemm.Multiply(a, k, b, n, c, n, m, n, k);
        for (int s = 0; s < samples.Length; s++)
        {
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < reps; r++) Gemm.Multiply(a, k, b, n, c, n, m, n, k);
            samples[s] = 2.0 * m * n * k * reps / sw.Elapsed.TotalSeconds / 1e9;
        }
        Array.Sort(samples);
        return (samples[^1], samples[samples.Length / 2]);
    }

    /// <summary>Single-thread streaming bandwidth over a buffer far larger than the last-level cache.</summary>
    private static (double Read, double Copy) CalibrateBandwidth()
    {
        const int bytes = 256 << 20;
        var src = new byte[bytes];
        var dst = new byte[bytes];
        Random.Shared.NextBytes(src.AsSpan(0, 4096));
        src.AsSpan().Fill(1);
        dst.AsSpan().Fill(2);

        double bestRead = 0, bestCopy = 0;
        for (int r = 0; r < 5; r++)
        {
            var sw = Stopwatch.StartNew();
            var sum = Vector256<byte>.Zero;
            ref byte s0 = ref MemoryMarshal.GetArrayDataReference(src);
            for (nuint i = 0; i < bytes; i += 128)
            {
                sum += Vector256.LoadUnsafe(ref s0, i) ^ Vector256.LoadUnsafe(ref s0, i + 32)
                     ^ Vector256.LoadUnsafe(ref s0, i + 64) ^ Vector256.LoadUnsafe(ref s0, i + 96);
            }
            double read = bytes / sw.Elapsed.TotalSeconds / 1e9;
            GC.KeepAlive(sum);

            sw.Restart();
            src.AsSpan().CopyTo(dst);
            double copy = 2.0 * bytes / sw.Elapsed.TotalSeconds / 1e9;

            bestRead = Math.Max(bestRead, read);
            bestCopy = Math.Max(bestCopy, copy);
        }
        return (bestRead, bestCopy);
    }
}
