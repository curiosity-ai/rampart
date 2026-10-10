using System.Security.Cryptography;
using Rampart.Detection;
using Rampart.Model;
using Rampart.Ner;
using Rampart.Session;

namespace Rampart.Tests;

public sealed class ResourceTests
{
    [Theory]
    // SHA-256 of the files at https://huggingface.co/nationaldesignstudio/rampart (main).
    [InlineData("Rampart.model_q4.onnx", "9f27d24949b0581701071ea5ef522d77ccd3f50c525cc91eac4d265b0fc2afe5")]
    [InlineData("Rampart.vocab.txt", "0fbe6b50061feabb9be68af471e9aa6df07a4bc428bdca4b0eff1fcd3612dee5")]
    [InlineData("Rampart.config.json", "003b84bbcd489f5e782fe5cad8f3249c3653ec880089abb1ccc398a0d895e3e6")]
    public void EmbeddedResourcesAreThePublishedFiles(string resource, string sha256)
    {
        var bytes = RampartModel.ReadResource(typeof(RampartModel).Assembly, resource);
        Assert.Equal(sha256, Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }

    [Fact]
    public void ConfigDescribesMiniLm()
    {
        var cfg = RampartModel.Default.Config;
        Assert.Equal((384, 6, 12, 1536, 512, 19730), (cfg.HiddenSize, cfg.NumHiddenLayers, cfg.NumAttentionHeads, cfg.IntermediateSize, cfg.MaxPositionEmbeddings, cfg.VocabSize));
        Assert.Equal(35, cfg.Labels.Count);
        Assert.Equal("O", cfg.Labels[0]);
        Assert.Equal("B-GIVEN_NAME", cfg.Labels[1]);
    }

    [Fact]
    public void ModelIsSharedAndThreadSafe()
    {
        var model = RampartModel.Default;
        var ids = model.Tokenizer.Encode("My name is Alex Rivera and I live at 12 Main Street.");
        var expected = model.ComputeLogits(ids);
        Parallel.For(0, 32, _ => Assert.Equal(expected, model.ComputeLogits(ids)));
    }
}

public sealed class KernelTests
{
    [Fact]
    public void ErfIsAccurateToFloatPrecision()
    {
        double worst = 0;
        for (double x = -6; x <= 6; x += 1e-4)
        {
            worst = Math.Max(worst, Math.Abs(Kernels.Erf((float)x) - ReferenceErf((float)x)));
        }
        Assert.True(worst < 5e-7, $"max |erf error| = {worst}");

        // The vector path is the same polynomial.
        var data = Enumerable.Range(-4000, 8001).Select(i => i * 1e-3f).ToArray();
        var scalar = data.Select(x => x * (Kernels.Erf(x * 0.70710678118654752440f) + 1f) * 0.5f).ToArray();
        Kernels.Gelu(data);
        for (int i = 0; i < data.Length; i++) Assert.Equal(scalar[i], data[i], 6);
    }

    /// <summary>erf in double: Maclaurin series near 0, continued fraction for erfc further out.</summary>
    private static double ReferenceErf(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 2.5)
        {
            double sum = 0, term = ax;
            for (int n = 0; n < 200; n++)
            {
                double t = term / (2 * n + 1);
                sum += t;
                if (Math.Abs(t) < 1e-18) break;
                term *= -ax * ax / (n + 1);
            }
            return Math.Sign(x) * 2 / Math.Sqrt(Math.PI) * sum;
        }
        // erfc(x) = exp(-x²)/√π · 1/(x + 1/2/(x + 1/(x + 3/2/(x + …))))
        double f = ax;
        for (int k = 60; k >= 1; k--) f = ax + k / 2.0 / f;
        double erfc = Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / f;
        return Math.Sign(x) * (1 - erfc);
    }

    [Fact]
    public void Float32SoftmaxMatchesTypedArraySemantics()
    {
        float[] row = [2f, 1f, 0.5f, -3f];
        double sum = row.Sum(x => (double)(float)Math.Exp(x - 2.0));
        Assert.Equal((double)(float)(1.0 / sum), TokenClassifier.Float32Softmax(row, 0));
    }
}

public sealed class ValidatorTests
{
    [Theory]
    [InlineData("4111111111111111", true)]
    [InlineData("378282246310005", true)]
    [InlineData("4111111111111112", false)]
    public void Luhn(string digits, bool valid) => Assert.Equal(valid, Validators.IsLuhnValid(digits));

    [Theory]
    [InlineData("472810094", true)]
    [InlineData("000123456", false)]
    [InlineData("666123456", false)]
    [InlineData("912345678", false)]
    [InlineData("472001234", false)]
    [InlineData("472810000", false)]
    [InlineData("47281009", false)]
    public void Ssn(string digits, bool valid) => Assert.Equal(valid, Validators.IsValidSsn(digits));

    [Fact]
    public void DigitRunsMustMatchTheWholeEntity()
    {
        Assert.Single(Heuristics.Detect("SSN 472-81-0094."), s => s.Label == PiiLabel.Ssn && s.Text == "472-81-0094");
        Assert.Empty(Heuristics.Detect("account 1472-81-00945")); // 10 digits: not an SSN
        Assert.Single(Heuristics.Detect("card 4111 1111 1111 1111 ok"), s => s.Label == PiiLabel.CreditCard && s.Text == "4111 1111 1111 1111");
        Assert.Empty(Heuristics.Detect("income $45,000 in 2026, age 34"));
    }

    [Fact]
    public void BoundariesAreAscii()
    {
        // JavaScript \b is ASCII-only: an accented letter is not a word character.
        Assert.Single(Heuristics.Detect("é192.168.0.1"), s => s.Label == PiiLabel.IpAddress && s.Text == "192.168.0.1");
        Assert.Empty(Heuristics.Detect("x192.168.0.1"));
        Assert.Empty(Heuristics.Detect("at 12:34 and 10:15"));
    }
}

public sealed class PolicyTests
{
    private static PiiSpan S(int start, int end, PiiLabel label, double score, SpanSource source = SpanSource.Ner, string text = "") => new(start, end, label, score, source, text);

    [Fact]
    public void HigherScoreWinsAndContainmentCollapses()
    {
        var merged = Policy.MergeSpans([S(0, 10, PiiLabel.Phone, 0.6), S(2, 5, PiiLabel.Ssn, 1, SpanSource.Heuristic)]);
        Assert.Equal([S(2, 5, PiiLabel.Ssn, 1, SpanSource.Heuristic)], merged);
    }

    [Fact]
    public void PartialOverlapTakesTheUnionUnderTheWinner()
    {
        var merged = Policy.MergeSpans([S(0, 6, PiiLabel.GivenName, 0.9, text: "winner"), S(4, 10, PiiLabel.Surname, 0.5)]);
        Assert.Equal([S(0, 10, PiiLabel.GivenName, 0.9, text: "winner")], merged);
    }

    [Fact]
    public void TiesPreferLongerThenHeuristic()
    {
        Assert.Equal(PiiLabel.Url, Policy.MergeSpans([S(0, 4, PiiLabel.Email, 1), S(0, 8, PiiLabel.Url, 1)]).Single().Label);
        Assert.Equal(SpanSource.Heuristic, Policy.MergeSpans([S(0, 4, PiiLabel.Email, 1), S(0, 4, PiiLabel.Url, 1, SpanSource.Heuristic)]).Single().Source);
    }

    [Fact]
    public void PolicyDropsKeptLabelsAndSortsRightToLeft()
    {
        var spans = Policy.ApplyPolicy([S(0, 2, PiiLabel.GivenName, 1), S(5, 8, PiiLabel.City, 1), S(10, 12, PiiLabel.Phone, 1)]);
        Assert.Equal([10, 0], spans.Select(s => s.Start));
        Assert.Equal(3, Policy.ApplyPolicy([S(0, 2, PiiLabel.GivenName, 1), S(5, 8, PiiLabel.City, 1), S(10, 12, PiiLabel.Phone, 1)], new HashSet<PiiLabel>()).Count);
    }

    [Fact]
    public void PremaskMapsSentinelsBackToTheirSpan()
    {
        const string raw = "SSN 472-81-0094 ok";
        var map = Premask.Apply(raw, Heuristics.Detect(raw));
        Assert.Equal("SSN [SSN] ok", map.Masked);
        var projected = Premask.Project(S(5, 7, PiiLabel.Phone, 0.5), raw, map);
        Assert.Equal((4, 15, "472-81-0094"), (projected!.Start, projected.End, projected.Text));
        Assert.Null(Premask.Project(S(3, 3, PiiLabel.Phone, 0.5), raw, map));
    }
}

public sealed class SessionTests
{
    [Fact]
    public void PlaceholdersAreStableAndCaseInsensitive()
    {
        var table = new SessionEntityTable();
        var a = table.PlaceholderFor(PiiLabel.GivenName, "John");
        Assert.Equal("[GIVEN_NAME_1]", a);
        Assert.Equal(a, table.PlaceholderFor(PiiLabel.GivenName, "  JOHN "));
        Assert.Equal("[GIVEN_NAME_2]", table.PlaceholderFor(PiiLabel.GivenName, "Jane"));
        Assert.Equal("[SURNAME_1]", table.PlaceholderFor(PiiLabel.Surname, "John"));
        Assert.Equal("John and Jane, [PHONE_1]", table.Rehydrate("[GIVEN_NAME_1] and [GIVEN_NAME_2], [PHONE_1]"));
    }

    [Fact]
    public void AliasesRenameTokens()
    {
        var table = new SessionEntityTable(new Dictionary<PiiLabel, string> { [PiiLabel.GivenName] = "NAME", [PiiLabel.Surname] = "NAME" });
        Assert.Equal("[NAME_1]", table.PlaceholderFor(PiiLabel.GivenName, "Ana"));
        Assert.Equal("[NAME_2]", table.PlaceholderFor(PiiLabel.Surname, "Silva"));
    }

    [Fact]
    public void ScrubKeepsGeographyByDefault()
    {
        var table = new SessionEntityTable();
        var result = table.Scrub("Ana in Lisbon", [new(0, 3, PiiLabel.GivenName, 0.9, SpanSource.Ner, "Ana"), new(7, 13, PiiLabel.City, 0.9, SpanSource.Ner, "Lisbon")]);
        Assert.Equal("[GIVEN_NAME_1] in Lisbon", result.Text);
        Assert.Equal(["[GIVEN_NAME_1]"], result.Placeholders);
    }

    [Theory]
    [InlineData("Thanks [GIVEN_|NAME_1], your SSN [SS|N_1] is on file.")]
    [InlineData("Thanks |[|GIVEN_NAME_1|]|, your SSN |[SSN_1] is on file.")]
    [InlineData("Thanks [GIVEN_NAME_1], your SSN [SSN_1] is on file.")]
    public void StreamingRevealHandlesSplitPlaceholders(string joined)
    {
        var chunks = joined.Split('|');
        var table = new SessionEntityTable();
        table.PlaceholderFor(PiiLabel.GivenName, "Alex");
        table.PlaceholderFor(PiiLabel.Ssn, "472-81-0094");
        var reveal = new StreamingReveal(table.Resolve);
        var output = string.Concat(chunks.Select(reveal.Push)) + reveal.Flush();
        Assert.Equal("Thanks Alex, your SSN 472-81-0094 is on file.", output);
    }

    [Fact]
    public void StreamingRevealFlushesAnUnfinishedTail()
    {
        var reveal = new StreamingReveal(_ => "x");
        Assert.Equal("a ", reveal.Push("a [B"));
        Assert.Equal("[B", reveal.Flush());
    }

    [Fact]
    public async Task GuardRevealsAnAsyncStream()
    {
        var guard = new ChatGuard(new RampartOptions { HeuristicsOnly = true });
        var safe = guard.Protect("My SSN is 472-81-0094.");
        Assert.Equal("My SSN is [SSN_1].", safe.Text);

        static async IAsyncEnumerable<string> Stream()
        {
            foreach (var c in new[] { "Your SSN ", "[SS", "N_1", "] is noted." })
            {
                await Task.Yield();
                yield return c;
            }
        }
        var parts = new List<string>();
        await foreach (var part in guard.RevealAsync(Stream(), TestContext.Current.CancellationToken)) parts.Add(part);
        Assert.Equal("Your SSN 472-81-0094 is noted.", string.Concat(parts));
    }
}

public sealed class GemmTests
{
    public static TheoryData<bool, bool> Paths() => new()
    {
        { true, true }, { false, true }, { false, false },
    };

    [Theory]
    [MemberData(nameof(Paths))]
    public void MatchesNaiveProductOnRaggedShapes(bool v512, bool v256)
    {
        if (v512 && !System.Runtime.Intrinsics.X86.Avx512F.IsSupported) return;
        if (v256 && !System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated) return;
        var random = new Random(42);
        foreach (var (m, n, k) in new[] { (1, 1, 1), (7, 33, 5), (13, 70, 64), (6, 32, 384), (31, 47, 129), (4, 16, 3) })
        {
            int lda = k + 3, ldb = n + 5, ldc = n + 2;
            var a = Enumerable.Range(0, m * lda).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
            var b = Enumerable.Range(0, k * ldb).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
            var bias = Enumerable.Range(0, n).Select(_ => (float)random.NextDouble()).ToArray();
            var c = Enumerable.Repeat(float.NaN, m * ldc).ToArray();
            Gemm.Multiply(a, lda, b, ldb, c, ldc, m, n, k, bias, v512, v256);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double expected = bias[j];
                    for (int p = 0; p < k; p++) expected += (double)a[i * lda + p] * b[p * ldb + j];
                    Assert.True(Math.Abs(expected - c[i * ldc + j]) < 1e-4, $"({m},{n},{k}) c[{i},{j}] = {c[i * ldc + j]} vs {expected}");
                }
                for (int j = n; j < ldc; j++) Assert.True(float.IsNaN(c[i * ldc + j]), "wrote outside the block");
            }
        }
    }

    [Fact]
    public void PanelOrderedWeightsDecodeToTheShippedNibbles()
    {
        // Decode layer 0's query weight straight from the ONNX tensors and compare every weight with the
        // permuted copy, both through the scalar accessor and the SIMD tile dequantizer.
        var onnx = Rampart.Onnx.OnnxModel.Parse(RampartModel.ReadResource(typeof(RampartModel).Assembly, "Rampart.model_q4.onnx"));
        var mm = onnx.Nodes.Single(n => n.Outputs[0] == OnnxNames.MatMul(0, "attention/self/query"));
        var packed = onnx.GetTensor(mm.Inputs[1]).AsBytes();
        var scales = onnx.GetTensor(mm.Inputs[2]).AsFloats();
        var linear = RampartModel.Default.Weights.Layers[0].Query;
        var panel = new float[linear.K * Q4Linear.Tile];
        for (int tile = 0; tile < linear.Tiles; tile++)
        {
            linear.DequantizeTile(tile, panel);
            for (int r = 0; r < Q4Linear.Tile; r++)
            {
                int row = tile * Q4Linear.Tile + r;
                for (int k = 0; k < linear.K; k++)
                {
                    byte b = packed[row * (linear.K / 2) + k / 2];
                    int q = k % 2 == 0 ? b & 0x0F : b >> 4;
                    float expected = (q - 8f) * scales[row * linear.BlocksPerRow + k / linear.BlockSize];
                    Assert.Equal(expected, linear.Weight(row, k));
                    Assert.Equal(expected, panel[k * Q4Linear.Tile + r]);
                }
            }
        }
    }
}

public sealed class ExpTests
{
    [Fact]
    public void FastExpIsWithinTwoUlpOverTheSoftmaxRange()
    {
        double worst = 0;
        var xs = new float[16];
        for (double x0 = -87; x0 <= 88; x0 += 16 * 1e-3)
        {
            for (int i = 0; i < 16; i++) xs[i] = (float)(x0 + i * 1e-3);
            var v512 = Kernels.Exp(System.Runtime.Intrinsics.Vector512.Create(xs));
            var v256 = Kernels.Exp(System.Runtime.Intrinsics.Vector256.Create(xs.AsSpan(0, 8)));
            for (int i = 0; i < 16; i++)
            {
                double exact = Math.Exp(xs[i]);
                worst = Math.Max(worst, Math.Abs(v512[i] - exact) / exact);
                worst = Math.Max(worst, Math.Abs(Kernels.Exp(xs[i]) - exact) / exact);
                if (i < 8) worst = Math.Max(worst, Math.Abs(v256[i] - exact) / exact);
            }
        }
        Assert.True(worst < 2.5e-7, $"max relative error {worst}");
    }

    [Fact]
    public void SoftmaxSumsToOne()
    {
        var row = Enumerable.Range(0, 37).Select(i => (float)Math.Sin(i) * 20).ToArray();
        Kernels.SoftmaxInPlace(row);
        Assert.Equal(1.0, row.Sum(x => (double)x), 5);
        Assert.All(row, p => Assert.InRange(p, 0f, 1f));
    }
}

public sealed class HeuristicsScannerTests
{
    /// <summary>The hand-written scanners against the regular expressions they replace, on random text.</summary>
    [Fact]
    public void ScannersMatchTheirExpressions()
    {
        var random = new Random(7);
        const string alphabet = "0123456789012345.....---   a_1225";
        for (int trial = 0; trial < 20000; trial++)
        {
            var chars = new char[random.Next(0, 40)];
            for (int i = 0; i < chars.Length; i++) chars[i] = alphabet[random.Next(alphabet.Length)];
            var text = new string(chars);

            var expected = Heuristics.DigitRun().Matches(text).Select(m => (m.Index, m.Length)).ToList();
            var actual = new List<(int, int)>();
            for (int at = 0; (at = Heuristics.NextDigitRun(text, at, out int length)) >= 0; at += length) actual.Add((at, length));
            Assert.True(expected.SequenceEqual(actual), $"digit runs differ on \"{text}\"");

            expected = Heuristics.Ipv4Rule().Matches(text).Select(m => (m.Index, m.Length)).ToList();
            actual.Clear();
            for (int at = 0; (at = Heuristics.NextIpv4(text, at, out int length)) >= 0; at += length) actual.Add((at, length));
            Assert.True(expected.SequenceEqual(actual), $"IPv4 differs on \"{text}\"");
        }
    }
}

/// <summary>
/// Runs <see cref="AllocationTests"/> on its own. <c>ArrayPool&lt;T&gt;.Shared</c> allocates a fresh array
/// when concurrent renters have drained its per-core stocks, so measured next to the parity tests a
/// pass that allocates nothing on its own can read a few kilobytes.
/// </summary>
[CollectionDefinition(nameof(AllocationTests), DisableParallelization = true)]
public sealed class AllocationCollection;

/// <summary>The hot paths allocate nothing once warm (see CLAUDE.md, "Performance and allocation notes").</summary>
[Collection(nameof(AllocationTests))]
public sealed class AllocationTests
{
    private static readonly RampartModel SingleThreaded = RampartModel.Load(new RampartModelOptions { MaxDegreeOfParallelism = 1 });

    private static long Measure(Action action)
    {
        for (int i = 0; i < 20; i++) action();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) action();
        return (GC.GetAllocatedBytesForCurrentThread() - before) / 20;
    }

    [Fact]
    public void ComputeLogitsIntoASpanAllocatesNothing()
    {
        var model = SingleThreaded;
        var ids = model.Tokenizer.Encode("My name is Alex Rivera and I live at 12 Main Street, Springfield.");
        int[] longIds = [model.Tokenizer.ClsId, .. model.Tokenizer.Encode(Corpus.Long(), addSpecialTokens: false).AsSpan(0, 300), model.Tokenizer.SepId];
        var logits = new float[512 * model.Labels.Count];
        Assert.Equal(0, Measure(() => model.ComputeLogits(ids, logits)));
        Assert.Equal(0, Measure(() => model.ComputeLogits(longIds, logits)));
    }

    [Fact]
    public void CountingTokensAllocatesNothing()
    {
        var tokenizer = SingleThreaded.Tokenizer;
        Assert.Equal(0, Measure(() => tokenizer.CountTokens("Hola, me llamo José Ángel García López; call (202) 555-0188 [SEP] ok.")));
    }

    [Fact]
    public void DetectionAllocatesOnlyItsResults()
    {
        var analyzer = new RampartAnalyzer(new RampartOptions { Model = SingleThreaded });
        const string text = "My name is Alex Rivera and I live at 12 Main Street, Springfield. Call me at (202) 555-0188.";
        int spans = analyzer.Detect(text).Count;
        // A list, and per span a record plus its text: a generous per-span bound, far below the
        // tens of kilobytes intermediate buffers would cost.
        Assert.InRange(Measure(() => analyzer.Detect(text)), 1, 128 + spans * 160);
    }
}
