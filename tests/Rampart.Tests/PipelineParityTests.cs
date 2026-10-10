using Rampart.Detection;
using Rampart.Model;
using Rampart.Ner;

namespace Rampart.Tests;

/// <summary>
/// Every stage of the C# pipeline against the original JavaScript pipeline, case by case: tokens,
/// per-token classifier output, heuristics, premask, NER spans, merged detections and the final
/// placeholdered text.
/// </summary>
public sealed class PipelineParityTests
{
    private static readonly RampartModel Model = RampartModel.Default;
    private static readonly NerDetector Ner = new(Model);
    private static readonly RampartAnalyzer Analyzer = new();

    public static TheoryData<string> Cases() => ReferenceFixture.CaseIds();

    /// <summary>
    /// Cases where the port deliberately does not reproduce the reference. The reference's
    /// <c>foldForModel</c> records one offset per code point while its folded string grows by two
    /// UTF-16 units for an astral character (an emoji), so every NER offset after the first emoji is
    /// shifted: it leaks the first letter of "Ana" and emits a span whose end is <c>undefined</c>.
    /// The port keeps the map aligned per UTF-16 unit; <see cref="AstralCharactersDoNotShiftOffsets"/>
    /// pins the corrected behaviour.
    /// </summary>
    private static readonly HashSet<string> KnownDivergences = ["x-unicode"];

    [Theory]
    [MemberData(nameof(Cases))]
    public void Tokenization(string id)
    {
        var c = ReferenceFixture.Case(id);
        var text = c.GetProperty("text").GetString()!;
        var expected = c.GetProperty("tokens").EnumerateArray().Select(t => t.GetString()!).ToList();
        Assert.Equal(expected, Model.Tokenizer.Tokenize(text));
        Assert.Equal(c.GetProperty("tokenCount").GetInt32(), Model.Tokenizer.CountTokens(text));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void TokenClassification(string id)
    {
        var c = ReferenceFixture.Case(id);
        var entities = c.GetProperty("entities");
        if (entities.ValueKind == System.Text.Json.JsonValueKind.Null) return; // longer than one window
        var inferText = c.GetProperty("text").GetString()!.Replace('-', ' ');
        var actual = new TokenClassifier(Model).Classify(inferText);
        var expected = entities.EnumerateArray().ToList();
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i].GetProperty("entity").GetString(), actual[i].Entity);
            Assert.Equal(expected[i].GetProperty("index").GetInt32(), actual[i].Index);
            Assert.Equal(expected[i].GetProperty("word").GetString(), actual[i].Word);
            Assert.True(Math.Abs(expected[i].GetProperty("score").GetDouble() - actual[i].Score) < 1e-4, $"token {i}: {actual[i].Score} vs {expected[i].GetProperty("score").GetDouble()}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void HeuristicsAndPremask(string id)
    {
        var c = ReferenceFixture.Case(id);
        var text = c.GetProperty("text").GetString()!;
        var heuristic = Heuristics.Detect(text);
        ReferenceFixture.AssertSpansEqual(ReferenceFixture.Spans(c.GetProperty("heuristic")), heuristic, id);
        Assert.Equal(c.GetProperty("masked").GetString(), Premask.Apply(text, heuristic).Masked);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void NerOnMaskedText(string id)
    {
        if (KnownDivergences.Contains(id)) return;
        var c = ReferenceFixture.Case(id);
        var masked = c.GetProperty("masked").GetString()!;
        ReferenceFixture.AssertSpansEqual(ReferenceFixture.Spans(c.GetProperty("nerOnMasked")), Ner.Detect(masked), id);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DetectionAndMerge(string id)
    {
        if (KnownDivergences.Contains(id)) return;
        var c = ReferenceFixture.Case(id);
        var text = c.GetProperty("text").GetString()!;
        var detected = Analyzer.Detect(text);
        ReferenceFixture.AssertSpansEqual(ReferenceFixture.Spans(c.GetProperty("detected")), detected, id);
        ReferenceFixture.AssertSpansEqual(ReferenceFixture.Spans(c.GetProperty("merged")), Policy.MergeSpans(detected), id);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Protect(string id)
    {
        if (KnownDivergences.Contains(id)) return;
        var c = ReferenceFixture.Case(id);
        var expected = c.GetProperty("protected");
        var result = new ChatGuard(Analyzer).Protect(c.GetProperty("text").GetString()!);
        Assert.Equal(expected.GetProperty("text").GetString(), result.Text);
        Assert.Equal(expected.GetProperty("placeholders").EnumerateArray().Select(p => p.GetString()!), result.Placeholders);
    }

    [Fact]
    public void MultiTurnConversation()
    {
        foreach (var conv in ReferenceFixture.Root.GetProperty("conversations").EnumerateArray())
        {
            var guard = new ChatGuard(Analyzer);
            foreach (var turn in conv.GetProperty("turns").EnumerateArray())
            {
                var result = guard.Protect(turn.GetProperty("user").GetString()!);
                var expected = turn.GetProperty("protected");
                Assert.Equal(expected.GetProperty("text").GetString(), result.Text);
                Assert.Equal(expected.GetProperty("placeholders").EnumerateArray().Select(p => p.GetString()!), result.Placeholders);
                Assert.Equal(turn.GetProperty("revealed").GetString(), guard.Reveal(turn.GetProperty("reply").GetString()!));
            }
        }
    }

    [Fact]
    public void AstralCharactersDoNotShiftOffsets()
    {
        var text = ReferenceFixture.Case("x-unicode").GetProperty("text").GetString()!;
        var result = new ChatGuard(Analyzer).Protect(text);
        Assert.True(result.Text.Contains("emoji \U0001F642 [GIVEN_NAME_2] \U0001F642 [SURNAME_1],"), result.Text);
        Assert.DoesNotContain("Ana", result.Text);
        Assert.DoesNotContain("Silva", result.Text);
        Assert.All(Analyzer.Detect(text), s => Assert.Equal(text[s.Start..s.End], s.Text));
    }
}
