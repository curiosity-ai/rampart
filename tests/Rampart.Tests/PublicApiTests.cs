using System.Runtime.CompilerServices;
using System.Security.Cryptography;

namespace Rampart.Tests;

/// <summary>The public surface: the Catalyst.Presidio flow of analyzer, anonymizer and cloaker.</summary>
public sealed class AnalyzerTests
{
    private static readonly RampartAnalyzer Analyzer = new();

    [Fact]
    public void ReportsEntitiesWithInclusiveEnds()
    {
        const string text = "My name is Alex Rivera and my SSN is 472-81-0094.";
        var results = Analyzer.Analyze(text);
        Assert.Equal(["GIVEN_NAME", "SURNAME", "SSN"], results.Select(r => r.EntityType));
        Assert.Equal(["Alex", "Rivera", "472-81-0094"], results.Select(r => text.Substring(r.Start, r.Length)));
        Assert.All(results, r => Assert.Equal(r.End - r.Start + 1, r.Length));
        Assert.All(results, r => Assert.InRange(r.Score, 0.4f, 1f));
    }

    [Fact]
    public void KeepsCityStateAndZipByDefault()
    {
        const string text = "Jordan Smith lives at 12 Elm Street, Denver, CO 80202.";
        Assert.DoesNotContain(Analyzer.Analyze(text), r => r.EntityType is "CITY" or "STATE" or "ZIP_CODE");
        var all = new RampartAnalyzer(new RampartOptions { KeepLabels = [] }).Analyze(text);
        Assert.Contains(all, r => r.EntityType == "ZIP_CODE" && text.Substring(r.Start, r.Length) == "80202");
    }

    [Fact]
    public void ResultsAreDisjointAndOrdered()
    {
        var results = Analyzer.Analyze("Call Jordan Smith at (202) 555-0188 or jordan.smith@example.com, card 4111 1111 1111 1111.");
        for (int i = 1; i < results.Count; i++) Assert.True(results[i].Start > results[i - 1].End);
    }
}

public sealed class AnonymizerTests
{
    private const string Text = "My name is Alex Rivera, SSN 472-81-0094, I live in Boston.";
    private static readonly List<RecognizerResult> Results = new RampartAnalyzer().Analyze(Text);
    private static readonly RampartAnonymizer Anonymizer = new();

    [Fact]
    public void Operators()
    {
        Assert.Equal("My name is **** ******, SSN ***********, I live in Boston.", Anonymizer.Anonymize(Text, Results));
        Assert.Equal("My name is #### ######, SSN ###########, I live in Boston.", Anonymizer.Anonymize(Text, Results, "mask", "#"));
        Assert.Equal("My name is <REDACTED> <REDACTED>, SSN <REDACTED>, I live in Boston.", Anonymizer.Anonymize(Text, Results, "replace"));
        Assert.Equal("My name is [PII] [PII], SSN [PII], I live in Boston.", Anonymizer.Anonymize(Text, Results, "replace", replacement: "[PII]"));
        Assert.Equal("My name is  , SSN , I live in Boston.", Anonymizer.Anonymize(Text, Results, "redact"));
        Assert.Contains(Convert.ToHexStringLower(SHA256.HashData("472-81-0094"u8)), Anonymizer.Anonymize(Text, Results, "hash"));
        Assert.Equal(Anonymizer.Anonymize(Text, Results), Anonymizer.Anonymize(Text, Results, "unknown"));
    }

    [Fact]
    public void OverlapsKeepTheEarliestAndInvalidResultsAreSkipped()
    {
        var results = new List<RecognizerResult>
        {
            new(11, 14, "GIVEN_NAME", 0.9f),
            new(13, 21, "SURNAME", 0.95f),   // overlaps the first: dropped
            new(50, 90, "CITY", 1f),         // past the end: skipped
        };
        Assert.Equal("My name is **** Rivera, SSN 472-81-0094, I live in Boston.", Anonymizer.Anonymize(Text, results));
        Assert.Equal(Text, Anonymizer.Anonymize(Text, []));
    }
}

public sealed class CloakerTests
{
    private static readonly RampartAnalyzer Analyzer = new();

    [Fact]
    public async Task CloaksAndRestores()
    {
        const string text = "My name is Alex Rivera and my SSN is 472-81-0094.";
        var cloaker = new RampartCloaker();
        var answer = await cloaker.CloakAsync(text, Analyzer.Analyze(text), cloaked =>
        {
            Assert.Equal("My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1].", cloaked);
            return Task.FromResult("Thanks [GIVEN_NAME_1], [SSN_1] is on file.");
        });
        Assert.Equal("Thanks Alex, 472-81-0094 is on file.", answer);
    }

    [Fact]
    public async Task SameValueSamePlaceholderAcrossCalls()
    {
        var cloaker = new RampartCloaker();
        string first = "Please call Jordan Smith at (202) 555-0188.", second = "Jordan Smith moved last year.";
        string? a = null, b = null;
        await cloaker.CloakAsync(first, Analyzer.Analyze(first), c => Task.FromResult(a = c));
        await cloaker.CloakAsync(second, Analyzer.Analyze(second), c => Task.FromResult(b = c));
        Assert.Equal("Please call [GIVEN_NAME_1] [SURNAME_1] at [PHONE_1].", a);
        Assert.StartsWith("[GIVEN_NAME_1] [SURNAME_1]", b);
    }

    [Fact]
    public async Task StreamingRestoresPlaceholdersSplitAcrossChunks()
    {
        const string text = "My name is Alex Rivera.";
        static async IAsyncEnumerable<string> Service(string cloaked, [EnumeratorCancellation] CancellationToken ct = default)
        {
            foreach (var chunk in new[] { "Hello [GIV", "EN_NAME_1] [SUR", "NAME_1]", ", how can I help?" })
            {
                await Task.Yield();
                yield return chunk;
            }
        }
        var parts = new List<string>();
        await foreach (var part in new RampartCloaker().CloakAsync(text, Analyzer.Analyze(text), c => Service(c), TestContext.Current.CancellationToken))
            parts.Add(part);
        Assert.Equal("Hello Alex Rivera, how can I help?", string.Concat(parts));
        Assert.DoesNotContain(parts, p => p.Contains('['));
    }

    [Fact]
    public async Task NoResultsSendsTheTextUnchanged()
    {
        string? sent = null;
        var answer = await new RampartCloaker().CloakAsync("Nothing to hide.", [], c => Task.FromResult(sent = c));
        Assert.Equal("Nothing to hide.", sent);
        Assert.Equal("Nothing to hide.", answer);
    }

    [Fact]
    public async Task RejectsResultsItCannotCloak()
    {
        const string text = "Hello John.";
        await Assert.ThrowsAsync<ArgumentException>(() => new RampartCloaker().CloakAsync(text, [new(6, 9, "PERSON", 1f)], Task.FromResult));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => new RampartCloaker().CloakAsync(text, [new(6, 20, "GIVEN_NAME", 1f)], Task.FromResult));
    }
}
