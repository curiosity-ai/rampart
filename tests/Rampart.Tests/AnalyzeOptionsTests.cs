using Rampart.Detection;

namespace Rampart.Tests;

/// <summary>
/// <see cref="AnalyzeOptions"/>: the capitalized-particle switch and hints. These depart from the
/// reference on purpose, so the default options must leave every result exactly as the reference has it.
/// </summary>
public sealed class AnalyzeOptionsTests
{
    private static readonly RampartAnalyzer Analyzer = new();

    private static string[] Found(string text, AnalyzeOptions? options = null)
        => [.. Analyzer.Analyze(text, options).Select(r => $"{r.EntityType}={text.Substring(r.Start, r.Length)}")];

    private static AnalyzeOptions Hints(bool particles, params (string Type, string[] Values)[] hints)
        => new() { ExpandCapitalizedParticles = particles, Hints = hints.ToDictionary(h => h.Type, h => h.Values) };

    public static TheoryData<string> Cases() => ReferenceFixture.CaseIds();

    [Theory]
    [MemberData(nameof(Cases))]
    public void DefaultOptionsAreTheReference(string id)
    {
        var text = ReferenceFixture.Case(id).GetProperty("text").GetString()!;
        Assert.Equal(Analyzer.Analyze(text), Analyzer.Analyze(text, new AnalyzeOptions()));
    }

    [Fact]
    public void ParticleRescueCanBeTurnedOff()
    {
        var off = new AnalyzeOptions { ExpandCapitalizedParticles = false };
        Assert.Equal(["GIVEN_NAME=I'm Alex", "SURNAME=Rivera"], Found("Hi, I'm Alex Rivera."));
        Assert.Equal(["GIVEN_NAME=Alex", "SURNAME=Rivera"], Found("Hi, I'm Alex Rivera.", off));
        Assert.Equal(["GIVEN_NAME=Maria", "SURNAME=Lopez", "PHONE=(202) 555-0188"], Found("Call Maria Lopez at (202) 555-0188.", off));
    }

    [Fact]
    public void WithoutTheRescueShortNamePartsCanLeakAndHintWordsCoverThem()
    {
        const string text = "Juan De La Vega Ruiz signed.";
        Assert.Equal(["GIVEN_NAME=Juan De La", "SURNAME=Vega Ruiz"], Found(text));
        // The rescue is what covered "De La Vega": off, they are left in the text.
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=Ruiz"], Found(text, new AnalyzeOptions { ExpandCapitalizedParticles = false }));
        // The hint's words extend the surname leftwards over them, rather than the given name rightwards.
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=De La Vega Ruiz"], Found(text, Hints(false, ("GIVEN_NAME", ["Juan De La Vega Ruiz"]))));
        // The words expand a span even where the whole hint does not occur (another member of the family).
        Assert.Equal(["GIVEN_NAME=Maria", "SURNAME=De La Vega Ruiz"], Found("Maria De La Vega Ruiz signed.", Hints(false, ("SURNAME", ["De La Vega Ruiz"]))));
        // Only the hinted words: "De La" stays out with just "Vega" hinted.
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=Vega Ruiz"], Found(text, Hints(false, ("SURNAME", ["Vega"]))));
    }

    [Fact]
    public void HintedValuesAreAlwaysReported()
    {
        const string text = "Ask alex rivera about it, he will know.";
        Assert.Empty(Found(text));
        var sender = Hints(true, ("GIVEN_NAME", ["Alex"]), ("SURNAME", ["Rivera"]));
        Assert.Equal(["GIVEN_NAME=alex", "SURNAME=rivera"], Found(text, sender));
        Assert.All(Analyzer.Analyze(text, sender), r => Assert.Equal(1f, r.Score));
        var caseSensitive = new AnalyzeOptions { Hints = sender.Hints, HintsIgnoreCase = false };
        Assert.Empty(Found(text, caseSensitive));
    }

    [Fact]
    public void HintsOverrideTheKeepSet()
    {
        const string text = "I moved to Springfield last year.";
        Assert.DoesNotContain(Found(text), f => f.StartsWith("CITY"));
        Assert.Equal(["CITY=Springfield"], Found(text, Hints(true, ("CITY", ["Springfield"]))));
    }

    [Fact]
    public void HintsMatchWholeWordsOnly()
    {
        var options = Hints(true, ("GIVEN_NAME", ["Al"]));
        Assert.Empty(Found("Also, the album is out.", options));
        Assert.Equal(["GIVEN_NAME=Al"], Found("Thanks Al, see you.", options).Where(f => f.EndsWith("=Al")));
    }

    [Fact]
    public void HintsFlowIntoTheCloaker()
    {
        const string text = "Ask alex rivera about it, he will know.";
        var results = Analyzer.Analyze(text, Hints(true, ("GIVEN_NAME", ["Alex"]), ("SURNAME", ["Rivera"])));
        string? sent = null;
        var answer = new RampartCloaker().CloakAsync(text, results, c => Task.FromResult(sent = c)).Result;
        Assert.Equal("Ask [GIVEN_NAME_1] [SURNAME_1] about it, he will know.", sent);
        Assert.Equal(text, answer);
    }

    [Fact]
    public void UnknownEntityTypeThrows()
        => Assert.Throws<ArgumentException>(() => Analyzer.Analyze("Hello.", Hints(true, ("PERSON", ["John"]))));
}

public sealed class HintSetTests
{
    private static (int, int, PiiLabel)[] Find(string text, bool ignoreCase, params (string Type, string[] Values)[] hints)
        => [.. HintSet.Build(hints.ToDictionary(h => h.Type, h => h.Values), ignoreCase).FindOccurrences(text)];

    [Fact]
    public void SpacingBetweenWordsIsFlexible()
    {
        const string text = "John  De-La Croix and JOHN DE LA CROIX";
        Assert.Equal([(0, 17, PiiLabel.GivenName), (22, 38, PiiLabel.GivenName)], Find(text, true, ("GIVEN_NAME", ["John De La Croix"])));
        Assert.Equal([(0, 17, PiiLabel.GivenName)], Find(text, false, ("GIVEN_NAME", ["John De La Croix"])));
    }

    [Fact]
    public void AHintsOwnSeparatorsMatchExactly()
    {
        Assert.Equal([(9, 23, PiiLabel.Phone)], Find("Call me: (202) 555-0188.", true, ("PHONE", ["(202) 555-0188"])));
        Assert.Equal([(5, 20, PiiLabel.Email)], Find("Mail a.b@example.com now", true, ("EMAIL", ["a.b@example.com"])));
        Assert.Empty(Find("Mail a.b@example.org now", true, ("EMAIL", ["a.b@example.com"])));
    }

    [Fact]
    public void PossessivesAndApostrophesSplitWords()
    {
        Assert.Equal([(0, 4, PiiLabel.GivenName)], Find("Alex's account", true, ("GIVEN_NAME", ["Alex"])));
        Assert.Equal([(4, 11, PiiLabel.Surname)], Find("Ask O'Brien", true, ("SURNAME", ["O'Brien"])));
    }
}
