using Catalyst.Rampart.Detection;

namespace Catalyst.Rampart.Tests;

/// <summary>
/// <see cref="AnalyzeOptions"/>: the capitalized-particle switch and hints. These depart from the
/// reference on purpose, so the default options must leave every result exactly as the reference has it.
/// </summary>
public sealed class AnalyzeOptionsTests
{
    private static readonly RampartAnalyzer Analyzer = new();

    private static string[] Found(string text, AnalyzeOptions? options = null)
        => [.. Analyzer.Analyze(text, options).Select(r => $"{r.EntityType}={text.Substring(r.Start, r.Length)}")];

    private static AnalyzeOptions Hints(bool particles, params (PiiLabel Type, string[] Values)[] hints)
    {
        var options = new AnalyzeOptions().WithCapitalizedParticles(particles);
        foreach (var (type, values) in hints) options.AddHint(type, values);
        return options;
    }

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
        var off = new AnalyzeOptions().WithCapitalizedParticles(false);
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
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=Ruiz"], Found(text, new AnalyzeOptions().WithCapitalizedParticles(false)));
        // The hint's words extend the surname leftwards over them, rather than the given name rightwards.
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=De La Vega Ruiz"], Found(text, Hints(false, (PiiLabel.GivenName, ["Juan De La Vega Ruiz"]))));
        // The words expand a span even where the whole hint does not occur (another member of the family).
        Assert.Equal(["GIVEN_NAME=Maria", "SURNAME=De La Vega Ruiz"], Found("Maria De La Vega Ruiz signed.", Hints(false, (PiiLabel.Surname, ["De La Vega Ruiz"]))));
        // Only the hinted words: "De La" stays out with just "Vega" hinted.
        Assert.Equal(["GIVEN_NAME=Juan", "SURNAME=Vega Ruiz"], Found(text, Hints(false, (PiiLabel.Surname, ["Vega"]))));
    }

    [Fact]
    public void HintedValuesAreAlwaysReported()
    {
        const string text = "Ask alex rivera about it, he will know.";
        Assert.Empty(Found(text));
        var sender = Hints(true, (PiiLabel.GivenName, ["Alex"]), (PiiLabel.Surname, ["Rivera"]));
        Assert.Equal(["GIVEN_NAME=alex", "SURNAME=rivera"], Found(text, sender));
        Assert.All(Analyzer.Analyze(text, sender), r => Assert.Equal(1f, r.Score));
        Assert.Empty(Found(text, sender.WithHintsIgnoringCase(false)));
        Assert.Equal(["GIVEN_NAME=alex", "SURNAME=rivera"], Found(text, sender.WithHintsIgnoringCase(true)));
    }

    [Fact]
    public void HintsOverrideTheKeepSet()
    {
        const string text = "I moved to Springfield last year.";
        Assert.DoesNotContain(Found(text), f => f.StartsWith("CITY"));
        Assert.Equal(["CITY=Springfield"], Found(text, Hints(true, (PiiLabel.City, ["Springfield"]))));
    }

    [Fact]
    public void HintsMatchWholeWordsOnly()
    {
        var options = Hints(true, (PiiLabel.GivenName, ["Al"]));
        Assert.Empty(Found("Also, the album is out.", options));
        Assert.Equal(["GIVEN_NAME=Al"], Found("Thanks Al, see you.", options).Where(f => f.EndsWith("=Al")));
    }

    [Fact]
    public void HintsFlowIntoTheCloaker()
    {
        const string text = "Ask alex rivera about it, he will know.";
        var results = Analyzer.Analyze(text, Hints(true, (PiiLabel.GivenName, ["Alex"]), (PiiLabel.Surname, ["Rivera"])));
        string? sent = null;
        var answer = new RampartCloaker().CloakAsync(text, results, c => Task.FromResult(sent = c)).Result;
        Assert.Equal("Ask [GIVEN_NAME_1] [SURNAME_1] about it, he will know.", sent);
        Assert.Equal(text, answer);
    }

    [Fact]
    public void UndefinedLabelThrowsWhenAdded()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new AnalyzeOptions().AddHint((PiiLabel)999, "John"));

    [Fact]
    public void ATypeTakesManyValuesAcrossCalls()
    {
        var options = new AnalyzeOptions()
            .AddHint(PiiLabel.GivenName, "Alex", "Sam")
            .AddHint(PiiLabel.GivenName, "Jordan")
            .AddHint(PiiLabel.GivenName, "Alex", " ", null!)   // a repeat, a blank and a null are ignored
            .AddHint(PiiLabel.Surname, "Rivera");
        Assert.Equal(["Alex", "Sam", "Jordan"], options.Hints[PiiLabel.GivenName]);
        Assert.Equal(["Rivera"], options.Hints[PiiLabel.Surname]);
        Assert.Equal(
            ["GIVEN_NAME=sam", "GIVEN_NAME=alex", "SURNAME=rivera", "GIVEN_NAME=jordan"],
            Found("ask sam and alex rivera, or jordan.", options));
    }

    [Fact]
    public void OverlappingValuesOfATypeAreMergedIntoOneSpan()
    {
        var options = new AnalyzeOptions().AddHint(PiiLabel.Surname, "De La Vega Ruiz", "Vega");
        Assert.Equal(["SURNAME=de la vega ruiz"], Found("send it to de la vega ruiz.", options));
    }

    [Fact]
    public void AddingAHintAfterUseRecompiles()
    {
        var options = new AnalyzeOptions().AddHint(PiiLabel.GivenName, "Alex");
        Assert.Equal(["GIVEN_NAME=alex"], Found("ask alex or sam.", options));
        options.AddHint(PiiLabel.GivenName, "Sam");
        Assert.Equal(["GIVEN_NAME=alex", "GIVEN_NAME=sam"], Found("ask alex or sam.", options));
    }

    [Fact]
    public void NoHintsNoDictionary()
    {
        var options = new AnalyzeOptions().WithCapitalizedParticles(false);
        Assert.False(options.HasHints);
        Assert.Empty(options.Hints);
        Assert.Null(options.CompiledHints);
    }
}

public sealed class HintSetTests
{
    private static (int, int, PiiLabel)[] Find(string text, bool ignoreCase, params (PiiLabel Type, string[] Values)[] hints)
    {
        var options = new AnalyzeOptions().WithHintsIgnoringCase(ignoreCase);
        foreach (var (type, values) in hints) options.AddHint(type, values);
        return [.. options.CompiledHints!.FindOccurrences(text)];
    }

    [Fact]
    public void SpacingBetweenWordsIsFlexible()
    {
        const string text = "John  De-La Croix and JOHN DE LA CROIX";
        Assert.Equal([(0, 17, PiiLabel.GivenName), (22, 38, PiiLabel.GivenName)], Find(text, true, (PiiLabel.GivenName, ["John De La Croix"])));
        Assert.Equal([(0, 17, PiiLabel.GivenName)], Find(text, false, (PiiLabel.GivenName, ["John De La Croix"])));
    }

    [Fact]
    public void AHintsOwnSeparatorsMatchExactly()
    {
        Assert.Equal([(9, 23, PiiLabel.Phone)], Find("Call me: (202) 555-0188.", true, (PiiLabel.Phone, ["(202) 555-0188"])));
        Assert.Equal([(5, 20, PiiLabel.Email)], Find("Mail a.b@example.com now", true, (PiiLabel.Email, ["a.b@example.com"])));
        Assert.Empty(Find("Mail a.b@example.org now", true, (PiiLabel.Email, ["a.b@example.com"])));
    }

    [Fact]
    public void PossessivesAndApostrophesSplitWords()
    {
        Assert.Equal([(0, 4, PiiLabel.GivenName)], Find("Alex's account", true, (PiiLabel.GivenName, ["Alex"])));
        Assert.Equal([(4, 11, PiiLabel.Surname)], Find("Ask O'Brien", true, (PiiLabel.Surname, ["O'Brien"])));
    }
}
