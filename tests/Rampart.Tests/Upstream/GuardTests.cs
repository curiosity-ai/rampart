using Catalyst.Rampart.Session;

namespace Catalyst.Rampart.Tests.Upstream;

/// <summary>Port of <c>__tests__/guard.test.ts</c>.</summary>
public sealed class GuardTests
{
    private static ChatGuard HeuristicsOnly() => new(new RampartOptions { HeuristicsOnly = true });

    [Fact]
    public void MintsPlaceholdersAndHidesTheRawValue()
    {
        var text = HeuristicsOnly().Protect("my ssn is 888-12-3456").Text;
        Assert.Contains("[SSN_1]", text);
        Assert.DoesNotContain("888-12-3456", text);
    }

    [Fact]
    public void RoundTripsAPlaceholder()
    {
        var guard = HeuristicsOnly();
        guard.Protect("my ssn is 472-81-0094");
        Assert.Equal("Your record 472-81-0094 is updated.", guard.Reveal("Your record [SSN_1] is updated."));
    }

    [Fact]
    public void AliasesGivenNameToName()
    {
        var table = new SessionEntityTable(new Dictionary<PiiLabel, string> { [PiiLabel.GivenName] = "NAME" });
        Assert.Equal("[NAME_1]", table.PlaceholderFor(PiiLabel.GivenName, "Alex"));
        Assert.Equal("[NAME_1]", table.PlaceholderFor(PiiLabel.GivenName, "alex"));
        Assert.Equal("[NAME_2]", table.PlaceholderFor(PiiLabel.GivenName, "Maya"));
        Assert.Equal("Maya called", table.Rehydrate("[NAME_2] called"));
    }

    private static Func<string, string?> ResolverFor(Dictionary<string, string> map) => token => map.GetValueOrDefault(token);

    [Fact]
    public void RevealsAPlaceholderSplitAcrossTwoChunks()
    {
        var reveal = new StreamingReveal(ResolverFor(new() { ["[NAME_1]"] = "Alex" }));
        var output = reveal.Push("thanks [NA") + reveal.Push("ME_1] for waiting") + reveal.Flush();
        Assert.Equal("thanks Alex for waiting", output);
    }

    [Fact]
    public void RevealsAPlaceholderSplitCharacterByCharacter()
    {
        var reveal = new StreamingReveal(ResolverFor(new() { ["[SSN_1]"] = "XXX" }));
        var output = string.Concat("ref [SSN_1] end".Select(c => reveal.Push(c.ToString()))) + reveal.Flush();
        Assert.Equal("ref XXX end", output);
    }

    [Fact]
    public void PassesThroughALoneBracket()
    {
        var reveal = new StreamingReveal(ResolverFor([]));
        Assert.Equal("array[0] is fine", reveal.Push("array[0] is fine") + reveal.Flush());
    }

    [Fact]
    public void LeavesUnknownPlaceholdersIntact()
    {
        var reveal = new StreamingReveal(ResolverFor([]));
        Assert.Equal("[NAME_9] unknown", reveal.Push("[NAME_9] unknown") + reveal.Flush());
    }

    [Fact]
    public async Task RevealsAcrossAChunkedStream()
    {
        var table = new SessionEntityTable(new Dictionary<PiiLabel, string> { [PiiLabel.GivenName] = "NAME" });
        table.PlaceholderFor(PiiLabel.GivenName, "Maya");

        static async IAsyncEnumerable<string> Source()
        {
            yield return "hi [NA";
            await Task.Yield();
            yield return "ME_1]!";
        }

        var output = "";
        await foreach (var chunk in StreamingReveal.RevealAsync(Source(), t => table.Knows(t) ? table.Rehydrate(t) : null, TestContext.Current.CancellationToken)) output += chunk;
        Assert.Equal("hi Maya!", output);
    }
}
