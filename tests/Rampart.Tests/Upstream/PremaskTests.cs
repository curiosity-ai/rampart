using Rampart.Detection;
using Rampart.Ner;

namespace Rampart.Tests.Upstream;

/// <summary>Port of <c>__tests__/premask.test.ts</c>.</summary>
public sealed class PremaskTests
{
    private static PiiSpan Heur(int start, int end, PiiLabel label, string text) => new(start, end, label, 1, SpanSource.Heuristic, text);

    [Fact]
    public void SubstitutesHeuristicSpansWithSentinels()
        => Assert.Equal("my ssn is [SSN] ok", Premask.Apply("my ssn is 472-81-0094 ok", [Heur(10, 21, PiiLabel.Ssn, "472-81-0094")]).Masked);

    [Fact]
    public void MapsVerbatimTextBackOneToOne()
    {
        const string raw = "card 4111 1111 1111 1111 here";
        var map = Premask.Apply(raw, [Heur(5, 24, PiiLabel.CreditCard, "4111 1111 1111 1111")]);
        int at = map.Masked.IndexOf("here", StringComparison.Ordinal);
        var projected = Premask.Project(new PiiSpan(at, at + 4, PiiLabel.City, 0.9, SpanSource.Ner, "here"), raw, map);
        Assert.NotNull(projected);
        Assert.Equal("here", raw[projected.Start..projected.End]);
    }

    [Fact]
    public void ProjectsASpanInsideASentinelOntoTheSourceSpan()
    {
        const string raw = "ip 10.0.0.1 done";
        var map = Premask.Apply(raw, [Heur(3, 11, PiiLabel.IpAddress, "10.0.0.1")]);
        int at = map.Masked.IndexOf(Premask.SentinelFor(PiiLabel.IpAddress), StringComparison.Ordinal);
        var projected = Premask.Project(new PiiSpan(at, at + 4, PiiLabel.City, 0.5, SpanSource.Ner, ""), raw, map);
        Assert.NotNull(projected);
        Assert.Equal((3, 11), (projected.Start, projected.End));
    }

    [Fact]
    public void HandlesMultipleSpansInOrder()
        => Assert.Equal("ssn [SSN] and ip [IP_ADDRESS]",
            Premask.Apply("ssn 472-81-0094 and ip 10.0.0.1", [Heur(4, 15, PiiLabel.Ssn, "472-81-0094"), Heur(23, 31, PiiLabel.IpAddress, "10.0.0.1")]).Masked);

    [Fact]
    public void FeedsTheModelMaskedTextAndMergesProjectedSpans()
    {
        const string raw = "Maya at 472-81-0094 lives in Reno";
        string sawByModel = "";
        var ner = new DelegateNerDetector(text =>
        {
            sawByModel = text;
            int at = text.IndexOf("Maya", StringComparison.Ordinal);
            return [new PiiSpan(at, at + 4, PiiLabel.GivenName, 0.95, SpanSource.Ner, "Maya")];
        });
        var text = new ChatGuard(new RampartOptions { Ner = ner }).Protect(raw).Text;

        Assert.DoesNotContain("472-81-0094", sawByModel);
        Assert.Contains("[SSN]", sawByModel);
        Assert.Contains("[SSN_1]", text);
        Assert.Contains("[GIVEN_NAME_1]", text);
        Assert.DoesNotContain("472-81-0094", text);
        Assert.DoesNotContain("Maya", text);
        Assert.Contains("Reno", text);
    }
}
