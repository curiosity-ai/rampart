using Rampart.Detection;
using Rampart.Ner;
using Rampart.Session;

namespace Rampart.Tests.Upstream;

/// <summary>Port of <c>__tests__/pipeline.test.ts</c>.</summary>
public sealed class PipelineTests
{
    private static PiiSpan S(int start, int end, PiiLabel label, double score, SpanSource source = SpanSource.Ner)
        => new(start, end, label, score, source, new string('x', end - start));

    [Fact]
    public void KeepsCityStateZipAndRedactsTheRest()
    {
        var redactable = Policy.ApplyPolicy([S(0, 5, PiiLabel.City, 1), S(6, 10, PiiLabel.GivenName, 0.9), S(11, 14, PiiLabel.State, 1), S(15, 20, PiiLabel.ZipCode, 1), S(21, 30, PiiLabel.StreetName, 1)]);
        Assert.Equal([PiiLabel.GivenName, PiiLabel.StreetName], redactable.Select(s => s.Label).Order());
    }

    [Fact]
    public void MergesOverlapsPreferringHigherConfidence()
        => Assert.Equal(PiiLabel.Surname, Assert.Single(Policy.MergeSpans([S(0, 10, PiiLabel.GivenName, 0.6), S(2, 8, PiiLabel.Surname, 0.95)])).Label);

    [Fact]
    public void ReturnsRedactableSpansRightToLeft()
    {
        var output = Policy.ApplyPolicy([S(0, 3, PiiLabel.GivenName, 1), S(10, 13, PiiLabel.Ssn, 1)]);
        Assert.True(output[0].Start > output[1].Start);
    }

    [Fact]
    public void HonorsACustomKeepSet()
    {
        var spans = new[] { S(0, 5, PiiLabel.City, 1), S(6, 10, PiiLabel.GivenName, 0.9), S(11, 14, PiiLabel.State, 1) };
        Assert.Equal([PiiLabel.GivenName, PiiLabel.City, PiiLabel.State], Policy.ApplyPolicy(spans, PiiLabels.ResolveKeepLabels([])).Select(s => s.Label).Order());
    }

    [Fact]
    public void DefaultsToTheKeepLabels()
    {
        Assert.False(PiiLabels.ShouldRedact(PiiLabel.City));
        Assert.True(PiiLabels.ShouldRedact(PiiLabel.GivenName));
        Assert.Same(PiiLabels.DefaultKeepLabels, PiiLabels.ResolveKeepLabels(null));
    }

    [Fact]
    public void AssignsStablePlaceholdersPerValue()
    {
        var t = new SessionEntityTable();
        Assert.Equal("[GIVEN_NAME_1]", t.PlaceholderFor(PiiLabel.GivenName, "John"));
        Assert.Equal("[GIVEN_NAME_1]", t.PlaceholderFor(PiiLabel.GivenName, "john"));
        Assert.Equal("[GIVEN_NAME_2]", t.PlaceholderFor(PiiLabel.GivenName, "Jane"));
        Assert.Equal("[SSN_1]", t.PlaceholderFor(PiiLabel.Ssn, "888-12-3456"));
    }

    [Fact]
    public void RehydratesPlaceholders()
    {
        var t = new SessionEntityTable();
        var token = t.PlaceholderFor(PiiLabel.GivenName, "John");
        Assert.Equal("Hello John, how are you?", t.Rehydrate($"Hello {token}, how are you?"));
    }

    [Fact]
    public void RedactsGeographyWhenTheKeepSetIsEmpty()
    {
        var ner = new DelegateNerDetector(_ =>
        [
            new PiiSpan(0, 6, PiiLabel.City, 1, SpanSource.Ner, "Austin"),
            new PiiSpan(7, 9, PiiLabel.State, 1, SpanSource.Ner, "TX"),
            new PiiSpan(10, 15, PiiLabel.ZipCode, 1, SpanSource.Ner, "78701"),
            new PiiSpan(16, 20, PiiLabel.GivenName, 0.9, SpanSource.Ner, "John"),
        ]);
        var guard = new ChatGuard(new RampartOptions { Ner = ner, KeepLabels = [] });
        Assert.Equal("[CITY_1] [STATE_1] [ZIP_CODE_1] [GIVEN_NAME_1]", guard.Protect("Austin TX 78701 John").Text);
    }

    [Fact]
    public void KeepsExtraLabelsWhenTheKeepSetIsWidened()
    {
        PiiSpan[] spans = [new(0, 3, PiiLabel.BuildingNumber, 1, SpanSource.Ner, "123"), new(4, 8, PiiLabel.StreetName, 1, SpanSource.Ner, "Pine"), new(9, 15, PiiLabel.City, 1, SpanSource.Ner, "Austin")];
        var keep = PiiLabels.ResolveKeepLabels([PiiLabel.City, PiiLabel.State, PiiLabel.ZipCode, PiiLabel.BuildingNumber, PiiLabel.StreetName]);
        Assert.Empty(Policy.ApplyPolicy(spans, keep));
    }

    [Fact]
    public void ThreadsKeepLabelsThroughTheTable()
    {
        var table = new SessionEntityTable(null, PiiLabels.ResolveKeepLabels([PiiLabel.City]));
        var text = table.Scrub("Austin John", [new(0, 6, PiiLabel.City, 1, SpanSource.Ner, "Austin"), new(7, 11, PiiLabel.GivenName, 1, SpanSource.Ner, "John")]).Text;
        Assert.Equal("Austin [GIVEN_NAME_1]", text);
    }

    [Fact]
    public void ScrubsStructuredPiiButKeepsTheSurroundingText()
    {
        var text = new ChatGuard(new RampartOptions { HeuristicsOnly = true }).Protect("My ssn is 888 12 3456 and income is 50000.").Text;
        Assert.DoesNotContain("888 12 3456", text);
        Assert.Contains("[SSN_1]", text);
        Assert.Contains("50000", text);
    }

    [Fact]
    public void RoundTripsPlaceholdersThroughAModelReply()
    {
        var guard = new ChatGuard(new RampartOptions { HeuristicsOnly = true });
        guard.Protect("My ssn is 888-12-3456 please");
        Assert.Equal("I used 888-12-3456 for you.", guard.Reveal("I used [SSN_1] for you."));
    }
}
