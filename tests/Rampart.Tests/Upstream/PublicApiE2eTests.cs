using Catalyst.Rampart.Ner;

namespace Catalyst.Rampart.Tests.Upstream;

/// <summary>
/// Port of <c>__tests__/public-api-e2e.test.ts</c>: every public eval case through the guard with a
/// fixture NER that reports exactly the case's private terms, isolating the policy/placeholder layer.
/// </summary>
public sealed class PublicApiE2eTests
{
    public static TheoryData<string> Cases()
    {
        var data = new TheoryData<string>();
        foreach (var c in ReferenceFixture.Cases.Where(c => c.TryGetProperty("privateTerms", out var p) && p.ValueKind == System.Text.Json.JsonValueKind.Array))
            data.Add(c.GetProperty("id").GetString()!);
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void RedactsPrivateTermsKeepsPublicTermsAndRoundTrips(string id)
    {
        var c = ReferenceFixture.Case(id);
        var privateTerms = c.GetProperty("privateTerms").EnumerateArray()
            .Select(t => (Text: t.GetProperty("text").GetString()!, Label: Enum.GetValues<PiiLabel>().Single(l => l.ToName() == t.GetProperty("label").GetString())))
            .ToList();
        var publicTerms = c.GetProperty("publicTerms").EnumerateArray().Select(t => t.GetString()!).ToList();

        var ner = new DelegateNerDetector(text =>
        {
            var spans = new List<PiiSpan>();
            foreach (var (term, label) in privateTerms)
            {
                for (int at = text.IndexOf(term, StringComparison.Ordinal); at >= 0; at = text.IndexOf(term, at + term.Length, StringComparison.Ordinal))
                    spans.Add(new PiiSpan(at, at + term.Length, label, 0.99, SpanSource.Ner, term));
            }
            return spans;
        });
        var guard = new ChatGuard(new RampartOptions { Ner = ner });
        var result = guard.Protect(c.GetProperty("text").GetString()!);

        foreach (var (term, _) in privateTerms) Assert.DoesNotContain(term, result.Text);
        foreach (var term in publicTerms) Assert.Contains(term, result.Text);

        var roundTrip = guard.Reveal(string.Join(" ", result.Placeholders));
        foreach (var (term, _) in privateTerms) Assert.Contains(term, roundTrip);
    }
}
