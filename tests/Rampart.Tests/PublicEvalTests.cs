namespace Catalyst.Rampart.Tests;

/// <summary>
/// The upstream package's public end-to-end eval (<c>eval/public-cases.ts</c>, 58 cases in 0.1.3) with its
/// strict gate: private-term recall and public-term retention must stay at or above 95%, and must
/// equal what the original pipeline scores on the same cases.
/// </summary>
public sealed class PublicEvalTests
{
    private const double StrictMinPrivateRecall = 0.95;
    private const double StrictMinPublicRetention = 0.95;

    [Fact]
    public void StrictGateAndParityWithReference()
    {
        var analyzer = new RampartAnalyzer();
        int privateFound = 0, privateTotal = 0, publicKept = 0, publicTotal = 0;
        int refPrivateFound = 0, refPublicKept = 0;
        var leaks = new List<string>();

        foreach (var c in ReferenceFixture.Cases.Where(c => c.TryGetProperty("privateTerms", out var p) && p.ValueKind == System.Text.Json.JsonValueKind.Array))
        {
            var input = c.GetProperty("text").GetString()!;
            var output = new ChatGuard(analyzer).Protect(input).Text;
            var reference = c.GetProperty("protected").GetProperty("text").GetString()!;

            foreach (var term in c.GetProperty("privateTerms").EnumerateArray().Select(t => t.GetProperty("text").GetString()!))
            {
                privateTotal++;
                if (!output.Contains(term, StringComparison.Ordinal)) privateFound++;
                else leaks.Add($"{c.GetProperty("id").GetString()}: {term}");
                if (!reference.Contains(term, StringComparison.Ordinal)) refPrivateFound++;
            }
            foreach (var term in c.GetProperty("publicTerms").EnumerateArray().Select(t => t.GetString()!))
            {
                publicTotal++;
                if (output.Contains(term, StringComparison.Ordinal)) publicKept++;
                if (reference.Contains(term, StringComparison.Ordinal)) refPublicKept++;
            }
        }

        var log = TestContext.Current.TestOutputHelper;
        log?.WriteLine($"Private recall: {privateFound}/{privateTotal}; public retention: {publicKept}/{publicTotal}");
        foreach (var leak in leaks) log?.WriteLine($"leak - {leak}");

        Assert.Equal(58, ReferenceFixture.Cases.Count(c => c.TryGetProperty("privateTerms", out var p) && p.ValueKind == System.Text.Json.JsonValueKind.Array));
        Assert.Equal(refPrivateFound, privateFound);
        Assert.Equal(refPublicKept, publicKept);
        Assert.True((double)privateFound / privateTotal >= StrictMinPrivateRecall, $"recall {privateFound}/{privateTotal}");
        Assert.True((double)publicKept / publicTotal >= StrictMinPublicRetention, $"retention {publicKept}/{publicTotal}");
    }
}
