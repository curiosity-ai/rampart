using System.Text.Json;

namespace Rampart.Tests;

/// <summary>
/// Outputs of the original Rampart pipeline (@nationaldesignstudio/rampart 0.1.3 on transformers.js
/// 3.7.5 + onnxruntime-node), produced by <c>tools/reference/generate-fixtures.ts</c>.
/// </summary>
internal static class ReferenceFixture
{
    private static readonly Lazy<JsonDocument> _doc = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "reference.json"))));

    public static JsonElement Root => _doc.Value.RootElement;

    public static IEnumerable<JsonElement> Cases => Root.GetProperty("cases").EnumerateArray();

    public static JsonElement Case(string id) => Cases.Single(c => c.GetProperty("id").GetString() == id);

    public static TheoryData<string> CaseIds()
    {
        var data = new TheoryData<string>();
        foreach (var c in Cases) data.Add(c.GetProperty("id").GetString()!);
        return data;
    }

    public static List<PiiSpan> Spans(JsonElement array) => array.EnumerateArray().Select(Span).ToList();

    public static PiiSpan Span(JsonElement e)
    {
        Assert.True(PiiLabels.TryParse(e.GetProperty("label").GetString()!, out var label));
        return new PiiSpan(
            e.GetProperty("start").GetInt32(),
            e.GetProperty("end").GetInt32(),
            label,
            e.GetProperty("score").GetDouble(),
            e.GetProperty("source").GetString() == "heuristic" ? SpanSource.Heuristic : SpanSource.Ner,
            e.GetProperty("text").GetString()!);
    }

    /// <summary>Asserts two span lists agree on everything but float noise in the score.</summary>
    public static void AssertSpansEqual(IReadOnlyList<PiiSpan> expected, IReadOnlyList<PiiSpan> actual, string context, double scoreTolerance = 1e-4)
    {
        string Show(IEnumerable<PiiSpan> s) => string.Join("\n  ", s.Select(x => $"[{x.Start},{x.End}) {x.Label.ToName()} {x.Source} {x.Score:F6} \"{x.Text}\""));
        var message = $"{context}\nexpected:\n  {Show(expected)}\nactual:\n  {Show(actual)}";
        Assert.True(expected.Count == actual.Count, message);
        for (int i = 0; i < expected.Count; i++)
        {
            var e = expected[i];
            var a = actual[i];
            Assert.True(e.Start == a.Start && e.End == a.End && e.Label == a.Label && e.Source == a.Source && e.Text == a.Text, message);
            Assert.True(Math.Abs(e.Score - a.Score) <= scoreTolerance, $"score {a.Score} vs {e.Score}\n{message}");
        }
    }
}
