using System.Text;

namespace Rampart.Detection;

/// <summary>A masked copy of the input plus a map from each masked character to raw offsets.</summary>
/// <param name="Masked">Input with heuristic spans replaced by sentinel tokens.</param>
/// <param name="RawStart"><c>RawStart[i]</c> is the raw offset of the source of <c>Masked[i]</c>.</param>
/// <param name="RawEnd"><c>RawEnd[i]</c> is the raw offset just past that source.</param>
public sealed record PremaskResult(string Masked, int[] RawStart, int[] RawEnd);

/// <summary>
/// Pre-mask harness: substitute deterministically detected structured PII with neutral sentinels
/// (<c>[SSN]</c>, <c>[CREDIT_CARD]</c>, <c>[EMAIL]</c>, …) <em>before</em> the contextual model runs.
/// The model was trained on text masked the same way, so train- and inference-time inputs match.
/// The heuristic spans themselves keep their raw offsets: only the text handed to the model is
/// masked, never the text that gets placeholdered.
/// </summary>
public static class Premask
{
    /// <summary>The sentinel substituted for a premasked label. Stable across train/serve.</summary>
    public static string SentinelFor(PiiLabel label) => $"[{label.ToName()}]";

    /// <summary>
    /// Replace each (merged, disjoint) heuristic span in <paramref name="raw"/> with its sentinel,
    /// recording per masked-character raw offsets. Verbatim runs map 1:1 to raw, and sentinel
    /// characters map onto the whole span they stand in for.
    /// </summary>
    public static PremaskResult Apply(string raw, IEnumerable<PiiSpan> spans)
    {
        var ordered = Policy.MergeSpans(spans).OrderBy(s => s.Start).ToList();
        var masked = new StringBuilder(raw.Length);
        var rawStart = new List<int>(raw.Length);
        var rawEnd = new List<int>(raw.Length);

        void CopyVerbatim(int from, int to)
        {
            for (int i = from; i < to; i++)
            {
                masked.Append(raw[i]);
                rawStart.Add(i);
                rawEnd.Add(i + 1);
            }
        }

        int cursor = 0;
        foreach (var span in ordered)
        {
            if (span.Start < cursor) continue; // defensive: skip any residual overlap
            CopyVerbatim(cursor, span.Start);
            foreach (var ch in SentinelFor(span.Label))
            {
                masked.Append(ch);
                rawStart.Add(span.Start);
                rawEnd.Add(span.End);
            }
            cursor = span.End;
        }
        CopyVerbatim(cursor, raw.Length);
        return new PremaskResult(masked.ToString(), [.. rawStart], [.. rawEnd]);
    }

    /// <summary>
    /// Project a span reported in masked coordinates back onto the raw input; <c>null</c> for an
    /// empty or degenerate span. The text is re-sliced from <paramref name="raw"/>.
    /// </summary>
    public static PiiSpan? Project(PiiSpan span, string raw, PremaskResult map)
    {
        if (span.End <= span.Start) return null;
        if ((uint)span.Start >= (uint)map.RawStart.Length || (uint)(span.End - 1) >= (uint)map.RawEnd.Length) return null;
        int start = map.RawStart[span.Start];
        int end = map.RawEnd[span.End - 1];
        if (end <= start) return null;
        return span with { Start = start, End = end, Text = raw[start..end] };
    }
}
