using Rampart.Buffers;

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
    public static string SentinelFor(PiiLabel label) => Sentinels[(int)label];

    private static readonly string[] Sentinels = Enum.GetValues<PiiLabel>().Select(l => $"[{l.ToName()}]").ToArray();

    /// <summary>
    /// Replace each (merged, disjoint) heuristic span in <paramref name="raw"/> with its sentinel,
    /// recording per masked-character raw offsets. Verbatim runs map 1:1 to raw, and sentinel
    /// characters map onto the whole span they stand in for.
    /// </summary>
    public static PremaskResult Apply(string raw, IEnumerable<PiiSpan> spans)
    {
        var input = new PooledList<PiiSpan>(16);
        var map = new PremaskBuffer(raw.Length);
        try
        {
            foreach (var s in spans) input.Add(s);
            map.Build(raw, input.AsSpan());
            return new PremaskResult(map.Masked.AsSpan().ToString(), map.RawStart.AsSpan().ToArray(), map.RawEnd.AsSpan().ToArray());
        }
        finally
        {
            input.Dispose();
            map.Dispose();
        }
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

/// <summary>The pooled form of <see cref="PremaskResult"/> the analyzer builds on every call.</summary>
internal struct PremaskBuffer(int capacity) : IDisposable
{
    public PooledList<char> Masked = new(capacity + 32);
    public PooledList<int> RawStart = new(capacity + 32);
    public PooledList<int> RawEnd = new(capacity + 32);

    /// <summary>Replace each merged heuristic span with its sentinel, recording per-character raw offsets.</summary>
    public void Build(ReadOnlySpan<char> raw, ReadOnlySpan<PiiSpan> spans)
    {
        var ordered = new PooledList<PiiSpan>(spans.Length);
        try
        {
            Policy.MergeInto(spans, ref ordered); // already in start order
            int cursor = 0;
            foreach (var span in ordered.AsSpan())
            {
                if (span.Start < cursor) continue; // defensive: skip any residual overlap
                CopyVerbatim(raw, cursor, span.Start);
                foreach (var ch in Premask.SentinelFor(span.Label))
                {
                    Masked.Add(ch);
                    RawStart.Add(span.Start);
                    RawEnd.Add(span.End);
                }
                cursor = span.End;
            }
            CopyVerbatim(raw, cursor, raw.Length);
        }
        finally
        {
            ordered.Dispose();
        }
    }

    private void CopyVerbatim(ReadOnlySpan<char> raw, int from, int to)
    {
        for (int i = from; i < to; i++)
        {
            Masked.Add(raw[i]);
            RawStart.Add(i);
            RawEnd.Add(i + 1);
        }
    }

    /// <summary>Project masked <c>[start, end)</c> onto the raw text; false for an empty or degenerate span.</summary>
    public readonly bool TryProject(int start, int end, out int rawStart, out int rawEnd)
    {
        rawStart = rawEnd = 0;
        if (end <= start || (uint)start >= (uint)RawStart.Count || (uint)(end - 1) >= (uint)RawEnd.Count) return false;
        rawStart = RawStart[start];
        rawEnd = RawEnd[end - 1];
        return rawEnd > rawStart;
    }

    public void Dispose()
    {
        Masked.Dispose();
        RawStart.Dispose();
        RawEnd.Dispose();
    }
}
