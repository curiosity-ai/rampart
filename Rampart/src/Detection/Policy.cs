using Rampart.Buffers;

namespace Rampart.Detection;

/// <summary>
/// Span reconciliation and the default-deny policy. Detectors (heuristic + NER) emit overlapping,
/// possibly conflicting spans; <see cref="MergeSpans"/> resolves them into a disjoint set and
/// <see cref="ApplyPolicy"/> drops anything in the keep-set so only redactable spans remain.
/// </summary>
public static class Policy
{
    /// <summary>
    /// Reduce overlapping spans to a disjoint set. When two spans overlap the higher-confidence one
    /// wins; ties break toward the longer span, then toward heuristics. On a partial overlap the
    /// result covers the byte-union under the winner's label, so no detected bytes are dropped; full
    /// containment collapses to the winner.
    /// </summary>
    public static List<PiiSpan> MergeSpans(IEnumerable<PiiSpan> spans)
    {
        var input = new PooledList<PiiSpan>(16);
        var merged = new PooledList<PiiSpan>(16);
        try
        {
            foreach (var s in spans) input.Add(s);
            MergeInto(input.AsSpan(), ref merged);
            return [.. merged.AsSpan()];
        }
        finally
        {
            input.Dispose();
            merged.Dispose();
        }
    }

    /// <summary>
    /// Apply the keep-set. Returns only spans that must be redacted, sorted right to left so callers
    /// can splice from the end and keep earlier offsets valid.
    /// </summary>
    public static List<PiiSpan> ApplyPolicy(IEnumerable<PiiSpan> spans, IReadOnlySet<PiiLabel>? keepLabels = null)
    {
        var input = new PooledList<PiiSpan>(16);
        var output = new PooledList<PiiSpan>(16);
        try
        {
            foreach (var s in spans) input.Add(s);
            ApplyPolicyInto(input.AsSpan(), keepLabels, ref output);
            return [.. output.AsSpan()];
        }
        finally
        {
            input.Dispose();
            output.Dispose();
        }
    }

    /// <summary><see cref="ApplyPolicy"/> into a pooled list (right to left).</summary>
    internal static void ApplyPolicyInto(ReadOnlySpan<PiiSpan> spans, IReadOnlySet<PiiLabel>? keepLabels, ref PooledList<PiiSpan> output)
    {
        var keep = keepLabels ?? PiiLabels.DefaultKeepLabels;
        var merged = new PooledList<PiiSpan>(spans.Length);
        try
        {
            MergeInto(spans, ref merged);
            // Merged spans are disjoint and non-empty, so start order is strict and right-to-left is a reversal.
            for (int i = merged.Count - 1; i >= 0; i--)
            {
                if (!keep.Contains(merged[i].Label)) output.Add(merged[i]);
            }
        }
        finally
        {
            merged.Dispose();
        }
    }

    /// <summary><see cref="MergeSpans"/> into a pooled list (left to right).</summary>
    internal static void MergeInto(ReadOnlySpan<PiiSpan> spans, ref PooledList<PiiSpan> merged)
    {
        // Stable sort, as Array.prototype.sort: start ascending, then end descending, then input order.
        var order = new PooledList<(PiiSpan Span, int Index)>(spans.Length);
        try
        {
            for (int i = 0; i < spans.Length; i++) order.Add((spans[i], i));
            order.AsSpan().Sort(new StartThenLongest());

            int first = merged.Count;
            foreach (var (span, _) in order.AsSpan())
            {
                if (merged.Count == first || span.Start >= merged[merged.Count - 1].End)
                {
                    merged.Add(span);
                    continue;
                }
                ref var prev = ref merged[merged.Count - 1];
                var winner = Preferred(prev, span);
                bool prevContains = prev.Start <= span.Start && prev.End >= span.End;
                bool spanContains = span.Start <= prev.Start && span.End >= prev.End;
                // Partial overlap: union the bytes under the preferred label. The reference keeps the
                // winner's text (not the union's) as the value to rehydrate; so does this port.
                prev = prevContains || spanContains ? winner : winner with { Start = Math.Min(prev.Start, span.Start), End = Math.Max(prev.End, span.End) };
            }
        }
        finally
        {
            order.Dispose();
        }
    }

    private readonly struct StartThenLongest : IComparer<(PiiSpan Span, int Index)>
    {
        public int Compare((PiiSpan Span, int Index) x, (PiiSpan Span, int Index) y)
        {
            int c = x.Span.Start.CompareTo(y.Span.Start);
            if (c != 0) return c;
            c = y.Span.End.CompareTo(x.Span.End);
            return c != 0 ? c : x.Index.CompareTo(y.Index);
        }
    }

    private static PiiSpan Preferred(PiiSpan a, PiiSpan b)
    {
        if (a.Score != b.Score) return a.Score > b.Score ? a : b;
        if (a.Length != b.Length) return a.Length > b.Length ? a : b;
        return a.Source == SpanSource.Heuristic ? a : b;
    }
}
