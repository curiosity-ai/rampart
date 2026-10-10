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
        // Stable sort, as Array.prototype.sort: start ascending, then end descending.
        var sorted = spans.OrderBy(s => s.Start).ThenByDescending(s => s.End);
        var merged = new List<PiiSpan>();
        foreach (var span in sorted)
        {
            if (merged.Count == 0 || span.Start >= merged[^1].End)
            {
                merged.Add(span);
                continue;
            }
            var prev = merged[^1];
            var winner = Preferred(prev, span);
            bool prevContains = prev.Start <= span.Start && prev.End >= span.End;
            bool spanContains = span.Start <= prev.Start && span.End >= prev.End;
            if (prevContains || spanContains)
            {
                merged[^1] = winner;
            }
            else
            {
                // Partial overlap: union the bytes under the preferred label. The reference keeps the
                // winner's text (not the union's) as the value to rehydrate; so does this port.
                merged[^1] = winner with { Start = Math.Min(prev.Start, span.Start), End = Math.Max(prev.End, span.End) };
            }
        }
        return merged;
    }

    private static PiiSpan Preferred(PiiSpan a, PiiSpan b)
    {
        if (a.Score != b.Score) return a.Score > b.Score ? a : b;
        if (a.Length != b.Length) return a.Length > b.Length ? a : b;
        return a.Source == SpanSource.Heuristic ? a : b;
    }

    /// <summary>
    /// Apply the keep-set. Returns only spans that must be redacted, sorted right to left so callers
    /// can splice from the end and keep earlier offsets valid.
    /// </summary>
    public static List<PiiSpan> ApplyPolicy(IEnumerable<PiiSpan> spans, IReadOnlySet<PiiLabel>? keepLabels = null)
    {
        var keep = keepLabels ?? PiiLabels.DefaultKeepLabels;
        return MergeSpans(spans).Where(s => !keep.Contains(s.Label)).OrderByDescending(s => s.Start).ToList();
    }
}
