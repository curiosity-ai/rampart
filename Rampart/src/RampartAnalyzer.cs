using System.Runtime.InteropServices;
using Rampart.Buffers;
using Rampart.Detection;
using Rampart.Model;
using Rampart.Ner;

namespace Rampart;

/// <summary>Options for <see cref="RampartAnalyzer"/>.</summary>
public sealed class RampartOptions
{
    /// <summary>Model to run; defaults to the embedded Rampart model (<see cref="RampartModel.Default"/>).</summary>
    public RampartModel? Model { get; init; }

    /// <summary>Model entities below this score are discarded (unless they bridge to a confident one). Default 0.4.</summary>
    public double MinScore { get; init; } = NerDetector.DefaultMinScore;

    /// <summary>
    /// Entity types the analyzer does not report, so they stay in the text. <c>null</c> (the default)
    /// keeps CITY, STATE and ZIP_CODE, as the original Rampart does; an empty set reports everything.
    /// </summary>
    public IEnumerable<PiiLabel>? KeepLabels { get; init; }

    // The reference's test hooks, kept for the ported upstream tests.
    internal INerDetector? Ner { get; init; }
    internal bool HeuristicsOnly { get; init; }
    internal bool NoPrefilter { get; init; }
    internal IReadOnlyDictionary<PiiLabel, string>? Aliases { get; init; }
}

/// <summary>
/// Finds the PII in a text, like Catalyst.Presidio's <c>PresidioAnalyzer</c>: the deterministic
/// recognizers (SSN, credit card, email, URL, IP/MAC) plus the Rampart model, which sees the text with
/// the structured entities already masked. Thread-safe; create one and share it.
/// <code>
/// var text = "My name is Alex Rivera, SSN 472-81-0094.";
/// var results = new RampartAnalyzer().Analyze(text);
/// var anonymized = new RampartAnonymizer().Anonymize(text, results);   // "My name is **** ******, SSN ***********."
/// </code>
/// </summary>
public sealed class RampartAnalyzer
{
    private readonly INerDetector? _ner;
    private readonly bool _noPrefilter;

    private readonly IReadOnlySet<PiiLabel> _keepLabels;

    public RampartAnalyzer(RampartOptions? options = null)
    {
        options ??= new RampartOptions();
        _keepLabels = PiiLabels.ResolveKeepLabels(options.KeepLabels);
        _noPrefilter = options.NoPrefilter;
        if (options.Ner is not null) _ner = options.Ner;
        else if (!options.HeuristicsOnly) _ner = new NerDetector(options.Model ?? RampartModel.Default, options.MinScore);
    }

    /// <summary>
    /// The PII in <paramref name="text"/>: disjoint, ordered by position, without the kept entity
    /// types (<see cref="RampartOptions.KeepLabels"/>). Pass the results to
    /// <see cref="RampartAnonymizer"/> or <see cref="RampartCloaker"/>.
    /// </summary>
    /// <param name="text">The text to analyze.</param>
    /// <param name="options">
    /// Per-call options: the capitalized-particle expansion and hints of values to always report.
    /// <c>null</c> (the default) behaves exactly like the original Rampart.
    /// </param>
    public List<RecognizerResult> Analyze(string text, AnalyzeOptions? options = null)
    {
        var hinted = options is { HasHints: true } ? new List<(int Start, int End, PiiLabel Label)>() : null;
        var results = new List<RecognizerResult>();
        foreach (var span in Policy.MergeSpans(Detect(text, options, hinted)))
        {
            // A hinted value is reported even when its type is kept: the caller asked for it.
            if (_keepLabels.Contains(span.Label) && (hinted is null || !Overlaps(hinted, span.Start, span.End))) continue;
            results.Add(new RecognizerResult(span.Start, span.End - 1, span.Label.ToName(), (float)span.Score));
        }
        return results;
    }

    private static bool Overlaps(List<(int Start, int End, PiiLabel Label)> ranges, int start, int end)
    {
        foreach (var r in ranges) if (r.Start < end && start < r.End) return true;
        return false;
    }

    /// <summary>
    /// <see cref="Detect(string)"/> with <paramref name="options"/> applied: the model's span repair
    /// follows its particle and hint settings, and every hint occurrence the layers did not already
    /// cover is added as a span of score 1 (and recorded in <paramref name="hinted"/>).
    /// </summary>
    internal List<PiiSpan> Detect(string text, AnalyzeOptions? options, List<(int Start, int End, PiiLabel Label)>? hinted = null)
    {
        if (options is null) return Detect(text);
        var hints = options.CompiledHints;
        var spans = Detect(text, new NerDetector.RepairOptions(!options.CapitalizedParticles, hints));
        if (hints is null) return spans;

        foreach (var (start, end, label) in hints.FindOccurrences(text))
        {
            hinted?.Add((start, end, label));
            if (IsCovered(spans, text, start, end)) continue;
            spans.Add(new PiiSpan(start, end, label, 1.0, SpanSource.Heuristic, text.Substring(start, end - start)));
        }
        return spans;
    }

    /// <summary>True when every word character of <c>text[start..end)</c> is already inside a detected span.</summary>
    private static bool IsCovered(List<PiiSpan> spans, string text, int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            if (!HintSet.IsWordChar(text[i])) continue;
            bool inside = false;
            foreach (var s in spans)
            {
                if (s.Start <= i && i < s.End) { inside = true; break; }
            }
            if (!inside) return false;
        }
        return true;
    }

    /// <summary>
    /// Every span both layers report over <paramref name="text"/>, before merging: heuristic spans
    /// first, then model spans projected back onto the raw text (end exclusive). Exactly the reference.
    /// </summary>
    internal List<PiiSpan> Detect(string text) => Detect(text, default(NerDetector.RepairOptions));

    private List<PiiSpan> Detect(string text, NerDetector.RepairOptions repair)
    {
        ArgumentNullException.ThrowIfNull(text);
        var heuristic = Heuristics.Detect(text);
        if (_ner is null) return heuristic;

        if (_noPrefilter)
        {
            heuristic.AddRange(_ner.Detect(text));
            return heuristic;
        }

        if (_ner is NerDetector builtIn)
        {
            // Allocation-free path: premask into pooled buffers, run the model on the masked span, and
            // project each model span straight onto the raw text.
            var map = new PremaskBuffer(text.Length);
            var spans = new PooledList<NerDetector.NerSpan>(16);
            try
            {
                map.Build(text, CollectionsMarshal.AsSpan(heuristic));
                builtIn.DetectCore(map.Masked.AsSpan(), ref spans, repair);
                foreach (var s in spans.AsSpan())
                {
                    if (map.TryProject(s.Start, s.End, out int start, out int end))
                    {
                        heuristic.Add(new PiiSpan(start, end, s.Label, s.Score, SpanSource.Ner, text.Substring(start, end - start)));
                    }
                }
                return heuristic;
            }
            finally
            {
                map.Dispose();
                spans.Dispose();
            }
        }

        var result = Premask.Apply(text, heuristic);
        var contextual = new List<PiiSpan>();
        foreach (var span in _ner.Detect(result.Masked))
        {
            if (Premask.Project(span, text, result) is { } projected) contextual.Add(projected);
        }
        heuristic.AddRange(contextual);
        return heuristic;
    }
}
