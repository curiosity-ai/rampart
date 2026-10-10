using System.Runtime.InteropServices;
using Rampart.Buffers;
using Rampart.Detection;
using Rampart.Model;
using Rampart.Ner;

namespace Rampart;

/// <summary>Options for <see cref="RampartAnalyzer"/> and <see cref="ChatGuard"/>.</summary>
public sealed class RampartOptions
{
    /// <summary>Model to run; defaults to the embedded Rampart model (<see cref="RampartModel.Default"/>).</summary>
    public RampartModel? Model { get; init; }

    /// <summary>A pre-built contextual detector. When set, <see cref="Model"/> and <see cref="MinScore"/> are ignored.</summary>
    public INerDetector? Ner { get; init; }

    /// <summary>Skip the classifier and run the deterministic heuristics only.</summary>
    public bool HeuristicsOnly { get; init; }

    /// <summary>
    /// Do not premask structured PII before the model. Only for a model trained without the prefilter;
    /// the heuristic spans still run as a safety net.
    /// </summary>
    public bool NoPrefilter { get; init; }

    /// <summary>NER spans below this score are discarded (unless they bridge to a confident span).</summary>
    public double MinScore { get; init; } = NerDetector.DefaultMinScore;

    /// <summary>Labels to preserve in the text; <c>null</c> → CITY, STATE, ZIP_CODE.</summary>
    public IEnumerable<PiiLabel>? KeepLabels { get; init; }

    /// <summary>Placeholder display names; e.g. GIVEN_NAME → "NAME" mints <c>[NAME_1]</c>.</summary>
    public IReadOnlyDictionary<PiiLabel, string>? Aliases { get; init; }
}

/// <summary>
/// Stateless PII detection: the deterministic layer (SSN, card, email, URL, IP/MAC) plus the
/// contextual classifier, which sees the text with the structured spans already replaced by
/// sentinels (<c>[SSN]</c>, …). Thread-safe.
/// <code>
/// var analyzer = new RampartAnalyzer();
/// foreach (var span in analyzer.Analyze("My name is Alex Rivera, SSN 472-81-0094."))
///     Console.WriteLine($"{span.Label} {span.Text} {span.Score:0.00}");
/// </code>
/// </summary>
public sealed class RampartAnalyzer
{
    private readonly INerDetector? _ner;
    private readonly bool _noPrefilter;

    public RampartAnalyzer(RampartOptions? options = null)
    {
        options ??= new RampartOptions();
        _noPrefilter = options.NoPrefilter;
        if (options.Ner is not null) _ner = options.Ner;
        else if (!options.HeuristicsOnly) _ner = new NerDetector(options.Model ?? RampartModel.Default, options.MinScore);
    }

    /// <summary>
    /// Every span both layers report over <paramref name="text"/>, before merging: heuristic spans
    /// first, then model spans projected back onto the raw text. Feed this to a
    /// <see cref="Session.SessionEntityTable"/> or <see cref="Policy"/>.
    /// </summary>
    public List<PiiSpan> Detect(string text)
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
                builtIn.DetectCore(map.Masked.AsSpan(), ref spans);
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

    /// <summary>The detected entities as a disjoint, start-ordered set (kept labels included).</summary>
    public List<PiiSpan> Analyze(string text) => Policy.MergeSpans(Detect(text));
}
