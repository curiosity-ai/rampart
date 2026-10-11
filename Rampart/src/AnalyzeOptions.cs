using Catalyst.Rampart.Detection;

namespace Catalyst.Rampart;

/// <summary>
/// Per-call options for <see cref="RampartAnalyzer.Analyze(string, AnalyzeOptions?)"/>, configured
/// fluently. A new instance reproduces the original Rampart exactly.
/// <code>
/// var options = new AnalyzeOptions()
///     .WithCapitalizedParticles(false)
///     .AddHint(PiiLabel.GivenName, "John")
///     .AddHint(PiiLabel.Surname, "De La Croix", "Croix");
/// </code>
/// An instance can be reused across calls and threads: its hints are compiled once, on first use.
/// Configure it before sharing it; changing it while another thread analyzes with it is not supported.
/// </summary>
public sealed class AnalyzeOptions
{
    // Allocated by the first AddHint only.
    private Dictionary<PiiLabel, List<string>>? _hints;
    private HintSet? _compiled;

    /// <summary>Whether the capitalized-particle expansion is on (see <see cref="WithCapitalizedParticles"/>). Default true.</summary>
    public bool CapitalizedParticles { get; private set; } = true;

    /// <summary>Whether hints match regardless of case (see <see cref="WithHintsIgnoringCase"/>). Default true.</summary>
    public bool HintsIgnoreCase { get; private set; } = true;

    /// <summary>True once at least one hint value has been added.</summary>
    public bool HasHints => _hints is not null;

    /// <summary>
    /// Turns the capitalized-particle expansion on or off. It extends a detected name over a short
    /// capitalized word right next to it: a capital and up to three more letters, such as the particles
    /// in "De La Croix". On by default, as in the original Rampart. It also takes in words like "I'm" or
    /// "Call" ("Hi, I'm Alex" → GIVEN_NAME "I'm Alex"). That redacts too much but leaks nothing; off,
    /// short name parts the model did not tag can be left in the text, which hints cover for the names
    /// you know.
    /// </summary>
    public AnalyzeOptions WithCapitalizedParticles(bool enabled)
    {
        CapitalizedParticles = enabled;
        return this;
    }

    /// <summary>
    /// Adds values of <paramref name="label"/> that
    /// are always reported, such as the name of the person sending the message. Call it as often as you
    /// like: values accumulate per type, and a value already added for the type is ignored.
    /// <list type="bullet">
    /// <item><b>Always reported.</b> Every occurrence of each value is reported as the type, with score 1,
    /// even when the model misses it and even when the type is in <see cref="RampartOptions.KeepLabels"/>.
    /// A value matches whole words only, with any spacing between them. When two values overlap in the
    /// text ("John De La Croix" and "De La Croix"), the spans are merged into one.</item>
    /// <item><b>Expansion.</b> The words of all of a type's values also extend a span of that type over the
    /// words right before it, then right after it, whatever their case or length, and whether or not
    /// the capitalized-particle expansion is on. With <c>"John De La Croix"</c> as a hint, a detected
    /// "Croix" grows to "De La Croix". The two name types share their words, because the model splits
    /// one person's name across GIVEN_NAME and SURNAME.</item>
    /// </list>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="label"/> is not a defined <see cref="PiiLabel"/>.</exception>
    public AnalyzeOptions AddHint(PiiLabel label, params string[] values)
    {
        if (!Enum.IsDefined(label)) throw new ArgumentOutOfRangeException(nameof(label), label, "Not a defined PiiLabel.");
        if (values is null) return this;

        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value)) continue;
            _hints ??= [];
            if (!_hints.TryGetValue(label, out var list)) _hints[label] = list = [];
            var trimmed = value.Trim();
            if (!list.Contains(trimmed, StringComparer.Ordinal))
            {
                list.Add(trimmed);
                _compiled = null;
            }
        }
        return this;
    }

    /// <summary>
    /// Whether hints match regardless of case. On by default, so a hint "John" also finds "john"; turn it
    /// off for a hint that is also a common word ("Will", "Mark", "May").
    /// </summary>
    public AnalyzeOptions WithHintsIgnoringCase(bool ignoreCase)
    {
        if (HintsIgnoreCase != ignoreCase) _compiled = null;
        HintsIgnoreCase = ignoreCase;
        return this;
    }

    /// <summary>The hint values added so far, by entity type. Empty when there are none.</summary>
    public IReadOnlyDictionary<PiiLabel, IReadOnlyList<string>> Hints
        => _hints is null
            ? new Dictionary<PiiLabel, IReadOnlyList<string>>()
            : _hints.ToDictionary(kv => kv.Key, kv => (IReadOnlyList<string>)kv.Value.AsReadOnly());

    internal HintSet? CompiledHints => _hints is null ? null : _compiled ??= HintSet.Build(_hints, HintsIgnoreCase);
}
