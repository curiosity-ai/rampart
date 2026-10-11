using Rampart.Detection;

namespace Rampart;

/// <summary>
/// Per-call options for <see cref="RampartAnalyzer.Analyze(string, AnalyzeOptions?)"/>. The defaults
/// reproduce the original Rampart exactly. An instance can be reused across calls: its hints are
/// compiled once, on first use.
/// </summary>
public sealed class AnalyzeOptions
{
    /// <summary>
    /// Extend a detected name over a short capitalized word right next to it: a capital and up to three
    /// more letters, such as the particles in "De La Croix". On by default, as in the original Rampart.
    /// It also takes in words like "I'm" or "Call" ("Hi, I'm Alex" → GIVEN_NAME "I'm Alex"). That
    /// redacts too much but leaks nothing; turn it off to report the name alone, and use
    /// <see cref="Hints"/> to keep the particles of names you know.
    /// </summary>
    public bool ExpandCapitalizedParticles { get; init; } = true;

    /// <summary>
    /// Values that are always reported, by entity type (<c>GIVEN_NAME</c>, <c>SURNAME</c>, <c>EMAIL</c>, …),
    /// such as the name of the person sending the message:
    /// <code>
    /// Hints = new Dictionary&lt;string, string[]&gt;
    /// {
    ///     ["GIVEN_NAME"] = ["John"],
    ///     ["SURNAME"] = ["De La Croix"],
    /// }
    /// </code>
    /// <list type="bullet">
    /// <item><b>Always reported.</b> Every occurrence of a value is reported as its type, with score 1,
    /// even when the model misses it and even when the type is in <see cref="RampartOptions.KeepLabels"/>.
    /// A value matches whole words only, with any spacing between them.</item>
    /// <item><b>Expansion.</b> The words of a type's values also extend a span of that type over the
    /// words right before or after it, whatever their case or length, and whether or not
    /// <see cref="ExpandCapitalizedParticles"/> is on. With <c>"John De La Croix"</c> as a hint, a
    /// detected "Croix" grows to "De La Croix". The two name types share their words, because the model
    /// splits one person's name across GIVEN_NAME and SURNAME.</item>
    /// </list>
    /// An unknown entity type throws <see cref="ArgumentException"/>.
    /// </summary>
    public IReadOnlyDictionary<string, string[]>? Hints { get; init; }

    /// <summary>
    /// Match <see cref="Hints"/> regardless of case (the default), so a hint "John" also finds "john".
    /// Turn it off for a hint that is also a common word ("Will", "Mark", "May").
    /// </summary>
    public bool HintsIgnoreCase { get; init; } = true;

    private HintSet? _compiled;

    internal HintSet? CompiledHints => Hints is null ? null : _compiled ??= HintSet.Build(Hints, HintsIgnoreCase);
}
