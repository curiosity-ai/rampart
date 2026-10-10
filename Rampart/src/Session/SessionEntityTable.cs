using System.Text;
using System.Text.RegularExpressions;
using Rampart.Detection;
using Rampart.Tokenization;

namespace Rampart.Session;

/// <summary>Result of scrubbing one message.</summary>
/// <param name="Text">Text with PII replaced by placeholders. Safe to send or log.</param>
/// <param name="Placeholders">Placeholders introduced or reused in this message, left to right.</param>
public sealed record ScrubResult(string Text, IReadOnlyList<string> Placeholders);

/// <summary>
/// A per-conversation store mapping raw PII values to stable, typed placeholders
/// (<c>[GIVEN_NAME_1]</c>, <c>[SSN_2]</c>). The same label + value always maps to the same token,
/// so the assistant can reason about "GIVEN_NAME_1" across turns, and <see cref="Rehydrate"/> puts
/// the real value back into its reply before display. The table lives only on the client.
/// <para>Not thread-safe: use one table (one <see cref="ChatGuard"/>) per conversation.</para>
/// </summary>
public sealed partial class SessionEntityTable
{
    private readonly Dictionary<string, string> _forward = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _reverse = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _counters = new(StringComparer.Ordinal);
    private readonly IReadOnlyDictionary<PiiLabel, string> _aliases;
    private readonly IReadOnlySet<PiiLabel> _keepLabels;

    /// <summary>
    /// Token shape minted by the table, and the pattern used to find tokens on the way back, kept in
    /// one place so scrub and rehydrate can never disagree.
    /// </summary>
    [GeneratedRegex(@"\[[A-Z][A-Z_]*_[0-9]+\]", RegexOptions.CultureInvariant)]
    public static partial Regex PlaceholderPattern();

    [GeneratedRegex(@"[\t\n\v\f\r \u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\uFEFF]+", RegexOptions.CultureInvariant)]
    private static partial Regex JsWhitespaceRun();

    /// <param name="aliases">Display names for placeholder tokens (e.g. GIVEN_NAME → "NAME" gives <c>[NAME_1]</c>).</param>
    /// <param name="keepLabels">Labels preserved in the text; <c>null</c> → CITY, STATE, ZIP_CODE.</param>
    public SessionEntityTable(IReadOnlyDictionary<PiiLabel, string>? aliases = null, IReadOnlySet<PiiLabel>? keepLabels = null)
    {
        _aliases = aliases ?? new Dictionary<PiiLabel, string>();
        _keepLabels = keepLabels ?? PiiLabels.DefaultKeepLabels;
    }

    private string DisplayName(PiiLabel label) => _aliases.TryGetValue(label, out var alias) ? alias : label.ToName();

    /// <summary>Get or mint the placeholder for a label + value. Idempotent; casing and whitespace are ignored.</summary>
    public string PlaceholderFor(PiiLabel label, string value)
    {
        var normalized = JsChar.Trim(JsWhitespaceRun().Replace(WordPieceTokenizer.JsLowerCase(value), " "));
        var key = $"{label.ToName()}:{normalized}";
        if (_forward.TryGetValue(key, out var existing)) return existing;
        var name = DisplayName(label);
        int next = _counters.GetValueOrDefault(name) + 1;
        _counters[name] = next;
        var token = $"[{name}_{next}]";
        _forward[key] = token;
        _reverse[token] = value;
        return token;
    }

    /// <summary>Replace each redactable span (after merge and keep-set policy) with its placeholder.</summary>
    public ScrubResult Scrub(string raw, IEnumerable<PiiSpan> spans)
    {
        var redactable = Policy.ApplyPolicy(spans, _keepLabels); // right to left
        var placeholders = new List<string>(redactable.Count);
        var text = new StringBuilder(raw);
        foreach (var span in redactable)
        {
            var token = PlaceholderFor(span.Label, span.Text);
            placeholders.Add(token);
            text.Remove(span.Start, span.End - span.Start).Insert(span.Start, token);
        }
        placeholders.Reverse();
        return new ScrubResult(text.ToString(), placeholders);
    }

    /// <summary>Restore real values in an assistant reply. Unknown tokens are left intact.</summary>
    public string Rehydrate(string text) => PlaceholderPattern().Replace(text, m => _reverse.GetValueOrDefault(m.Value, m.Value));

    /// <summary>The real value behind a placeholder, or <c>null</c> when the table does not know it.</summary>
    public string? Resolve(string token) => _reverse.GetValueOrDefault(token);

    /// <summary>True if <paramref name="token"/> is a placeholder this table can resolve.</summary>
    public bool Knows(string token) => _reverse.ContainsKey(token);

    /// <summary>Every placeholder minted so far and the value it stands for.</summary>
    public IReadOnlyDictionary<string, string> Entries => _reverse;
}
