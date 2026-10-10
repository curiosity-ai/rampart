using System.Runtime.CompilerServices;
using Rampart.Buffers;
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
    private readonly Dictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> _forwardLookup;
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


    /// <param name="aliases">Display names for placeholder tokens (e.g. GIVEN_NAME → "NAME" gives <c>[NAME_1]</c>).</param>
    /// <param name="keepLabels">Labels preserved in the text; <c>null</c> → CITY, STATE, ZIP_CODE.</param>
    public SessionEntityTable(IReadOnlyDictionary<PiiLabel, string>? aliases = null, IReadOnlySet<PiiLabel>? keepLabels = null)
    {
        _forwardLookup = _forward.GetAlternateLookup<ReadOnlySpan<char>>();
        _aliases = aliases ?? new Dictionary<PiiLabel, string>();
        _keepLabels = keepLabels ?? PiiLabels.DefaultKeepLabels;
    }

    private string DisplayName(PiiLabel label) => _aliases.TryGetValue(label, out var alias) ? alias : label.ToName();

    /// <summary>Get or mint the placeholder for a label + value. Idempotent; casing and whitespace are ignored.</summary>
    public string PlaceholderFor(PiiLabel label, string value)
    {
        // Key: LABEL ":" value lowercased (JavaScript toLowerCase), whitespace runs collapsed, trimmed —
        // built in a pooled buffer and looked up by span, so a known value allocates nothing.
        var key = new PooledList<char>(value.Length + 24);
        try
        {
            key.AddRange(label.ToName());
            key.Add(':');
            int prefix = key.Count;
            WordPieceTokenizer.JsLowerCaseInto(value, ref key);
            int write = prefix;
            bool inSpace = false;
            for (int i = prefix; i < key.Count; i++)
            {
                char c = key[i];
                if (JsChar.IsWhiteSpace(c))
                {
                    if (!inSpace) key[write++] = ' ';
                    inSpace = true;
                }
                else
                {
                    key[write++] = c;
                    inSpace = false;
                }
            }
            int from = prefix, to = write;
            while (from < to && JsChar.IsWhiteSpace(key[from])) from++;
            while (to > from && JsChar.IsWhiteSpace(key[to - 1])) to--;
            key.AsSpan()[from..to].CopyTo(key.AsSpan()[prefix..]);
            var keySpan = key.AsSpan()[..(prefix + to - from)];

            if (_forwardLookup.TryGetValue(keySpan, out var existing)) return existing;
            var name = DisplayName(label);
            int next = _counters.GetValueOrDefault(name) + 1;
            _counters[name] = next;
            var token = $"[{name}_{next}]";
            _forward[keySpan.ToString()] = token;
            _reverse[token] = value;
            return token;
        }
        finally
        {
            key.Dispose();
        }
    }

    /// <summary>Replace each redactable span (after merge and keep-set policy) with its placeholder.</summary>
    public ScrubResult Scrub(string raw, IEnumerable<PiiSpan> spans)
    {
        var input = new PooledList<PiiSpan>(16);
        try
        {
            foreach (var s in spans) input.Add(s);
            return Scrub(raw, input.AsSpan());
        }
        finally
        {
            input.Dispose();
        }
    }

    /// <inheritdoc cref="Scrub(string, IEnumerable{PiiSpan})"/>
    public ScrubResult Scrub(string raw, ReadOnlySpan<PiiSpan> spans)
    {
        var redactable = new PooledList<PiiSpan>(spans.Length);
        var text = new PooledList<char>(raw.Length + 32);
        try
        {
            Policy.ApplyPolicyInto(spans, _keepLabels, ref redactable); // right to left
            // Mint right to left, as the reference does (it numbers the last entity first), then
            // emit left to right.
            var tokens = new string[redactable.Count];
            for (int i = 0; i < redactable.Count; i++) tokens[i] = PlaceholderFor(redactable[i].Label, redactable[i].Text);

            int cursor = 0;
            for (int i = redactable.Count - 1; i >= 0; i--)
            {
                var span = redactable[i];
                text.AddRange(raw.AsSpan(cursor, span.Start - cursor));
                text.AddRange(tokens[i]);
                cursor = span.End;
            }
            text.AddRange(raw.AsSpan(cursor));
            Array.Reverse(tokens);
            return new ScrubResult(text.AsSpan().ToString(), tokens);
        }
        finally
        {
            redactable.Dispose();
            text.Dispose();
        }
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
