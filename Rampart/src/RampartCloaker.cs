using System.Runtime.CompilerServices;
using Rampart.Session;

namespace Rampart;

/// <summary>
/// Reversible cloaking for calls to an external service (an LLM, a translation API, …), like
/// Catalyst.Presidio's <c>PresidioCloaker</c>: every result is replaced with a typed placeholder
/// (<c>[GIVEN_NAME_1]</c>, <c>[SSN_1]</c>), the service only ever sees the placeholders, and its answer
/// comes back with the original values restored.
/// <code>
/// var text = "My name is Alex Rivera and my SSN is 472-81-0094.";
/// var results = analyzer.Analyze(text);
/// var answer = await cloaker.CloakAsync(text, results, cloaked => CallServiceAsync(cloaked));
/// </code>
/// <para>
/// The cloaker remembers every value it has cloaked, so the same value gets the same placeholder on
/// every call: use one cloaker per conversation. Not thread-safe.
/// </para>
/// </summary>
public sealed class RampartCloaker
{
    // Keep-labels were applied by the analyzer: every result handed to the cloaker is cloaked.
    private readonly SessionEntityTable _table = new(keepLabels: new HashSet<PiiLabel>());

    /// <summary>
    /// Replaces the <paramref name="results"/> in <paramref name="text"/> with placeholders, calls
    /// <paramref name="action"/> with the cloaked text, and returns its answer with the placeholders
    /// replaced by the original values.
    /// </summary>
    public async Task<string> CloakAsync(string text, IEnumerable<RecognizerResult> results, Func<string, Task<string>> action)
    {
        ArgumentNullException.ThrowIfNull(action);
        var response = await action(Cloak(text, results)).ConfigureAwait(false);
        return string.IsNullOrEmpty(response) ? response : _table.Rehydrate(response);
    }

    /// <summary>
    /// As <see cref="CloakAsync(string, IEnumerable{RecognizerResult}, Func{string, Task{string}})"/>
    /// for a service that streams its answer: chunks come back with the original values restored, also
    /// when a placeholder is split across chunks.
    /// </summary>
    public async IAsyncEnumerable<string> CloakAsync(string text, IEnumerable<RecognizerResult> results, Func<string, IAsyncEnumerable<string>> action, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(action);
        var chunks = action(Cloak(text, results));
        await foreach (var chunk in StreamingReveal.RevealAsync(chunks, _table.Resolve, cancellationToken).ConfigureAwait(false))
        {
            yield return chunk;
        }
    }

    /// <summary>The cloaked text: each result replaced by its placeholder (overlapping results are merged).</summary>
    internal string Cloak(string text, IEnumerable<RecognizerResult> results)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (results is null) return text;

        var spans = new List<PiiSpan>();
        foreach (var r in results)
        {
            if (!PiiLabels.TryParse(r.EntityType, out var label))
                throw new ArgumentException($"Unknown entity type '{r.EntityType}'; the cloaker takes the entity types RampartAnalyzer reports.", nameof(results));
            if (r.Start < 0 || r.End >= text.Length || r.Start > r.End)
                throw new ArgumentOutOfRangeException(nameof(results), $"Result [{r.Start}, {r.End}] is outside the text (length {text.Length}).");
            spans.Add(new PiiSpan(r.Start, r.End + 1, label, r.Score, SpanSource.Ner, text.Substring(r.Start, r.Length)));
        }
        return _table.Scrub(text, spans).Text;
    }
}
