using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

namespace Rampart.Session;

/// <summary>
/// Streaming placeholder reveal. An assistant streams its reply, so a placeholder can be split
/// across chunks (<c>"…thanks [NA"</c> + <c>"ME_1] for…"</c>). This buffers the smallest suffix that
/// could still become a placeholder, emits everything safe before it, and flushes the remainder at
/// the end of the stream.
/// <para>Not thread-safe: one instance per stream.</para>
/// </summary>
internal sealed partial class StreamingReveal
{
    private readonly Func<string, string?> _resolve;
    private string _buffer = "";

    /// <param name="resolve">Resolves a placeholder to its real value, or <c>null</c> to leave it as-is.</param>
    public StreamingReveal(Func<string, string?> resolve)
    {
        _resolve = resolve;
    }

    // A run that might grow into a placeholder: '[' then [A-Z_] then optional '_' digits, unclosed.
    [GeneratedRegex(@"\[[A-Z_]*(?:_[0-9]*)?\z", RegexOptions.CultureInvariant)]
    private static partial Regex PartialToken();

    /// <summary>Reveal complete placeholders in <paramref name="chunk"/>, holding back any partial tail.</summary>
    public string Push(string chunk)
    {
        _buffer += chunk;
        var revealed = ReplaceComplete(_buffer);
        var partial = PartialToken().Match(revealed);
        if (!partial.Success)
        {
            _buffer = "";
            return revealed;
        }
        int cut = revealed.Length - partial.Length;
        _buffer = revealed[cut..];
        return revealed[..cut];
    }

    /// <summary>Emit any buffered tail (e.g. a lone <c>[</c> that never became a token).</summary>
    public string Flush()
    {
        var output = ReplaceComplete(_buffer);
        _buffer = "";
        return output;
    }

    private string ReplaceComplete(string text)
        => SessionEntityTable.PlaceholderPattern().Replace(text, m => _resolve(m.Value) ?? m.Value);

    /// <summary>Reveal placeholders in an asynchronous text stream (e.g. an LLM streaming reply).</summary>
    public static async IAsyncEnumerable<string> RevealAsync(IAsyncEnumerable<string> chunks, Func<string, string?> resolve, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var reveal = new StreamingReveal(resolve);
        await foreach (var chunk in chunks.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            var output = reveal.Push(chunk);
            if (output.Length > 0) yield return output;
        }
        var tail = reveal.Flush();
        if (tail.Length > 0) yield return tail;
    }
}
