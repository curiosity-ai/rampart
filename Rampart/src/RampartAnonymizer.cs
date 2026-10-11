using System.Security.Cryptography;
using System.Text;

namespace Catalyst.Rampart;

/// <summary>
/// One-way anonymization of analyzed text, like Catalyst.Presidio's <c>PresidioAnonymizer</c>: each
/// result is masked, replaced, removed or hashed. Nothing can be restored afterwards; to send text to
/// a service and restore its answer, use <see cref="RampartCloaker"/>.
/// </summary>
public sealed class RampartAnonymizer
{
    /// <summary>
    /// Applies <paramref name="operatorName"/> to every result in <paramref name="text"/>.
    /// <list type="bullet">
    /// <item><c>mask</c> (default): every character becomes the first character of <paramref name="maskChar"/>.</item>
    /// <item><c>replace</c>: the entity becomes <paramref name="replacement"/>.</item>
    /// <item><c>redact</c>: the entity is removed.</item>
    /// <item><c>hash</c>: the entity becomes the lowercase hex SHA-256 of its UTF-8 bytes.</item>
    /// </list>
    /// An unknown operator masks. Overlapping results keep the earliest one (then the higher score,
    /// then the longer), and results outside the text are skipped.
    /// </summary>
    public string Anonymize(string text, IEnumerable<RecognizerResult> results, string operatorName = "mask", string maskChar = "*", string replacement = "<REDACTED>")
    {
        if (string.IsNullOrEmpty(text) || results is null) return text;

        var sb = new StringBuilder(text);
        // Right to left, so earlier offsets stay valid.
        foreach (var result in RemoveOverlaps(results).OrderByDescending(r => r.Start))
        {
            if (result.Start < 0 || result.End >= text.Length || result.Start > result.End) continue;

            var newText = operatorName.ToLowerInvariant() switch
            {
                "replace" => replacement,
                "redact" => "",
                "hash" => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text.Substring(result.Start, result.Length)))),
                _ => new string(string.IsNullOrEmpty(maskChar) ? '*' : maskChar[0], result.Length),
            };
            sb.Remove(result.Start, result.Length).Insert(result.Start, newText);
        }
        return sb.ToString();
    }

    internal static List<RecognizerResult> RemoveOverlaps(IEnumerable<RecognizerResult> results)
    {
        var output = new List<RecognizerResult>();
        int lastEnd = -1;
        foreach (var result in results.OrderBy(r => r.Start).ThenByDescending(r => r.Score).ThenByDescending(r => r.Length))
        {
            if (result.Start > lastEnd)
            {
                output.Add(result);
                lastEnd = result.End;
            }
        }
        return output;
    }
}
