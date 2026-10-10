using System.Security.Cryptography;
using System.Text;
using Rampart.Detection;

namespace Rampart;

/// <summary>How <see cref="Redactor"/> replaces a span.</summary>
public enum RedactionMode
{
    /// <summary>Replace with the label tag, e.g. <c>[GIVEN_NAME]</c>.</summary>
    Label,
    /// <summary>Replace every character with a mask character.</summary>
    Mask,
    /// <summary>Remove the span.</summary>
    Remove,
    /// <summary>Replace with the lowercase hex SHA-256 of the span text.</summary>
    Hash,
}

/// <summary>
/// One-way redaction for logs, analytics and traces, where nothing needs to be restored. For chat,
/// prefer <see cref="ChatGuard"/>, whose placeholders can be revealed again.
/// </summary>
public static class Redactor
{
    /// <summary>
    /// Redact every span that the default-deny policy says to redact (labels outside
    /// <paramref name="keepLabels"/>; default keeps CITY, STATE and ZIP_CODE).
    /// </summary>
    public static string Redact(string text, IEnumerable<PiiSpan> spans, RedactionMode mode = RedactionMode.Label, char maskChar = '*', IReadOnlySet<PiiLabel>? keepLabels = null)
    {
        var sb = new StringBuilder(text);
        foreach (var span in Policy.ApplyPolicy(spans, keepLabels)) // right to left
        {
            if (span.Start < 0 || span.End > text.Length || span.Start >= span.End) continue;
            var replacement = mode switch
            {
                RedactionMode.Label => Premask.SentinelFor(span.Label),
                RedactionMode.Mask => new string(maskChar, span.Length),
                RedactionMode.Remove => "",
                RedactionMode.Hash => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text[span.Start..span.End]))),
                _ => throw new ArgumentOutOfRangeException(nameof(mode)),
            };
            sb.Remove(span.Start, span.Length).Insert(span.Start, replacement);
        }
        return sb.ToString();
    }

    /// <summary>Detect with <paramref name="analyzer"/> and redact in one call.</summary>
    public static string Redact(this RampartAnalyzer analyzer, string text, RedactionMode mode = RedactionMode.Label, char maskChar = '*', IReadOnlySet<PiiLabel>? keepLabels = null)
        => Redact(text, analyzer.Detect(text), mode, maskChar, keepLabels);
}
