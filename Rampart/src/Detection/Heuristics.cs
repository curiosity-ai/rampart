using System.Text.RegularExpressions;

namespace Rampart.Detection;

/// <summary>
/// Heuristic detectors: the cheap, synchronous, zero-model first pass.
/// <para>
/// Digit-bearing PII (SSN, card) is found over digit runs, so every separator variant collapses to
/// one rule: <c>888-88-8888</c>, <c>888 88 8888</c>, <c>888.88.8888</c> and <c>888888888</c> all match.
/// Text-shaped PII (email, URL, IP/MAC) is matched on the raw string, where the structure lives in
/// the punctuation.
/// </para>
/// <para>
/// The expressions are the JavaScript ones, transcribed so they mean the same thing in .NET:
/// <c>\d</c>, <c>\w</c> and <c>\b</c> are ASCII in JavaScript and are spelled out as ASCII classes
/// here, and <c>\s</c> is spelled out as JavaScript's whitespace set.
/// </para>
/// </summary>
public static partial class Heuristics
{
    // ASCII word boundary, as JavaScript's \b (without the u/v flags).
    private const string W = "[A-Za-z0-9_]";
    private const string B = "(?:(?<=" + W + ")(?!" + W + ")|(?<!" + W + ")(?=" + W + "))";
    // JavaScript \s.
    private const string S = @"\t\n\v\f\r \u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\uFEFF";

    private const RegexOptions Options = RegexOptions.CultureInvariant;

    // A run is digits joined only by single inline separators (space, dash, dot).
    [GeneratedRegex(@"[0-9](?:[ .-]?[0-9])*", Options)]
    private static partial Regex DigitRun();

    // Email: local part (with +tags and dots) @ dotted domain.
    [GeneratedRegex(B + @"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" + B, Options)]
    private static partial Regex EmailRule();

    // URL with explicit scheme: everything up to whitespace / closing bracket / quote.
    [GeneratedRegex(B + @"https?://[^" + S + @"<>""'\])}]+", Options)]
    private static partial Regex SchemeUrlRule();

    // Schemeless web URL: `www.` host followed by the rest of the URL.
    [GeneratedRegex(B + @"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^" + S + @"<>""'\])}]*)?", Options)]
    private static partial Regex WwwUrlRule();

    [GeneratedRegex(B + @"(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])" + B, Options)]
    private static partial Regex Ipv4Rule();

    // IPv6: full and `::`-compressed forms; every alternative needs 8 groups or a `::`, so times
    // ("12:34") and MAC addresses never match, and the lookarounds keep it off larger tokens.
    [GeneratedRegex(
        @"(?<![:.A-Za-z0-9_])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,7}:|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4})(?![:.A-Za-z0-9_])",
        Options)]
    private static partial Regex Ipv6Rule();

    // MAC address: six hex pairs joined by ":" or "-".
    [GeneratedRegex(B + @"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}" + B, Options)]
    private static partial Regex MacRule();

    private static readonly (PiiLabel Label, Func<Regex> Pattern)[] TextRules =
    [
        (PiiLabel.Email, EmailRule),
        (PiiLabel.Url, SchemeUrlRule),
        (PiiLabel.Url, WwwUrlRule),
        (PiiLabel.IpAddress, Ipv4Rule),
        (PiiLabel.IpAddress, Ipv6Rule),
        (PiiLabel.IpAddress, MacRule),
    ];

    /// <summary>
    /// Run all heuristic detectors over the raw input. Spans may overlap; the merge step resolves
    /// conflicts before redaction.
    /// </summary>
    public static List<PiiSpan> Detect(string raw)
    {
        ArgumentNullException.ThrowIfNull(raw);
        var spans = new List<PiiSpan>();
        DetectDigitEntities(raw, spans);
        DetectTextEntities(raw, spans);
        return spans;
    }

    /// <summary>
    /// Scan each digit run for fixed-length, validator-backed entities. A candidate must cover the
    /// <em>whole</em> run, so a 9-digit SSN never fires inside a longer account number, and the raw
    /// span keeps the interior separators so <c>888-12-3456</c> is redacted whole. Longer/stricter
    /// entities are tried first so a 16-digit card is never carved into a 9-digit "SSN".
    /// </summary>
    private static void DetectDigitEntities(string raw, List<PiiSpan> spans)
    {
        Span<char> digits = stackalloc char[64];
        foreach (var m in DigitRun().EnumerateMatches(raw))
        {
            var run = raw.AsSpan(m.Index, m.Length);
            int count = 0, first = -1, last = -1;
            for (int i = 0; i < run.Length; i++)
            {
                if (!char.IsAsciiDigit(run[i])) continue;
                if (first < 0) first = i;
                last = i;
                if (count < digits.Length) digits[count] = run[i];
                count++;
            }
            if (count > 16) continue; // longer than every rule
            var d = digits[..count];

            PiiLabel? label = null;
            if (count is 16 or 15 or 14 && Validators.IsLuhnValid(d)) label = PiiLabel.CreditCard;
            else if (count == 9 && Validators.IsValidSsn(d)) label = PiiLabel.Ssn;
            if (label is not { } l) continue;

            int start = m.Index + first, end = m.Index + last + 1;
            spans.Add(new PiiSpan(start, end, l, 1, SpanSource.Heuristic, raw[start..end]));
        }
    }

    private static void DetectTextEntities(string raw, List<PiiSpan> spans)
    {
        foreach (var (label, pattern) in TextRules)
        {
            foreach (var m in pattern().EnumerateMatches(raw))
            {
                spans.Add(new PiiSpan(m.Index, m.Index + m.Length, label, 1, SpanSource.Heuristic, raw.Substring(m.Index, m.Length)));
            }
        }
    }
}
