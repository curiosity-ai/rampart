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
internal static partial class Heuristics
{
    // ASCII word boundary, as JavaScript's \b (without the u/v flags).
    private const string W = "[A-Za-z0-9_]";
    private const string B = "(?:(?<=" + W + ")(?!" + W + ")|(?<!" + W + ")(?=" + W + "))";
    // JavaScript \s.
    private const string S = @"\t\n\v\f\r \u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\uFEFF";

    private const RegexOptions Options = RegexOptions.CultureInvariant;

    // A run is digits joined only by single inline separators (space, dash, dot).
    [GeneratedRegex(@"[0-9](?:[ .-]?[0-9])*", Options)]
    internal static partial Regex DigitRun();

    // Email: local part (with +tags and dots) @ dotted domain.
    [GeneratedRegex(B + @"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" + B, Options)]
    internal static partial Regex EmailRule();

    // URL with explicit scheme: everything up to whitespace / closing bracket / quote.
    [GeneratedRegex(B + @"https?://[^" + S + @"<>""'\])}]+", Options)]
    internal static partial Regex SchemeUrlRule();

    // Schemeless web URL: `www.` host followed by the rest of the URL.
    [GeneratedRegex(B + @"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^" + S + @"<>""'\])}]*)?", Options)]
    internal static partial Regex WwwUrlRule();

    [GeneratedRegex(B + @"(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])" + B, Options)]
    internal static partial Regex Ipv4Rule();

    // IPv6: full and `::`-compressed forms; every alternative needs 8 groups or a `::`, so times
    // ("12:34") and MAC addresses never match, and the lookarounds keep it off larger tokens.
    [GeneratedRegex(
        @"(?<![:.A-Za-z0-9_])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,7}:|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4})(?![:.A-Za-z0-9_])",
        Options)]
    internal static partial Regex Ipv6Rule();

    // MAC address: six hex pairs joined by ":" or "-".
    [GeneratedRegex(B + @"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}" + B, Options)]
    internal static partial Regex MacRule();

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
        for (int at = 0; (at = NextDigitRun(raw, at, out int length)) >= 0; at += length)
        {
            var m = (Index: at, Length: length);
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
            if (pattern == Ipv4Rule)
            {
                for (int at = 0; (at = NextIpv4(raw, at, out int length)) >= 0; at += length)
                {
                    spans.Add(new PiiSpan(at, at + length, label, 1, SpanSource.Heuristic, raw.Substring(at, length)));
                }
                continue;
            }
            foreach (var m in pattern().EnumerateMatches(raw))
            {
                spans.Add(new PiiSpan(m.Index, m.Index + m.Length, label, 1, SpanSource.Heuristic, raw.Substring(m.Index, m.Length)));
            }
        }
    }

    // The two backtracking rules are scanned by hand: the generated matchers for them allocate a
    // backtracking stack per call. Both scanners reproduce the regex's leftmost, greedy, backtracking
    // result exactly (HeuristicsScannerTests fuzzes them against the expressions).

    private static bool IsDigit(char c) => (uint)(c - '0') <= 9;

    private static bool IsAsciiWord(char c) => IsDigit(c) || (uint)((c | 0x20) - 'a') <= 'z' - 'a' || c == '_';

    /// <summary>The next match of <c>[0-9](?:[ .-]?[0-9])*</c> at or after <paramref name="from"/>.</summary>
    internal static int NextDigitRun(string raw, int from, out int length)
    {
        int n = raw.Length, i = from;
        while (i < n && !IsDigit(raw[i])) i++;
        if (i >= n)
        {
            length = 0;
            return -1;
        }
        int j = i + 1;
        while (true)
        {
            if (j < n && IsDigit(raw[j])) j++;
            else if (j + 1 < n && raw[j] is ' ' or '.' or '-' && IsDigit(raw[j + 1])) j += 2;
            else break;
        }
        length = j - i;
        return i;
    }

    /// <summary>
    /// The next match of
    /// <c>\b(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\b</c>
    /// (ASCII <c>\b</c>) at or after <paramref name="from"/>.
    /// </summary>
    internal static int NextIpv4(string raw, int from, out int length)
    {
        for (int i = from; i < raw.Length; i++)
        {
            // \b before a digit: the previous character is not a word character.
            if (!IsDigit(raw[i]) || (i > 0 && IsAsciiWord(raw[i - 1]))) continue;
            int end = MatchOctets(raw, i, 0);
            if (end >= 0)
            {
                length = end - i;
                return i;
            }
        }
        length = 0;
        return -1;
    }

    /// <summary>Depth-first over the octet alternatives in the regex's order; returns the match end or -1.</summary>
    private static int MatchOctets(string raw, int at, int octet)
    {
        Span<int> lengths = stackalloc int[6];
        int count = OctetLengths(raw, at, lengths);
        for (int k = 0; k < count; k++)
        {
            int end = at + lengths[k];
            if (octet < 3)
            {
                if (end < raw.Length && raw[end] == '.')
                {
                    int result = MatchOctets(raw, end + 1, octet + 1);
                    if (result >= 0) return result;
                }
            }
            else if (end >= raw.Length || !IsAsciiWord(raw[end]))
            {
                return end; // \b after the last digit
            }
        }
        return -1;
    }

    /// <summary>Lengths an octet can match at <paramref name="at"/>, in the order the regex tries them.</summary>
    private static int OctetLengths(string raw, int at, Span<int> lengths)
    {
        int n = 0;
        char c0 = at < raw.Length ? raw[at] : '\0', c1 = at + 1 < raw.Length ? raw[at + 1] : '\0', c2 = at + 2 < raw.Length ? raw[at + 2] : '\0';
        if (c0 == '2' && c1 == '5' && c2 is >= '0' and <= '5') lengths[n++] = 3;           // 25[0-5]
        if (c0 == '2' && c1 is >= '0' and <= '4' && IsDigit(c2)) lengths[n++] = 3;         // 2[0-4][0-9]
        // 1?[0-9]?[0-9]: greedy, so with the '1' first, then without; within each, \d? taken first.
        if (c0 == '1')
        {
            if (IsDigit(c1) && IsDigit(c2)) lengths[n++] = 3;
            if (IsDigit(c1)) lengths[n++] = 2;
        }
        if (IsDigit(c0) && IsDigit(c1)) lengths[n++] = 2;
        if (IsDigit(c0)) lengths[n++] = 1;
        return n;
    }
}
