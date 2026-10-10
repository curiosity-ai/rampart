using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Rampart.Detection;
using Rampart.Model;
using Rampart.Tokenization;

namespace Rampart.Ner;

/// <summary>A contextual PII detector: text in, spans over that text out.</summary>
public interface INerDetector
{
    IReadOnlyList<PiiSpan> Detect(string text);
}

/// <summary>
/// Contextual PII detection with the Rampart token classifier — the residual layer that catches
/// what the heuristics cannot: names, contact and document identifiers, address components.
/// <para>
/// A port of the reference <c>detectNer</c>: long input is scanned in word-aligned windows sized by
/// the tokenizer to <see cref="TokenBudget"/> tokens and overlapping by <see cref="TokenOverlap"/>;
/// per-token BIO labels are merged into spans whose offsets are recovered by walking the WordPiece
/// tokens through a folded (lowercase, NFKD, mark-stripped) projection of the input; and a repair pass
/// rescues low-confidence fragments that bridge to confident ones, merges fragments joined by name
/// connectors, and extends person names over short capitalized particles (<c>De La Croix</c>).
/// </para>
/// <para>Thread-safe: holds no per-call state.</para>
/// </summary>
public sealed partial class NerDetector : INerDetector
{
    /// <summary>The model's hard context window.</summary>
    public const int ModelMaxTokens = 512;
    /// <summary>Per-window content-token budget: the model max less [CLS]/[SEP] and a safety margin.</summary>
    public const int TokenBudget = ModelMaxTokens - 2 - 10;
    /// <summary>Tokens shared by consecutive windows, so an entity on a seam is wholly inside a neighbour.</summary>
    public const int TokenOverlap = 64;
    /// <summary>Default confidence floor for a span to be reported on its own.</summary>
    public const double DefaultMinScore = 0.4;
    /// <summary>Floor for a low-confidence fragment to be considered at all (it must bridge to a kept span).</summary>
    public const double ExtendScore = 0.15;

    private const int MaxRepairIterations = 32;

    private static readonly Dictionary<string, PiiLabel> GroupToLabel = new(StringComparer.Ordinal)
    {
        ["GIVEN_NAME"] = PiiLabel.GivenName,
        ["GIVENNAME"] = PiiLabel.GivenName,
        ["SURNAME"] = PiiLabel.Surname,
        ["LASTNAME"] = PiiLabel.Surname,
        ["EMAIL"] = PiiLabel.Email,
        ["PHONE"] = PiiLabel.Phone,
        ["URL"] = PiiLabel.Url,
        ["TAX_ID"] = PiiLabel.TaxId,
        ["BANK_ACCOUNT"] = PiiLabel.BankAccount,
        ["ROUTING_NUMBER"] = PiiLabel.RoutingNumber,
        ["GOVERNMENT_ID"] = PiiLabel.GovernmentId,
        ["PASSPORT"] = PiiLabel.Passport,
        ["DRIVERS_LICENSE"] = PiiLabel.DriversLicense,
        ["BUILDING_NUMBER"] = PiiLabel.BuildingNumber,
        ["STREET_NAME"] = PiiLabel.StreetName,
        ["SECONDARY_ADDRESS"] = PiiLabel.SecondaryAddress,
        ["SECADDRESS"] = PiiLabel.SecondaryAddress,
        ["CITY"] = PiiLabel.City,
        ["STATE"] = PiiLabel.State,
        ["ZIP_CODE"] = PiiLabel.ZipCode,
    };

    private readonly TokenClassifier _classifier;

    public double MinScore { get; }

    public NerDetector(RampartModel model, double minScore = DefaultMinScore) : this(new TokenClassifier(model), minScore) { }

    public NerDetector(TokenClassifier classifier, double minScore = DefaultMinScore)
    {
        _classifier = classifier;
        MinScore = minScore;
    }

    /// <summary>A detector over the embedded model.</summary>
    public static NerDetector CreateDefault(double minScore = DefaultMinScore) => new(RampartModel.Default, minScore);

    private WordPieceTokenizer Tokenizer => _classifier.Tokenizer;

    /// <summary>Detect contextual PII across the whole input, regardless of length.</summary>
    public IReadOnlyList<PiiSpan> Detect(string raw)
    {
        ArgumentNullException.ThrowIfNull(raw);
        var windows = PlanTokenWindows(raw, TokenBudget, TokenOverlap);
        if (windows.Count <= 1) return DetectWindow(raw);

        var spans = new List<PiiSpan>();
        foreach (var (start, end) in windows)
        {
            foreach (var span in DetectWindow(raw[start..end]))
            {
                spans.Add(span with { Start = span.Start + start, End = span.End + start });
            }
        }
        return Policy.MergeSpans(spans);
    }

    // ---------------------------------------------------------------- windows

    /// <summary>
    /// Plan the sliding windows over <paramref name="raw"/>: each holds at most
    /// <paramref name="budget"/> tokens and overlaps its predecessor by at least
    /// <paramref name="overlap"/> tokens. Windows snap to word boundaries; a single word longer than
    /// the budget is the only thing hard-split, by character.
    /// </summary>
    internal List<(int Start, int End)> PlanTokenWindows(string raw, int budget, int overlap)
    {
        var segments = ToSegments(raw, budget);
        var windows = new List<(int, int)>();
        if (segments.Count == 0) return windows;

        int i = 0;
        while (i < segments.Count)
        {
            int tokens = 0, j = i;
            while (j < segments.Count && (j == i || tokens + segments[j].Tokens <= budget))
            {
                tokens += segments[j].Tokens;
                j++;
            }
            windows.Add((segments[i].Start, segments[j - 1].End));
            if (j == segments.Count) break;

            int shared = 0, next = j;
            while (next > i + 1 && shared < overlap)
            {
                next--;
                shared += segments[next].Tokens;
            }
            i = next;
        }
        return windows;
    }

    private readonly record struct Segment(int Start, int End, int Tokens);

    private List<Segment> ToSegments(string raw, int budget)
    {
        var segments = new List<Segment>();
        foreach (var (start, end) in WordSpans(raw))
        {
            int from = start;
            while (from < end)
            {
                int tokens = Tokenizer.CountTokens(raw[from..end]);
                if (tokens <= budget)
                {
                    segments.Add(new Segment(from, end, tokens));
                    break;
                }
                int cut = FitCharsToBudget(raw, from, end, budget);
                segments.Add(new Segment(from, cut, Tokenizer.CountTokens(raw[from..cut])));
                from = cut;
            }
        }
        return segments;
    }

    /// <summary>Contiguous <c>[start, end)</c> spans of <paramref name="raw"/>, each a word plus its trailing whitespace.</summary>
    private static IEnumerable<(int, int)> WordSpans(string raw)
    {
        int n = raw.Length, i = 0;
        while (i < n)
        {
            int j = i;
            while (j < n && !JsChar.IsWhiteSpace(raw[j])) j++;
            while (j < n && JsChar.IsWhiteSpace(raw[j])) j++;
            yield return (i, j);
            i = j;
        }
    }

    private int FitCharsToBudget(string raw, int from, int end, int budget)
    {
        int lo = from + 1, hi = end, best = from + 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (Tokenizer.CountTokens(raw[from..mid]) <= budget)
            {
                best = mid;
                lo = mid + 1;
            }
            else hi = mid - 1;
        }
        return best;
    }

    // ---------------------------------------------------------------- one window

    /// <summary>
    /// Run the classifier over a single window and convert its output to spans over that window's
    /// text. Hyphens are shown to the model as spaces (a 1:1 substitution, so offsets carry over).
    /// </summary>
    internal List<PiiSpan> DetectWindow(string raw)
    {
        var inferText = raw.Replace('-', ' ');
        var tokens = Tokenizer.Tokenize(inferText);
        var ids = new int[tokens.Count + 2];
        ids[0] = Tokenizer.ClsId;
        for (int i = 0; i < tokens.Count; i++) ids[i + 1] = Tokenizer.TokenToId(tokens[i]);
        ids[^1] = Tokenizer.SepId;
        if (ids.Length > ModelMaxTokens)
            throw new InvalidOperationException($"NER window of {ids.Length} tokens exceeds the model maximum of {ModelMaxTokens}.");

        var entities = _classifier.Classify(ids);
        var folded = FoldForModel(inferText);
        var indexOffsets = BuildTokenIndexOffsets(folded, tokens);

        var candidates = new List<PiiSpan>();
        foreach (var entity in MergeBioTokens(entities, folded, indexOffsets))
        {
            if (!GroupToLabel.TryGetValue(entity.Group.ToUpperInvariant(), out var label)) continue;
            if (entity.Score < ExtendScore || entity.End <= entity.Start) continue;
            candidates.Add(new PiiSpan(entity.Start, entity.End, label, entity.Score, SpanSource.Ner, raw[entity.Start..entity.End]));
        }
        return RepairSpans(raw, candidates, MinScore);
    }

    /// <summary>A folded copy of the model input plus a map from each folded char to raw offsets.</summary>
    internal sealed record FoldedProjection(string Text, int[] RawStart, int[] RawEnd);

    /// <summary>
    /// Fold to the model's normalized space (lowercase + NFKD + strip combining marks), recording per
    /// folded character the <c>[start, end)</c> raw offsets of the code point it came from.
    /// </summary>
    internal static FoldedProjection FoldForModel(string raw)
    {
        var text = new StringBuilder(raw.Length);
        var rawStart = new List<int>(raw.Length);
        var rawEnd = new List<int>(raw.Length);
        int i = 0;
        while (i < raw.Length)
        {
            int width = char.IsHighSurrogate(raw[i]) && i + 1 < raw.Length && char.IsLowSurrogate(raw[i + 1]) ? 2 : 1;
            var codePoint = raw.Substring(i, width);
            var folded = WordPieceTokenizer.JsLowerCase(codePoint);
            if (!HasLoneSurrogate(folded) && !folded.IsNormalized(NormalizationForm.FormKD)) folded = folded.Normalize(NormalizationForm.FormKD);
            foreach (var ch in StripMarks(folded))
            {
                text.Append(ch);
                rawStart.Add(i);
                rawEnd.Add(i + width);
            }
            i += width;
        }
        return new FoldedProjection(text.ToString(), [.. rawStart], [.. rawEnd]);
    }

    private static bool HasLoneSurrogate(string s) => s.Length == 1 && char.IsSurrogate(s[0]);

    private static string StripMarks(string s)
    {
        if (s.Length == 1 && !char.IsSurrogate(s[0]))
        {
            return CharUnicodeInfo.GetUnicodeCategory(s[0]) is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark ? "" : s;
        }
        var sb = new StringBuilder(s.Length);
        int i = 0;
        while (i < s.Length)
        {
            if (Rune.TryGetRuneAt(s, i, out var rune))
            {
                if (Rune.GetUnicodeCategory(rune) is not (UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark)) sb.Append(rune.ToString());
                i += rune.Utf16SequenceLength;
            }
            else
            {
                sb.Append(s[i]); // a lone surrogate passes through, as in JavaScript
                i++;
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// Map each token index (<c>[CLS]</c> = 0, content, <c>[SEP]</c> last) to raw char offsets by walking
    /// the WordPiece tokens through the folded projection: a word-initial piece is searched forward
    /// from the cursor, a <c>##</c> piece must continue at the cursor. Unmatched tokens map to (0, 0).
    /// </summary>
    internal static (int Start, int End)[] BuildTokenIndexOffsets(FoldedProjection folded, IReadOnlyList<string> contentTokens)
    {
        var offsets = new (int, int)[contentTokens.Count + 2];
        int cursor = 0;
        for (int t = 0; t < contentTokens.Count; t++)
        {
            var token = contentTokens[t];
            bool continuation = token.StartsWith(WordPieceTokenizer.ContinuationPrefix, StringComparison.Ordinal);
            var piece = continuation ? token[2..] : token;
            int at = continuation ? cursor : IndexOfOrdinal(folded.Text, piece, cursor);
            if (piece.Length == 0 || at < 0 || at + piece.Length > folded.Text.Length)
            {
                offsets[t + 1] = (0, 0);
                continue;
            }
            offsets[t + 1] = (folded.RawStart[at], folded.RawEnd[at + piece.Length - 1]);
            cursor = at + piece.Length;
        }
        return offsets;
    }

    private static int IndexOfOrdinal(string text, string value, int from)
        => from > text.Length ? (value.Length == 0 ? text.Length : -1) : text.IndexOf(value, from, StringComparison.Ordinal);

    internal sealed record AggregatedEntity(string Group, double Score, int Start, int End);

    /// <summary>
    /// Merge raw BIO tokens into entities (B starts a span, a matching I extends it, and a <c>##</c>
    /// continuation always extends a span of the same base label even when labelled B-). Scores are
    /// averaged over the merged tokens.
    /// </summary>
    internal static List<AggregatedEntity> MergeBioTokens(IReadOnlyList<TokenEntity> entities, FoldedProjection folded, (int Start, int End)[]? indexOffsets)
    {
        var output = new List<AggregatedEntity>();
        int cursor = 0;
        string? group = null;
        double scoreSum = 0;
        int count = 0, curStart = 0, curEnd = 0;

        void Flush()
        {
            if (group is null) return;
            output.Add(new AggregatedEntity(group, scoreSum / count, curStart, curEnd));
            group = null;
        }

        foreach (var entity in entities)
        {
            var (prefix, baseLabel) = StripBio(entity.Entity);

            int start, end;
            if (indexOffsets is not null && (uint)entity.Index < (uint)indexOffsets.Length && indexOffsets[entity.Index] is var pair && pair.Start != pair.End)
            {
                (start, end) = pair;
            }
            else
            {
                var word = WordPieceTokenizer.JsLowerCase(entity.Word.StartsWith("##", StringComparison.Ordinal) ? entity.Word[2..] : entity.Word);
                if (word.Length == 0) continue;
                int at = IndexOfOrdinal(folded.Text, word, cursor);
                if (at < 0) continue;
                start = folded.RawStart[at];
                end = folded.RawEnd[at + word.Length - 1];
                cursor = at + word.Length;
            }

            bool isSubword = entity.Word.StartsWith("##", StringComparison.Ordinal);
            bool continues = group is not null && group == baseLabel && (prefix != 'B' || isSubword);
            if (continues)
            {
                curEnd = end;
                scoreSum += entity.Score;
                count++;
            }
            else
            {
                Flush();
                group = baseLabel;
                scoreSum = entity.Score;
                count = 1;
                curStart = start;
                curEnd = end;
            }
        }
        Flush();
        return output;
    }

    private static (char? Prefix, string Base) StripBio(string label)
        => label.Length > 2 && label[1] == '-' && label[0] is 'B' or 'I' ? (label[0], label[2..]) : (null, label);

    // ---------------------------------------------------------------- repair

    private static readonly IReadOnlySet<PiiLabel> PersonLabels = new HashSet<PiiLabel> { PiiLabel.GivenName, PiiLabel.Surname };

    internal static List<PiiSpan> RepairSpans(string raw, IReadOnlyList<PiiSpan> spans, double anchorScore)
    {
        var kept = spans.Where(s => s.Score >= anchorScore).ToList();
        var candidates = spans.Where(s => s.Score >= ExtendScore && s.Score < anchorScore).ToList();

        // Real text converges in O(max label length) iterations; the cap bounds rows with many
        // same-label single-token fragments that could otherwise keep flipping.
        int iterations = 0;
        bool changed = true;
        while (changed && iterations < MaxRepairIterations)
        {
            changed = false;
            iterations++;

            for (int i = candidates.Count - 1; i >= 0; i--)
            {
                var candidate = candidates[i];
                if (kept.Any(span => CanBridge(raw, candidate, span)))
                {
                    kept.Add(candidate);
                    candidates.RemoveAt(i);
                    changed = true;
                }
            }

            var merged = MergeAdjacentConnectors(raw, kept);
            bool didMerge = merged.Count != kept.Count;
            for (int i = 0; !didMerge && i < merged.Count; i++)
            {
                didMerge = merged[i].Start != kept[i].Start || merged[i].End != kept[i].End;
            }
            if (didMerge) changed = true;
            kept = merged;

            for (int i = 0; i < kept.Count; i++)
            {
                var repaired = RescueCapitalizedParticles(raw, kept[i], kept, i);
                if (repaired.Start != kept[i].Start || repaired.End != kept[i].End)
                {
                    kept[i] = repaired;
                    changed = true;
                }
            }
        }

        return kept
            .Select(s => s with { Text = raw[s.Start..s.End] })
            .OrderBy(s => s.Start).ThenBy(s => s.End)
            .ToList();
    }

    /// <summary>Connector characters: JavaScript whitespace, apostrophes, periods and hyphens.</summary>
    private static bool IsConnector(char c) => JsChar.IsWhiteSpace(c) || c is '\'' or '\u2019' or '.' or '-';

    private static bool IsAllConnectors(ReadOnlySpan<char> s)
    {
        foreach (var c in s) if (!IsConnector(c)) return false;
        return true;
    }

    /// <summary>
    /// True when the character at <paramref name="index"/> is a lone initial — a capital letter with a
    /// non-letter to its left (the "F" in "John F."), whose trailing dot is name-internal.
    /// </summary>
    private static bool IsInitialChar(string raw, int index)
    {
        if ((uint)index >= (uint)raw.Length || !char.IsUpper(raw[index])) return false;
        return index == 0 || !char.IsLetter(raw[index - 1]);
    }

    internal static bool CanBridge(string raw, PiiSpan a, PiiSpan b)
    {
        if (a.Label != b.Label) return false;
        var (left, right) = a.Start <= b.Start ? (a, b) : (b, a);
        var gap = right.Start > left.End ? raw.AsSpan(left.End, right.Start - left.End) : [];
        if (!IsAllConnectors(gap)) return false;
        // A period bridges fragments only across an initial ("J." + "R."); after a full word it is a
        // sentence boundary ("Garcia." + "I").
        if (gap.Contains('.') && !IsInitialChar(raw, left.End - 1)) return false;
        return true;
    }

    private static List<PiiSpan> MergeAdjacentConnectors(string raw, IReadOnlyList<PiiSpan> spans)
    {
        var merged = new List<PiiSpan>();
        foreach (var span in spans.OrderBy(s => s.Start).ThenBy(s => s.End))
        {
            if (merged.Count > 0 && CanBridge(raw, merged[^1], span))
            {
                var previous = merged[^1];
                int end = Math.Max(previous.End, span.End);
                merged[^1] = previous with { End = end, Score = Math.Max(previous.Score, span.Score), Text = raw[previous.Start..end] };
            }
            else merged.Add(span);
        }
        return merged;
    }

    // ([\p{Lu}][\p{L}\p{M}’']{0,3})([\s'’.-]{1,3})$ and ^([\s'’.-]{1,3})([\p{Lu}][\p{L}\p{M}’']{0,3}),
    // with JavaScript's \s. A match spans at most 7 code points, so a 16-char window is enough.
    private const string JsSpace = @"\t\n\v\f\r \u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\uFEFF";
    private const int ParticleWindow = 16;

    [GeneratedRegex(@"(\p{Lu}[\p{L}\p{M}\u2019']{0,3})([" + JsSpace + @"'\u2019.-]{1,3})\z", RegexOptions.CultureInvariant)]
    private static partial Regex LeftParticle();

    [GeneratedRegex(@"\A([" + JsSpace + @"'\u2019.-]{1,3})(\p{Lu}[\p{L}\p{M}\u2019']{0,3})", RegexOptions.CultureInvariant)]
    private static partial Regex RightParticle();

    internal static PiiSpan RescueCapitalizedParticles(string raw, PiiSpan span, IReadOnlyList<PiiSpan> all, int selfIndex)
    {
        if (!PersonLabels.Contains(span.Label)) return span;

        // Never extend into a region another kept span already claims (a SURNAME beside a GIVEN_NAME):
        // rescue is only for untagged particles.
        int leftBound = 0, rightBound = raw.Length;
        for (int i = 0; i < all.Count; i++)
        {
            if (i == selfIndex) continue;
            var other = all[i];
            if (other.End <= span.Start && other.End > leftBound) leftBound = other.End;
            if (other.Start >= span.End && other.Start < rightBound) rightBound = other.Start;
        }

        int start = span.Start, end = span.End;

        int leftFrom = Math.Max(0, start - ParticleWindow);
        var left = LeftParticle().Match(raw[leftFrom..start]);
        // Extend left across the connector; a period only behind a one-letter initial ("J."), never a word ("Dr.").
        if (left.Success
            && (!left.Groups[2].ValueSpan.Contains('.') || left.Groups[1].Length == 1)
            && start - left.Length >= leftBound)
        {
            start -= left.Length;
        }

        var right = RightParticle().Match(raw[end..Math.Min(raw.Length, end + ParticleWindow)]);
        // Never extend right across a period: it always crosses a sentence boundary ("Garcia. I").
        if (right.Success && !right.Groups[1].ValueSpan.Contains('.') && end + right.Length <= rightBound)
        {
            end += right.Length;
        }

        return start == span.Start && end == span.End ? span : span with { Start = start, End = end, Text = raw[start..end] };
    }
}
