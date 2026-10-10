using System.Collections.Frozen;
using System.Globalization;
using System.Text;
using Rampart.Buffers;
using Rampart.Detection;
using Rampart.Model;
using Rampart.Tokenization;

namespace Rampart.Ner;

/// <summary>A contextual PII detector: text in, spans over that text out.</summary>
internal interface INerDetector
{
    IReadOnlyList<PiiSpan> Detect(string text);
}

/// <summary>An <see cref="INerDetector"/> backed by a delegate — for another detector, or a test double.</summary>
internal sealed class DelegateNerDetector(Func<string, IReadOnlyList<PiiSpan>> detect) : INerDetector
{
    public IReadOnlyList<PiiSpan> Detect(string text) => detect(text);
}

/// <summary>One classifier row, as the merge step sees it (see <see cref="TokenEntity"/> for the shapes).</summary>
internal readonly record struct NerRow(string? Entity, string? Group, double Score, int Index, string Word, int Start, int End, bool HasOffsets);

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
/// <para>
/// With the built-in <see cref="TokenClassifier"/> a window is tokenized once and every intermediate
/// (ids, logits, folded text, offsets, rows, candidates) lives in pooled buffers: the only allocations
/// are the returned spans and their text.
/// </para>
/// <para>Thread-safe: holds no per-call state.</para>
/// </summary>
internal sealed class NerDetector : INerDetector
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

    private static readonly FrozenDictionary<string, PiiLabel> GroupToLabel = new Dictionary<string, PiiLabel>(StringComparer.Ordinal)
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
    }.ToFrozenDictionary(StringComparer.Ordinal);

    private static readonly FrozenDictionary<string, PiiLabel>.AlternateLookup<ReadOnlySpan<char>> GroupLookup = GroupToLabel.GetAlternateLookup<ReadOnlySpan<char>>();

    private readonly ITokenClassifier _classifier;
    private readonly TokenClassifier? _builtIn;

    public double MinScore { get; }

    public NerDetector(RampartModel model, double minScore = DefaultMinScore) : this(new TokenClassifier(model), minScore) { }

    /// <summary>A detector over any classifier honouring the reference's contract (another model, a test double).</summary>
    public NerDetector(ITokenClassifier classifier, double minScore = DefaultMinScore)
    {
        _classifier = classifier;
        _builtIn = classifier as TokenClassifier;
        MinScore = minScore;
    }

    /// <summary>A detector over the embedded model.</summary>
    public static NerDetector CreateDefault(double minScore = DefaultMinScore) => new(RampartModel.Default, minScore);

    /// <summary>Detect contextual PII across the whole input, regardless of length.</summary>
    public IReadOnlyList<PiiSpan> Detect(string raw)
    {
        ArgumentNullException.ThrowIfNull(raw);
        var spans = new PooledList<NerSpan>(16);
        try
        {
            DetectCore(raw, ref spans);
            var output = new List<PiiSpan>(spans.Count);
            foreach (var s in spans.AsSpan())
            {
                output.Add(new PiiSpan(s.Start, s.End, s.Label, s.Score, SpanSource.Ner, raw.Substring(s.TextStart, s.TextEnd - s.TextStart)));
            }
            return output;
        }
        finally
        {
            spans.Dispose();
        }
    }

    /// <summary>A detected span without its text: <see cref="TextStart"/>/<see cref="TextEnd"/> is the range its text is sliced from.</summary>
    internal readonly record struct NerSpan(int Start, int End, PiiLabel Label, double Score, int TextStart, int TextEnd);

    /// <summary>The allocation-free core of <see cref="Detect"/>: spans over <paramref name="raw"/>, appended to <paramref name="output"/>.</summary>
    internal void DetectCore(ReadOnlySpan<char> raw, ref PooledList<NerSpan> output)
    {
        // Input that fits one window is classified whole — the planner would produce that one window.
        // (For the built-in tokenizer the sum of per-word counts equals the whole-text count.)
        if (_builtIn is not null ? _builtIn.Tokenizer.CountTokens(raw) <= TokenBudget : _classifier.CountTokens is null)
        {
            DetectWindow(raw, 0, raw.Length, ref output);
            return;
        }

        var windows = new PooledList<(int Start, int End)>(8);
        var all = new PooledList<NerSpan>(32);
        try
        {
            if (_builtIn is not null) PlanTokenWindows(raw, new TokenizerCounter(_builtIn.Tokenizer), TokenBudget, TokenOverlap, ref windows);
            else PlanTokenWindows(raw, new DelegateCounter(_classifier.CountTokens!), TokenBudget, TokenOverlap, ref windows);

            if (windows.Count <= 1)
            {
                DetectWindow(raw, 0, raw.Length, ref output);
                return;
            }
            foreach (var (start, end) in windows.AsSpan()) DetectWindow(raw, start, end - start, ref all);
            MergeSpans(ref all, ref output);
        }
        finally
        {
            windows.Dispose();
            all.Dispose();
        }
    }

    /// <summary>
    /// <see cref="Policy.MergeSpans"/> over model spans (all from one source, so a full tie goes to the
    /// later span), keeping the winner's text range on a partial overlap as the reference does.
    /// </summary>
    private static void MergeSpans(ref PooledList<NerSpan> spans, ref PooledList<NerSpan> output)
    {
        var sorted = spans.AsSpan();
        // Stable insertion sort by start ascending, end descending.
        for (int i = 1; i < sorted.Length; i++)
        {
            var item = sorted[i];
            int j = i - 1;
            while (j >= 0 && (sorted[j].Start > item.Start || (sorted[j].Start == item.Start && sorted[j].End < item.End)))
            {
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = item;
        }

        int first = output.Count;
        foreach (var span in sorted)
        {
            if (output.Count == first || span.Start >= output[output.Count - 1].End)
            {
                output.Add(span);
                continue;
            }
            ref var prev = ref output[output.Count - 1];
            var winner = prev.Score != span.Score ? (prev.Score > span.Score ? prev : span)
                : (prev.End - prev.Start) != (span.End - span.Start) ? ((prev.End - prev.Start) > (span.End - span.Start) ? prev : span)
                : span;
            bool prevContains = prev.Start <= span.Start && prev.End >= span.End;
            bool spanContains = span.Start <= prev.Start && span.End >= prev.End;
            prev = prevContains || spanContains ? winner : winner with { Start = Math.Min(prev.Start, span.Start), End = Math.Max(prev.End, span.End) };
        }
    }

    // ---------------------------------------------------------------- windows

    internal interface ITokenCounter
    {
        int Count(ReadOnlySpan<char> text);
    }

    private readonly struct TokenizerCounter(WordPieceTokenizer tokenizer) : ITokenCounter
    {
        public int Count(ReadOnlySpan<char> text) => tokenizer.CountTokens(text);
    }

    private readonly struct DelegateCounter(Func<string, int> count) : ITokenCounter
    {
        public int Count(ReadOnlySpan<char> text) => count(text.ToString());
    }

    /// <summary>
    /// Plan the sliding windows over <paramref name="raw"/>: each holds at most
    /// <paramref name="budget"/> tokens and overlaps its predecessor by at least
    /// <paramref name="overlap"/> tokens. Windows snap to word boundaries (a word plus its trailing
    /// whitespace); a single word longer than the budget is the only thing hard-split, by character.
    /// </summary>
    internal static void PlanTokenWindows<TCounter>(ReadOnlySpan<char> raw, TCounter counter, int budget, int overlap, ref PooledList<(int Start, int End)> windows)
        where TCounter : struct, ITokenCounter
    {
        var segments = new PooledList<(int Start, int End, int Tokens)>(64);
        try
        {
            int n = raw.Length, w = 0;
            while (w < n)
            {
                int end = w;
                while (end < n && !JsChar.IsWhiteSpace(raw[end])) end++;
                while (end < n && JsChar.IsWhiteSpace(raw[end])) end++;
                int from = w;
                while (from < end)
                {
                    int tokens = counter.Count(raw[from..end]);
                    if (tokens <= budget)
                    {
                        segments.Add((from, end, tokens));
                        break;
                    }
                    int cut = FitCharsToBudget(raw, from, end, budget, counter);
                    segments.Add((from, cut, counter.Count(raw[from..cut])));
                    from = cut;
                }
                w = end;
            }

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
        }
        finally
        {
            segments.Dispose();
        }
    }

    private static int FitCharsToBudget<TCounter>(ReadOnlySpan<char> raw, int from, int end, int budget, TCounter counter) where TCounter : struct, ITokenCounter
    {
        int lo = from + 1, hi = end, best = from + 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (counter.Count(raw[from..mid]) <= budget)
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
    /// Run the classifier over the window <c>raw[offset .. offset + length)</c> and append its spans,
    /// in whole-text coordinates, to <paramref name="output"/>. Hyphens are shown to the model as spaces
    /// (a 1:1 substitution, so offsets carry over); all repair logic sees only the window's text.
    /// </summary>
    private void DetectWindow(ReadOnlySpan<char> raw, int offset, int length, ref PooledList<NerSpan> output)
    {
        var window = raw.Slice(offset, length);
        var infer = new PooledList<char>(length);
        var folded = new FoldedProjection(length);
        var rows = new PooledList<NerRow>(16);
        var offsets = new PooledList<(int Start, int End)>(0);
        var merged = new PooledList<AggregatedEntity>(16);
        var candidates = new PooledList<Candidate>(16);
        try
        {
            infer.AddRange(window);
            infer.AsSpan().Replace('-', ' ');
            var inferText = infer.AsSpan();
            folded.Fold(inferText);

            bool haveOffsets;
            if (_builtIn is not null)
            {
                _builtIn.ClassifyWindow(inferText, ref rows, out var tokens);
                try
                {
                    BuildTokenIndexOffsets(ref folded, _builtIn.Tokenizer, tokens.AsSpan()[1..^1], ref offsets);
                }
                finally
                {
                    tokens.Dispose();
                }
                haveOffsets = true;
            }
            else
            {
                var text = inferText.ToString();
                foreach (var e in _classifier.Classify(text))
                {
                    bool hasOffsets = e.Start is not null && e.End is not null;
                    rows.Add(new NerRow(e.Entity, e.EntityGroup, e.Score, e.Index ?? -1, e.Word, e.Start ?? 0, e.End ?? 0, hasOffsets));
                }
                var tokenize = _classifier.Tokenize;
                haveOffsets = tokenize is not null;
                if (haveOffsets) BuildTokenIndexOffsets(ref folded, tokenize!(text), ref offsets);
            }

            MergeBioTokens(rows.AsSpan(), ref folded, haveOffsets ? offsets.AsSpan() : default, haveOffsets, ref merged);

            Span<char> upper = stackalloc char[64];
            foreach (var entity in merged.AsSpan())
            {
                var group = entity.Group;
                if (group.Length > upper.Length) continue;
                int written = group.ToUpperInvariant(upper);
                if (!GroupLookup.TryGetValue(upper[..written], out var label)) continue;
                if (entity.Score < ExtendScore || entity.End <= entity.Start) continue;
                candidates.Add(new Candidate(entity.Start, entity.End, label, entity.Score));
            }

            RepairSpans(window, ref candidates, MinScore);
            foreach (var c in candidates.AsSpan())
            {
                output.Add(new NerSpan(offset + c.Start, offset + c.End, c.Label, c.Score, offset + c.Start, offset + c.End));
            }
        }
        finally
        {
            infer.Dispose();
            folded.Dispose();
            rows.Dispose();
            offsets.Dispose();
            merged.Dispose();
            candidates.Dispose();
        }
    }

    // ---------------------------------------------------------------- folding and offsets

    /// <summary>
    /// A folded copy of the model input (lowercase, NFKD, combining marks stripped — the fold BERT's
    /// BasicTokenizer applies) plus, per folded character, the <c>[start, end)</c> offsets of the input
    /// code point it came from.
    /// </summary>
    internal struct FoldedProjection(int capacity) : IDisposable
    {
        public PooledList<char> Text = new(capacity + 16);
        public PooledList<int> RawStart = new(capacity + 16);
        public PooledList<int> RawEnd = new(capacity + 16);

        /// <summary>
        /// Folds code point by code point, as the reference does. Like it, an astral code point records
        /// one offset pair however many UTF-16 units its fold has — except that here every folded unit
        /// gets its pair, so offsets after an emoji do not drift (the documented divergence).
        /// </summary>
        public void Fold(ReadOnlySpan<char> raw)
        {
            int i = 0;
            while (i < raw.Length)
            {
                char c = raw[i];
                if (c < 0x80)
                {
                    Text.Add(c is >= 'A' and <= 'Z' ? (char)(c + 32) : c);
                    RawStart.Add(i);
                    RawEnd.Add(i + 1);
                    i++;
                    continue;
                }
                int width = char.IsHighSurrogate(c) && i + 1 < raw.Length && char.IsLowSurrogate(raw[i + 1]) ? 2 : 1;
                ReadOnlySpan<char> folded = width == 1 && char.IsSurrogate(c) ? raw.Slice(i, 1) : FoldCodePoint(raw.Slice(i, width));
                foreach (var ch in folded)
                {
                    Text.Add(ch);
                    RawStart.Add(i);
                    RawEnd.Add(i + width);
                }
                i += width;
            }
        }

        public void Dispose()
        {
            Text.Dispose();
            RawStart.Dispose();
            RawEnd.Dispose();
        }
    }

    private static readonly string?[] BmpModelFolds = new string?[0x10000];
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, string> AstralModelFolds = new();

    /// <summary><c>codePoint.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "")</c>, cached per code point.</summary>
    private static ReadOnlySpan<char> FoldCodePoint(ReadOnlySpan<char> codePoint)
    {
        if (codePoint.Length == 1) return BmpModelFolds[codePoint[0]] ??= ComputeModelFold(codePoint.ToString());
        Rune.DecodeFromUtf16(codePoint, out var rune, out _);
        return AstralModelFolds.GetOrAdd(rune.Value, static cp => ComputeModelFold(char.ConvertFromUtf32(cp)));
    }

    private static string ComputeModelFold(string codePoint)
    {
        var folded = WordPieceTokenizer.JsLowerCase(codePoint);
        if (!folded.IsNormalized(NormalizationForm.FormKD)) folded = folded.Normalize(NormalizationForm.FormKD);
        var sb = new StringBuilder(folded.Length);
        foreach (var rune in folded.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) is not (UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark)) sb.Append(rune.ToString());
        }
        return sb.ToString();
    }

    /// <summary>
    /// Map each token index (<c>[CLS]</c> = 0, content, <c>[SEP]</c> last) to input offsets by walking the
    /// WordPiece tokens through the folded projection: a word-initial piece is searched forward from the
    /// cursor, a <c>##</c> piece must continue at the cursor. Unmatched tokens map to (0, 0).
    /// </summary>
    private static void BuildTokenIndexOffsets(ref FoldedProjection folded, WordPieceTokenizer tokenizer, ReadOnlySpan<int> contentIds, ref PooledList<(int, int)> offsets)
    {
        offsets.Add((0, 0));
        int cursor = 0;
        foreach (var id in contentIds) cursor = AddTokenOffset(ref folded, tokenizer.IdToToken(id), cursor, ref offsets);
        offsets.Add((0, 0));
    }

    private static void BuildTokenIndexOffsets(ref FoldedProjection folded, IReadOnlyList<string> contentTokens, ref PooledList<(int, int)> offsets)
    {
        offsets.Add((0, 0));
        int cursor = 0;
        foreach (var token in contentTokens) cursor = AddTokenOffset(ref folded, token, cursor, ref offsets);
        offsets.Add((0, 0));
    }

    private static int AddTokenOffset(ref FoldedProjection folded, string token, int cursor, ref PooledList<(int, int)> offsets)
    {
        var text = folded.Text.AsSpan();
        bool continuation = token.StartsWith(WordPieceTokenizer.ContinuationPrefix, StringComparison.Ordinal);
        var piece = continuation ? token.AsSpan(2) : token.AsSpan();
        int at = continuation ? cursor : IndexOf(text, piece, cursor);
        if (piece.Length == 0 || at < 0 || at + piece.Length > text.Length)
        {
            offsets.Add((0, 0));
            return cursor;
        }
        offsets.Add((folded.RawStart[at], folded.RawEnd[at + piece.Length - 1]));
        return at + piece.Length;
    }

    /// <summary><c>String.prototype.indexOf(value, from)</c> over a span.</summary>
    private static int IndexOf(ReadOnlySpan<char> text, ReadOnlySpan<char> value, int from)
    {
        if (from > text.Length) return value.IsEmpty ? text.Length : -1;
        int at = text[from..].IndexOf(value, StringComparison.Ordinal);
        return at < 0 ? -1 : at + from;
    }

    internal readonly record struct AggregatedEntity(string Source, int Offset, double Score, int Start, int End)
    {
        public ReadOnlySpan<char> Group => Source.AsSpan(Offset);
    }

    /// <summary>
    /// Normalize either row shape into input-offset entities. Aggregated rows are used as they are. Raw
    /// BIO tokens are merged (B starts a span, a matching I extends it, and a <c>##</c> continuation
    /// always extends a span of the same base label even when labelled B-); offsets come from the token
    /// index, or failing that from a forward search for the folded word. Scores are averaged.
    /// </summary>
    private static void MergeBioTokens(ReadOnlySpan<NerRow> rows, ref FoldedProjection folded, ReadOnlySpan<(int Start, int End)> indexOffsets, bool haveOffsets, ref PooledList<AggregatedEntity> output)
    {
        int cursor = 0;
        string? source = null;
        int baseOffset = 0, count = 0, curStart = 0, curEnd = 0;
        double scoreSum = 0;

        foreach (var row in rows)
        {
            if (row.Group is not null && row.HasOffsets)
            {
                Flush(ref output);
                output.Add(new AggregatedEntity(row.Group, 0, row.Score, row.Start, row.End));
                continue;
            }
            var rawLabel = row.Entity ?? row.Group;
            if (rawLabel is null) continue;
            bool prefixed = rawLabel.Length > 2 && rawLabel[1] == '-' && rawLabel[0] is 'B' or 'I';
            char prefix = prefixed ? rawLabel[0] : '\0';
            int labelOffset = prefixed ? 2 : 0;

            int start, end;
            if (haveOffsets && row.Index >= 0 && row.Index < indexOffsets.Length && indexOffsets[row.Index] is var pair && pair.Start != pair.End)
            {
                (start, end) = pair;
            }
            else
            {
                var word = WordPieceTokenizer.JsLowerCase(row.Word.StartsWith("##", StringComparison.Ordinal) ? row.Word[2..] : row.Word);
                if (word.Length == 0) continue;
                int at = IndexOf(folded.Text.AsSpan(), word, cursor);
                if (at < 0) continue;
                start = folded.RawStart[at];
                end = folded.RawEnd[at + word.Length - 1];
                cursor = at + word.Length;
            }

            bool isSubword = row.Word.StartsWith("##", StringComparison.Ordinal);
            bool continues = source is not null
                && source.AsSpan(baseOffset).SequenceEqual(rawLabel.AsSpan(labelOffset))
                && (prefix != 'B' || isSubword);
            if (continues)
            {
                curEnd = end;
                scoreSum += row.Score;
                count++;
            }
            else
            {
                Flush(ref output);
                source = rawLabel;
                baseOffset = labelOffset;
                scoreSum = row.Score;
                count = 1;
                curStart = start;
                curEnd = end;
            }
        }
        Flush(ref output);

        void Flush(ref PooledList<AggregatedEntity> list)
        {
            if (source is null) return;
            list.Add(new AggregatedEntity(source, baseOffset, scoreSum / count, curStart, curEnd));
            source = null;
        }
    }

    // ---------------------------------------------------------------- repair

    internal readonly record struct Candidate(int Start, int End, PiiLabel Label, double Score);

    private static bool IsPerson(PiiLabel label) => label is PiiLabel.GivenName or PiiLabel.Surname;

    /// <summary>
    /// Keep spans at or above <paramref name="anchorScore"/>; let weaker fragments (≥ <see cref="ExtendScore"/>)
    /// in only where they bridge to a kept span; merge spans joined by name connectors; extend names over
    /// short capitalized particles — repeated until nothing changes. Leaves the result in
    /// <paramref name="spans"/>, sorted by (start, end).
    /// </summary>
    private static void RepairSpans(ReadOnlySpan<char> raw, ref PooledList<Candidate> spans, double anchorScore)
    {
        var kept = new PooledList<Candidate>(spans.Count);
        var weak = new PooledList<Candidate>(spans.Count);
        var merged = new PooledList<Candidate>(spans.Count);
        try
        {
            foreach (var s in spans.AsSpan())
            {
                if (s.Score >= anchorScore) kept.Add(s);
                else if (s.Score >= ExtendScore) weak.Add(s);
            }

            // Real text converges in O(max label length) iterations; the cap bounds rows with many
            // same-label single-token fragments that could otherwise keep flipping.
            int iterations = 0;
            bool changed = true;
            while (changed && iterations < MaxRepairIterations)
            {
                changed = false;
                iterations++;

                for (int i = weak.Count - 1; i >= 0; i--)
                {
                    var candidate = weak[i];
                    foreach (var k in kept.AsSpan())
                    {
                        if (!CanBridge(raw, candidate, k)) continue;
                        kept.Add(candidate);
                        weak.RemoveAt(i);
                        changed = true;
                        break;
                    }
                }

                MergeAdjacentConnectors(raw, kept.AsSpan(), ref merged);
                bool didMerge = merged.Count != kept.Count;
                for (int i = 0; !didMerge && i < merged.Count; i++)
                {
                    didMerge = merged[i].Start != kept[i].Start || merged[i].End != kept[i].End;
                }
                if (didMerge) changed = true;
                kept.Clear();
                kept.AddRange(merged.AsSpan());

                for (int i = 0; i < kept.Count; i++)
                {
                    var repaired = RescueCapitalizedParticles(raw, kept[i], kept.AsSpan(), i);
                    if (repaired.Start != kept[i].Start || repaired.End != kept[i].End)
                    {
                        kept[i] = repaired;
                        changed = true;
                    }
                }
            }

            SortByStartEnd(kept.AsSpan());
            spans.Clear();
            spans.AddRange(kept.AsSpan());
        }
        finally
        {
            kept.Dispose();
            weak.Dispose();
            merged.Dispose();
        }
    }

    /// <summary>Stable sort by (start, end) — insertion sort, as the lists are a handful of spans.</summary>
    private static void SortByStartEnd(Span<Candidate> spans)
    {
        for (int i = 1; i < spans.Length; i++)
        {
            var item = spans[i];
            int j = i - 1;
            while (j >= 0 && (spans[j].Start > item.Start || (spans[j].Start == item.Start && spans[j].End > item.End)))
            {
                spans[j + 1] = spans[j];
                j--;
            }
            spans[j + 1] = item;
        }
    }

    /// <summary>Connector characters: JavaScript whitespace, apostrophes, periods and hyphens.</summary>
    private static bool IsConnector(char c) => JsChar.IsWhiteSpace(c) || c is '\'' or '’' or '.' or '-';

    /// <summary><c>[\p{L}\p{M}’']</c>.</summary>
    private static bool IsNameChar(char c) => c is '\'' or '’' || CharUnicodeInfo.GetUnicodeCategory(c) is
        UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter
        or UnicodeCategory.OtherLetter or UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;

    private static bool IsUpper(char c) => CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.UppercaseLetter;

    /// <summary>
    /// True when the character at <paramref name="index"/> is a lone initial — a capital letter with a
    /// non-letter to its left (the "F" in "John F."), whose trailing dot is name-internal.
    /// </summary>
    private static bool IsInitialChar(ReadOnlySpan<char> raw, int index)
    {
        if ((uint)index >= (uint)raw.Length || !char.IsUpper(raw[index])) return false;
        return index == 0 || !char.IsLetter(raw[index - 1]);
    }

    private static bool CanBridge(ReadOnlySpan<char> raw, Candidate a, Candidate b)
    {
        if (a.Label != b.Label) return false;
        var (left, right) = a.Start <= b.Start ? (a, b) : (b, a);
        var gap = right.Start > left.End ? raw[left.End..right.Start] : [];
        foreach (var c in gap) if (!IsConnector(c)) return false;
        // A period bridges fragments only across an initial ("J." + "R."); after a full word it is a
        // sentence boundary ("Garcia." + "I").
        if (gap.Contains('.') && !IsInitialChar(raw, left.End - 1)) return false;
        return true;
    }

    private static void MergeAdjacentConnectors(ReadOnlySpan<char> raw, ReadOnlySpan<Candidate> spans, ref PooledList<Candidate> merged)
    {
        merged.Clear();
        var sorted = new PooledList<Candidate>(spans.Length);
        try
        {
            sorted.AddRange(spans);
            SortByStartEnd(sorted.AsSpan());
            foreach (var span in sorted.AsSpan())
            {
                if (merged.Count > 0 && CanBridge(raw, merged[merged.Count - 1], span))
                {
                    ref var previous = ref merged[merged.Count - 1];
                    previous = previous with { End = Math.Max(previous.End, span.End), Score = Math.Max(previous.Score, span.Score) };
                }
                else merged.Add(span);
            }
        }
        finally
        {
            sorted.Dispose();
        }
    }

    /// <summary>
    /// Extend a person span over an untagged short capitalized particle on either side, within the
    /// neighbouring spans' bounds — the reference's
    /// <c>([\p{Lu}][\p{L}\p{M}’']{0,3})([\s'’.-]{1,3})$</c> before the span and
    /// <c>^([\s'’.-]{1,3})([\p{Lu}][\p{L}\p{M}’']{0,3})</c> after it, matched by hand with the same
    /// leftmost-then-greedy semantics.
    /// </summary>
    private static Candidate RescueCapitalizedParticles(ReadOnlySpan<char> raw, Candidate span, ReadOnlySpan<Candidate> all, int selfIndex)
    {
        if (!IsPerson(span.Label)) return span;

        int leftBound = 0, rightBound = raw.Length;
        for (int i = 0; i < all.Length; i++)
        {
            if (i == selfIndex) continue;
            var other = all[i];
            if (other.End <= span.Start && other.End > leftBound) leftBound = other.End;
            if (other.Start >= span.End && other.Start < rightBound) rightBound = other.Start;
        }

        int start = span.Start, end = span.End;

        // Extend left across the connector; a period only behind a one-letter initial ("J."), never a word ("Dr.").
        if (MatchLeftParticle(raw[..start], out int leftLength, out int nameLength, out bool connectorHasPeriod)
            && (!connectorHasPeriod || nameLength == 1)
            && start - leftLength >= leftBound)
        {
            start -= leftLength;
        }

        // Never extend right across a period: it always crosses a sentence boundary ("Garcia. I").
        if (MatchRightParticle(raw[end..], out int rightLength, out bool rightHasPeriod) && !rightHasPeriod && end + rightLength <= rightBound)
        {
            end += rightLength;
        }

        return start == span.Start && end == span.End ? span : span with { Start = start, End = end };
    }

    /// <summary>The leftmost <c>(Lu name{0,3})(connector{1,3})</c> that ends exactly at the end of <paramref name="prefix"/>.</summary>
    private static bool MatchLeftParticle(ReadOnlySpan<char> prefix, out int length, out int nameLength, out bool connectorHasPeriod)
    {
        int n = prefix.Length;
        for (int i = Math.Max(0, n - 7); i <= n - 2; i++)
        {
            if (!IsUpper(prefix[i])) continue;
            // Greedy name part: up to three name characters after the capital, backing off as needed.
            int maxExtra = 0;
            while (maxExtra < 3 && i + 1 + maxExtra < n && IsNameChar(prefix[i + 1 + maxExtra])) maxExtra++;
            for (int extra = maxExtra; extra >= 0; extra--)
            {
                int connectorStart = i + 1 + extra;
                int connectorLength = n - connectorStart;
                if (connectorLength is < 1 or > 3) continue;
                var connector = prefix[connectorStart..];
                bool all = true;
                foreach (var c in connector) all &= IsConnector(c);
                if (!all) continue;
                length = n - i;
                nameLength = 1 + extra;
                connectorHasPeriod = connector.Contains('.');
                return true;
            }
        }
        length = nameLength = 0;
        connectorHasPeriod = false;
        return false;
    }

    /// <summary>The <c>(connector{1,3})(Lu name{0,3})</c> at the start of <paramref name="suffix"/>, greedy.</summary>
    private static bool MatchRightParticle(ReadOnlySpan<char> suffix, out int length, out bool connectorHasPeriod)
    {
        int run = 0;
        while (run < 3 && run < suffix.Length && IsConnector(suffix[run])) run++;
        for (int connector = run; connector >= 1; connector--)
        {
            if (connector >= suffix.Length || !IsUpper(suffix[connector])) continue;
            int extra = 0;
            while (extra < 3 && connector + 1 + extra < suffix.Length && IsNameChar(suffix[connector + 1 + extra])) extra++;
            length = connector + 1 + extra;
            connectorHasPeriod = suffix[..connector].Contains('.');
            return true;
        }
        length = 0;
        connectorHasPeriod = false;
        return false;
    }
}
