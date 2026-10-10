using System.Buffers;
using System.Collections.Frozen;
using Rampart.Buffers;
using System.Globalization;
using System.Text;

namespace Rampart.Tokenization;

/// <summary>
/// BERT-uncased WordPiece tokenizer, reproducing the tokenizer the Rampart pipeline runs in
/// transformers.js (<c>BertNormalizer</c> → <c>BertPreTokenizer</c> → <c>WordPiece</c>):
/// <list type="number">
/// <item>split out literal special tokens (<c>[CLS]</c>, <c>[SEP]</c>, …) exactly as written;</item>
/// <item>clean the text (drop NUL, U+FFFD and control/format characters, map whitespace to a space);</item>
/// <item>pad CJK ideographs with spaces;</item>
/// <item>lowercase, then NFD and strip non-spacing marks (<c>José</c> → <c>jose</c>);</item>
/// <item>split on whitespace and punctuation (every punctuation character is its own word);</item>
/// <item>greedy longest-match WordPiece with the <c>##</c> continuation prefix; a word longer than
/// 100 code points, or one that cannot be fully covered, becomes <c>[UNK]</c>.</item>
/// </list>
/// </summary>
internal sealed class WordPieceTokenizer
{
    public const string ContinuationPrefix = "##";
    private const int MaxInputCharsPerWord = 100;

    private static readonly string[] SpecialTokenStrings = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

    private readonly FrozenDictionary<string, int> _vocab;
    private readonly FrozenDictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _lookup;
    private readonly string[] _idToToken;
    private readonly FrozenSet<int> _specialIds;

    public int PadId { get; }
    public int UnknownId { get; }
    public int ClsId { get; }
    public int SepId { get; }
    public int MaskId { get; }
    public int VocabularySize => _idToToken.Length;

    private WordPieceTokenizer(string[] tokens)
    {
        _idToToken = tokens;
        var map = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++) map.TryAdd(tokens[i], i);
        _vocab = map.ToFrozenDictionary(StringComparer.Ordinal);
        _lookup = _vocab.GetAlternateLookup<ReadOnlySpan<char>>();
        PadId = Id("[PAD]");
        UnknownId = Id("[UNK]");
        ClsId = Id("[CLS]");
        SepId = Id("[SEP]");
        MaskId = Id("[MASK]");
        _specialIds = new[] { PadId, UnknownId, ClsId, SepId, MaskId }.ToFrozenSet();

        int Id(string token) => _vocab.TryGetValue(token, out var id) ? id : throw new InvalidDataException($"Vocabulary has no {token} token.");
    }

    /// <summary>Builds the tokenizer from a <c>vocab.txt</c> (one token per line, id = line number).</summary>
    public static WordPieceTokenizer FromVocab(ReadOnlySpan<byte> vocabTxt)
    {
        var text = Encoding.UTF8.GetString(vocabTxt);
        var lines = text.Split('\n');
        int count = lines.Length;
        if (count > 0 && lines[^1].Length == 0) count--;
        var tokens = new string[count];
        for (int i = 0; i < count; i++) tokens[i] = lines[i].TrimEnd('\r');
        return new WordPieceTokenizer(tokens);
    }

    /// <summary>The token string for an id.</summary>
    public string IdToToken(int id) => _idToToken[id];

    /// <summary>The id of a vocabulary token, or the <c>[UNK]</c> id.</summary>
    public int TokenToId(string token) => _vocab.TryGetValue(token, out var id) ? id : UnknownId;

    /// <summary>True for <c>[PAD]</c>, <c>[UNK]</c>, <c>[CLS]</c>, <c>[SEP]</c> and <c>[MASK]</c>.</summary>
    public bool IsSpecial(int id) => _specialIds.Contains(id);

    /// <summary>WordPiece tokens for <paramref name="text"/>, without <c>[CLS]</c>/<c>[SEP]</c>.</summary>
    public List<string> Tokenize(string text)
    {
        var ids = new PooledList<int>(text.Length / 3 + 8);
        try
        {
            EncodeTo(text, ref ids);
            var output = new List<string>(ids.Count);
            foreach (var id in ids.AsSpan()) output.Add(_idToToken[id]);
            return output;
        }
        finally
        {
            ids.Dispose();
        }
    }

    /// <summary>Token ids for <paramref name="text"/>, optionally wrapped in <c>[CLS]</c> … <c>[SEP]</c>.</summary>
    public int[] Encode(string text, bool addSpecialTokens = true) => Encode(text.AsSpan(), addSpecialTokens);

    /// <inheritdoc cref="Encode(string, bool)"/>
    public int[] Encode(ReadOnlySpan<char> text, bool addSpecialTokens = true)
    {
        var ids = new PooledList<int>(text.Length / 3 + 8);
        try
        {
            if (addSpecialTokens) ids.Add(ClsId);
            EncodeTo(text, ref ids);
            if (addSpecialTokens) ids.Add(SepId);
            return ids.AsSpan().ToArray();
        }
        finally
        {
            ids.Dispose();
        }
    }

    /// <summary>Number of model tokens in <paramref name="text"/>, excluding <c>[CLS]</c>/<c>[SEP]</c>. Allocates nothing.</summary>
    public int CountTokens(string text) => CountTokens(text.AsSpan());

    /// <inheritdoc cref="CountTokens(string)"/>
    public int CountTokens(ReadOnlySpan<char> text)
    {
        var ids = new PooledList<int>(text.Length / 3 + 8);
        try
        {
            EncodeTo(text, ref ids);
            return ids.Count;
        }
        finally
        {
            ids.Dispose();
        }
    }

    /// <summary>
    /// Appends the WordPiece ids of <paramref name="text"/> (no <c>[CLS]</c>/<c>[SEP]</c>) to
    /// <paramref name="ids"/>. The whole pipeline runs over pooled buffers: once the pool and the
    /// per-character fold cache are warm, encoding allocates nothing.
    /// </summary>
    internal void EncodeTo(ReadOnlySpan<char> text, ref PooledList<int> ids)
    {
        var normalized = new PooledList<char>(text.Length + 16);
        try
        {
            int start = 0, i = 0;
            while (i < text.Length)
            {
                int length = 0;
                int special = text[i] == '[' ? MatchSpecialToken(text[i..], out length) : -1;
                if (special < 0)
                {
                    i++;
                    continue;
                }
                if (i > start) EncodeSection(text[start..i], ref ids, ref normalized);
                ids.Add(special);
                i += length;
                start = i;
            }
            if (start < text.Length) EncodeSection(text[start..], ref ids, ref normalized);
        }
        finally
        {
            normalized.Dispose();
        }
    }

    /// <summary>transformers.js splits added tokens out first: the longest literal special token at the start of <paramref name="text"/>.</summary>
    private int MatchSpecialToken(ReadOnlySpan<char> text, out int length)
    {
        int best = -1;
        length = 0;
        foreach (var s in SpecialTokenStrings)
        {
            if (s.Length > length && text.StartsWith(s, StringComparison.Ordinal))
            {
                length = s.Length;
                best = _vocab[s];
            }
        }
        return best;
    }

    private void EncodeSection(ReadOnlySpan<char> section, ref PooledList<int> ids, ref PooledList<char> normalized)
    {
        normalized.Clear();
        NormalizeInto(section, ref normalized);
        var text = JsChar.Trim(normalized.AsSpan());

        // BertPreTokenizer: [^\s\p{P}…]+ | [\p{P}…] — every punctuation character is its own word.
        int i = 0;
        while (i < text.Length)
        {
            Rune.DecodeFromUtf16(text[i..], out var rune, out int width);
            if (rune.IsBmp && JsChar.IsWhiteSpace((char)rune.Value))
            {
                i += width;
                continue;
            }
            if (IsPunctuation(rune))
            {
                WordPiece(text.Slice(i, width), ref ids);
                i += width;
                continue;
            }
            int start = i;
            while (i < text.Length)
            {
                Rune.DecodeFromUtf16(text[i..], out var r, out int w);
                if ((r.IsBmp && JsChar.IsWhiteSpace((char)r.Value)) || IsPunctuation(r)) break;
                i += w;
            }
            WordPiece(text[start..i], ref ids);
        }
    }

    // ----- BertNormalizer -----

    /// <summary>The model-space fold of <paramref name="text"/> (clean, CJK padding, lowercase, strip accents).</summary>
    public static string Normalize(string text)
    {
        var buffer = new PooledList<char>(text.Length + 16);
        try
        {
            NormalizeInto(text, ref buffer);
            return buffer.AsSpan().ToString();
        }
        finally
        {
            buffer.Dispose();
        }
    }

    /// <summary>
    /// <c>BertNormalizer</c> in one pass: drop NUL, U+FFFD and control/format/private-use/surrogate code
    /// points (tab, newline and carriage return are whitespace); map whitespace to a space; pad CJK
    /// ideographs with spaces; then lowercase, NFD-decompose and strip non-spacing marks — per code point
    /// through <see cref="Fold"/>, which is what the reference's whole-string calls amount to.
    /// </summary>
    internal static void NormalizeInto(ReadOnlySpan<char> text, ref PooledList<char> output)
    {
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            if (c < 0x80)
            {
                i++;
                if (c == 0 || (c < 0x20 && c is not ('\t' or '\n' or '\r')) || c == 0x7F) continue; // NUL, Cc
                if (c is ' ' or '\t' or '\n' or '\r') output.Add(' ');
                else output.Add(c is >= 'A' and <= 'Z' ? (char)(c + 32) : c);
                continue;
            }

            var status = Rune.DecodeFromUtf16(text[i..], out var rune, out int width);
            i += width;
            // A lone surrogate is \p{Cs} to transformers.js and is dropped, like U+FFFD and controls.
            if (status != System.Buffers.OperationStatus.Done || rune.Value == 0xFFFD || IsControl(rune)) continue;
            if (rune.IsBmp && JsChar.IsWhiteSpace((char)rune.Value))
            {
                output.Add(' ');
                continue;
            }
            // transformers.js pads by UTF-16 code unit, so only the BMP CJK ranges ever match.
            bool cjk = rune.Value is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0xF900 and <= 0xFAFF);
            if (cjk) output.Add(' ');
            if (rune.Value == 0x03A3)
            {
                // Σ lowercases by context (final sigma), so it cannot come from the per-code-point cache.
                output.Add(IsFinalSigma(text, i - 1) ? '\u03C2' : '\u03C3');
            }
            else
            {
                output.AddRange(Fold(rune));
            }
            if (cjk) output.Add(' ');
        }
    }

    private static readonly string?[] BmpFolds = new string?[0x10000];
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, string> AstralFolds = new();

    /// <summary>
    /// <c>toLowerCase</c>, NFD, minus non-spacing marks, of one non-ASCII code point — computed on first
    /// sight and cached, since ICU normalization allocates. Concurrent first writers store equal strings.
    /// </summary>
    private static ReadOnlySpan<char> Fold(Rune rune)
    {
        if (rune.IsBmp)
        {
            return BmpFolds[rune.Value] ??= ComputeFold(rune);
        }
        return AstralFolds.GetOrAdd(rune.Value, static cp => ComputeFold(new Rune(cp)));
    }

    private static string ComputeFold(Rune rune)
    {
        var lower = JsLowerCase(rune.ToString());
        var nfd = lower.IsNormalized(NormalizationForm.FormD) ? lower : lower.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(nfd.Length);
        foreach (var r in nfd.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(r) != UnicodeCategory.NonSpacingMark) sb.Append(r.ToString());
        }
        return sb.ToString();
    }

    private static bool IsControl(Rune rune)
    {
        if (rune.Value is '\t' or '\n' or '\r') return false;
        return Rune.GetUnicodeCategory(rune) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.PrivateUse or UnicodeCategory.Surrogate;
    }

    /// <summary>
    /// <c>String.prototype.toLowerCase</c>: the invariant simple mapping plus the two unconditional
    /// context rules JavaScript applies — U+0130 → "i̇" and the Greek final sigma.
    /// </summary>
    internal static string JsLowerCase(string text)
    {
        var lower = text.ToLowerInvariant();
        if (text.AsSpan().IndexOfAny('\u0130', '\u03A3') < 0) return lower;

        var sb = new StringBuilder(lower.Length + 4);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\u0130') sb.Append("i\u0307");
            else if (c == '\u03A3') sb.Append(IsFinalSigma(text, i) ? '\u03C2' : '\u03C3');
            else sb.Append(lower[i]);
        }
        return sb.ToString();
    }

    /// <summary><see cref="JsLowerCase"/> of a span, appended to <paramref name="output"/> without allocating.</summary>
    internal static void JsLowerCaseInto(ReadOnlySpan<char> text, ref PooledList<char> output)
    {
        Span<char> pair = stackalloc char[2];
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c < 0x80) output.Add(c is >= 'A' and <= 'Z' ? (char)(c + 32) : c);
            else if (c == '\u0130')
            {
                output.Add('i');
                output.Add('\u0307');
            }
            else if (c == '\u03A3') output.Add(IsFinalSigma(text, i) ? '\u03C2' : '\u03C3');
            else if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                var lower = Rune.ToLowerInvariant(new Rune(c, text[i + 1]));
                output.AddRange(pair[..lower.EncodeToUtf16(pair)]);
                i++;
            }
            else output.Add(char.ToLowerInvariant(c));
        }
    }

    private static bool IsFinalSigma(ReadOnlySpan<char> text, int index)
    {
        // Final_Sigma: preceded by a cased letter and not followed by one (case-ignorables skipped).
        int i = index - 1;
        while (i >= 0 && IsCaseIgnorable(text[i])) i--;
        if (i < 0 || !IsCased(text[i])) return false;
        int j = index + 1;
        while (j < text.Length && IsCaseIgnorable(text[j])) j++;
        return j >= text.Length || !IsCased(text[j]);

        static bool IsCased(char c) => char.IsUpper(c) || char.IsLower(c) || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.TitlecaseLetter;
        static bool IsCaseIgnorable(char c) => CharUnicodeInfo.GetUnicodeCategory(c) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark
            or UnicodeCategory.Format or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol || c is '\'' or '.' or ':' or '\u2019' or '\u00B7';
    }

    internal static bool IsPunctuation(Rune rune)
    {
        int cp = rune.Value;
        if (cp is (>= 0x21 and <= 0x2F) or (>= 0x3A and <= 0x40) or (>= 0x5B and <= 0x60) or (>= 0x7B and <= 0x7E)) return true;
        return Rune.GetUnicodeCategory(rune) is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
            or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
    }

    // ----- WordPiece (greedy longest match over code points) -----

    private void WordPiece(ReadOnlySpan<char> word, ref PooledList<int> ids)
    {
        // Code point boundaries (the JS implementation spreads the word into code points).
        int[]? rentedBounds = null;
        char[]? rentedBuffer = null;
        Span<int> bounds = word.Length <= 128 ? stackalloc int[word.Length + 1] : (rentedBounds = ArrayPool<int>.Shared.Rent(word.Length + 1));
        Span<char> buffer = word.Length <= 254 ? stackalloc char[word.Length + 2] : (rentedBuffer = ArrayPool<char>.Shared.Rent(word.Length + 2));
        try
        {
            int count = 0;
            for (int i = 0; i < word.Length;)
            {
                bounds[count++] = i;
                i += char.IsHighSurrogate(word[i]) && i + 1 < word.Length && char.IsLowSurrogate(word[i + 1]) ? 2 : 1;
            }
            bounds[count] = word.Length;

            if (count > MaxInputCharsPerWord)
            {
                ids.Add(UnknownId);
                return;
            }

            int mark = ids.Count;
            int start = 0;
            while (start < count)
            {
                int end = count, found = -1;
                while (start < end)
                {
                    var sub = word[bounds[start]..bounds[end]];
                    scoped ReadOnlySpan<char> candidate;
                    if (start > 0)
                    {
                        ContinuationPrefix.AsSpan().CopyTo(buffer);
                        sub.CopyTo(buffer[2..]);
                        candidate = buffer[..(sub.Length + 2)];
                    }
                    else candidate = sub;

                    if (_lookup.TryGetValue(candidate, out found)) break;
                    found = -1;
                    end--;
                }
                if (found < 0)
                {
                    while (ids.Count > mark) ids.RemoveAt(ids.Count - 1);
                    ids.Add(UnknownId);
                    return;
                }
                ids.Add(found);
                start = end;
            }
        }
        finally
        {
            if (rentedBounds is not null) ArrayPool<int>.Shared.Return(rentedBounds);
            if (rentedBuffer is not null) ArrayPool<char>.Shared.Return(rentedBuffer);
        }
    }
}

/// <summary>JavaScript's notion of whitespace (<c>\s</c> and <c>String.prototype.trim</c>).</summary>
internal static class JsChar
{
    public static bool IsWhiteSpace(char c) => c is '\t' or '\n' or '\v' or '\f' or '\r' or ' ' or '\u00A0' or '\u1680'
        or (>= '\u2000' and <= '\u200A') or '\u2028' or '\u2029' or '\u202F' or '\u205F' or '\u3000' or '\uFEFF';

    public static ReadOnlySpan<char> Trim(ReadOnlySpan<char> text)
    {
        int s = 0, e = text.Length;
        while (s < e && IsWhiteSpace(text[s])) s++;
        while (e > s && IsWhiteSpace(text[e - 1])) e--;
        return text[s..e];
    }

    public static string Trim(string text)
    {
        int s = 0, e = text.Length;
        while (s < e && IsWhiteSpace(text[s])) s++;
        while (e > s && IsWhiteSpace(text[e - 1])) e--;
        return s == 0 && e == text.Length ? text : text[s..e];
    }
}
