using System.Collections.Frozen;
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
public sealed class WordPieceTokenizer
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
        var output = new List<string>();
        foreach (var (section, special) in SplitSpecialTokens(text))
        {
            if (special)
            {
                output.Add(section);
                continue;
            }
            var normalized = Normalize(section);
            if (normalized.Length == 0) continue;
            foreach (var word in PreTokenize(normalized)) WordPiece(word, output);
        }
        return output;
    }

    /// <summary>Token ids for <paramref name="text"/>, optionally wrapped in <c>[CLS]</c> … <c>[SEP]</c>.</summary>
    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var tokens = Tokenize(text);
        var ids = new int[tokens.Count + (addSpecialTokens ? 2 : 0)];
        int o = 0;
        if (addSpecialTokens) ids[o++] = ClsId;
        foreach (var t in tokens) ids[o++] = TokenToId(t);
        if (addSpecialTokens) ids[o] = SepId;
        return ids;
    }

    /// <summary>Number of model tokens in <paramref name="text"/>, excluding <c>[CLS]</c>/<c>[SEP]</c>.</summary>
    public int CountTokens(string text) => Tokenize(text).Count;

    // ----- added-token splitting (transformers.js DictionarySplitter: longest literal match) -----

    private static IEnumerable<(string Section, bool Special)> SplitSpecialTokens(string text)
    {
        int start = 0, i = 0;
        while (i < text.Length)
        {
            string? match = null;
            if (text[i] == '[')
            {
                foreach (var s in SpecialTokenStrings)
                {
                    if (string.CompareOrdinal(text, i, s, 0, s.Length) == 0 && (match is null || s.Length > match.Length)) match = s;
                }
            }
            if (match is not null)
            {
                if (i > start) yield return (text[start..i], false);
                yield return (match, true);
                i += match.Length;
                start = i;
            }
            else i++;
        }
        if (start < text.Length) yield return (text[start..], false);
    }

    // ----- BertNormalizer -----

    /// <summary>The model-space fold of <paramref name="text"/> (clean, CJK padding, lowercase, strip accents).</summary>
    public static string Normalize(string text)
    {
        var cleaned = CleanText(text);
        var spaced = PadChineseChars(cleaned);
        var lower = JsLowerCase(spaced);
        return StripAccents(lower);
    }

    private static string CleanText(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            int cp = rune.Value;
            // A lone surrogate enumerates as U+FFFD; transformers.js drops it too (as \p{Cs}).
            if (cp == 0 || cp == 0xFFFD || IsControl(rune)) continue;
            if (rune.IsBmp && JsChar.IsWhiteSpace((char)cp)) sb.Append(' ');
            else sb.Append(rune.ToString());
        }
        return sb.ToString();
    }

    private static bool IsControl(Rune rune)
    {
        if (rune.Value is '\t' or '\n' or '\r') return false;
        return Rune.GetUnicodeCategory(rune) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.PrivateUse or UnicodeCategory.Surrogate;
    }

    private static string PadChineseChars(string text)
    {
        // transformers.js walks UTF-16 code units here, so only the BMP ranges can ever match.
        StringBuilder? sb = null;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c is (>= '\u4E00' and <= '\u9FFF') or (>= '\u3400' and <= '\u4DBF') or (>= '\uF900' and <= '\uFAFF'))
            {
                sb ??= new StringBuilder(text, 0, i, text.Length + 16);
                sb.Append(' ').Append(c).Append(' ');
            }
            else sb?.Append(c);
        }
        return sb?.ToString() ?? text;
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

    private static bool IsFinalSigma(string text, int index)
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

    private static string StripAccents(string text)
    {
        var nfd = text.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(nfd.Length);
        foreach (var rune in nfd.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.NonSpacingMark) sb.Append(rune.ToString());
        }
        return sb.ToString();
    }

    // ----- BertPreTokenizer: [^\s\p{P}…]+ | [\p{P}…] over the trimmed text -----

    internal static List<string> PreTokenize(string text)
    {
        var words = new List<string>();
        var span = JsChar.Trim(text);
        int i = 0;
        while (i < span.Length)
        {
            Rune.DecodeFromUtf16(span.AsSpan(i), out var rune, out int width);
            if (rune.IsBmp && JsChar.IsWhiteSpace((char)rune.Value))
            {
                i += width;
                continue;
            }
            if (IsPunctuation(rune))
            {
                words.Add(span.Substring(i, width));
                i += width;
                continue;
            }
            int start = i;
            while (i < span.Length)
            {
                Rune.DecodeFromUtf16(span.AsSpan(i), out var r, out int w);
                if ((r.IsBmp && JsChar.IsWhiteSpace((char)r.Value)) || IsPunctuation(r)) break;
                i += w;
            }
            words.Add(span[start..i]);
        }
        return words;
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

    private void WordPiece(string word, List<string> output)
    {
        // Code point boundaries (the JS implementation spreads the word into code points).
        Span<int> bounds = word.Length <= 256 ? stackalloc int[word.Length + 1] : new int[word.Length + 1];
        int count = 0;
        for (int i = 0; i < word.Length;)
        {
            bounds[count++] = i;
            i += char.IsHighSurrogate(word[i]) && i + 1 < word.Length && char.IsLowSurrogate(word[i + 1]) ? 2 : 1;
        }
        bounds[count] = word.Length;

        if (count > MaxInputCharsPerWord)
        {
            output.Add("[UNK]");
            return;
        }

        int mark = output.Count;
        Span<char> buffer = word.Length + 2 <= 512 ? stackalloc char[word.Length + 2] : new char[word.Length + 2];
        int start = 0;
        while (start < count)
        {
            int end = count;
            string? piece = null;
            while (start < end)
            {
                var sub = word.AsSpan(bounds[start], bounds[end] - bounds[start]);
                scoped ReadOnlySpan<char> candidate;
                if (start > 0)
                {
                    ContinuationPrefix.AsSpan().CopyTo(buffer);
                    sub.CopyTo(buffer[2..]);
                    candidate = buffer[..(sub.Length + 2)];
                }
                else candidate = sub;

                if (_lookup.TryGetValue(candidate, out var id))
                {
                    piece = _idToToken[id];
                    break;
                }
                end--;
            }
            if (piece is null)
            {
                output.RemoveRange(mark, output.Count - mark);
                output.Add("[UNK]");
                return;
            }
            output.Add(piece);
            start = end;
        }
    }
}

/// <summary>JavaScript's notion of whitespace (<c>\s</c> and <c>String.prototype.trim</c>).</summary>
internal static class JsChar
{
    public static bool IsWhiteSpace(char c) => c is '\t' or '\n' or '\v' or '\f' or '\r' or ' ' or '\u00A0' or '\u1680'
        or (>= '\u2000' and <= '\u200A') or '\u2028' or '\u2029' or '\u202F' or '\u205F' or '\u3000' or '\uFEFF';

    public static string Trim(string text)
    {
        int s = 0, e = text.Length;
        while (s < e && IsWhiteSpace(text[s])) s++;
        while (e > s && IsWhiteSpace(text[e - 1])) e--;
        return s == 0 && e == text.Length ? text : text[s..e];
    }
}
