using System.Globalization;
using Rampart.Tokenization;

namespace Rampart.Detection;

/// <summary>
/// The hints of an <see cref="AnalyzeOptions"/>, compiled: one phrase per value to report wherever it
/// occurs, and per entity type the words of all its values, which extend a detected span of that type.
/// Not part of the reference.
/// </summary>
internal sealed class HintSet
{
    /// <summary>
    /// A hint split into words, with the exact text between consecutive words and the punctuation
    /// around them (the "(" of "(202) 555-0188"), which an occurrence includes when the text has it too.
    /// </summary>
    private sealed record Phrase(PiiLabel Label, string[] Words, string[] Gaps, string Prefix, string Suffix);

    private readonly List<Phrase> _phrases = [];
    private readonly Dictionary<PiiLabel, HashSet<string>> _vocabulary = [];
    private readonly StringComparer _comparer;
    private readonly StringComparison _comparison;

    private HintSet(bool ignoreCase)
    {
        _comparer = ignoreCase ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        _comparison = ignoreCase ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    }

    public static HintSet Build(IReadOnlyDictionary<PiiLabel, List<string>> hints, bool ignoreCase)
    {
        var set = new HintSet(ignoreCase);
        foreach (var (label, values) in hints)
        {
            foreach (var value in values)
            {
                if (string.IsNullOrWhiteSpace(value)) continue;
                var words = new List<string>();
                var gaps = new List<string>();
                int gapStart = -1;
                foreach (var (start, end) in Words(value))
                {
                    if (gapStart >= 0) gaps.Add(value[gapStart..start]);
                    words.Add(value[start..end]);
                    gapStart = end;
                }
                if (words.Count == 0) continue;
                var trimmed = value.Trim();
                int first = trimmed.IndexOf(words[0], StringComparison.Ordinal);
                int last = trimmed.LastIndexOf(words[^1], StringComparison.Ordinal) + words[^1].Length;
                set._phrases.Add(new Phrase(label, [.. words], [.. gaps], trimmed[..first], trimmed[last..]));
                // The model splits one person's name across the two name types, so they share words.
                foreach (var target in IsPerson(label) ? [PiiLabel.GivenName, PiiLabel.Surname] : new[] { label })
                {
                    if (!set._vocabulary.TryGetValue(target, out var vocabulary))
                        set._vocabulary[target] = vocabulary = new HashSet<string>(set._comparer);
                    vocabulary.UnionWith(words);
                }
            }
        }
        return set;
    }

    private static bool IsPerson(PiiLabel label) => label is PiiLabel.GivenName or PiiLabel.Surname;

    /// <summary>A word character: a letter, combining mark or digit. Apostrophes and hyphens separate words.</summary>
    internal static bool IsWordChar(char c) => CharUnicodeInfo.GetUnicodeCategory(c) is
        UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter
        or UnicodeCategory.OtherLetter or UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark
        or UnicodeCategory.DecimalDigitNumber;

    /// <summary>Separators that may stand between the words of a name in either text: whitespace, apostrophes, periods, hyphens.</summary>
    private static bool IsLooseGap(ReadOnlySpan<char> gap)
    {
        if (gap.IsEmpty || gap.Length > 3) return false;
        foreach (var c in gap)
        {
            if (!(JsChar.IsWhiteSpace(c) || c is '\'' or '’' or '.' or '-')) return false;
        }
        return true;
    }

    private static List<(int Start, int End)> Words(ReadOnlySpan<char> text)
    {
        var words = new List<(int, int)>();
        int i = 0;
        while (i < text.Length)
        {
            if (!IsWordChar(text[i])) { i++; continue; }
            int start = i;
            while (i < text.Length && IsWordChar(text[i])) i++;
            words.Add((start, i));
        }
        return words;
    }

    /// <summary>
    /// Every occurrence of a hint in <paramref name="text"/>: whole words, in order, separated either by
    /// the hint's own separator or by up to three whitespace, apostrophe, period or hyphen characters.
    /// </summary>
    public List<(int Start, int End, PiiLabel Label)> FindOccurrences(string text)
    {
        var found = new List<(int, int, PiiLabel)>();
        if (_phrases.Count == 0) return found;
        var words = Words(text);
        foreach (var phrase in _phrases)
        {
            for (int i = 0; i + phrase.Words.Length <= words.Count; i++)
            {
                bool match = true;
                for (int w = 0; match && w < phrase.Words.Length; w++)
                {
                    var (start, end) = words[i + w];
                    match = text.AsSpan(start, end - start).Equals(phrase.Words[w], _comparison);
                    if (match && w > 0)
                    {
                        var gap = text.AsSpan(words[i + w - 1].End, start - words[i + w - 1].End);
                        match = gap.SequenceEqual(phrase.Gaps[w - 1]) || (IsLooseGap(gap) && IsLooseGap(phrase.Gaps[w - 1]));
                    }
                }
                if (!match) continue;
                int from = words[i].Start, to = words[i + phrase.Words.Length - 1].End;
                if (phrase.Prefix.Length > 0 && text.AsSpan(0, from).EndsWith(phrase.Prefix, StringComparison.Ordinal)) from -= phrase.Prefix.Length;
                if (phrase.Suffix.Length > 0 && text.AsSpan(to).StartsWith(phrase.Suffix, StringComparison.Ordinal)) to += phrase.Suffix.Length;
                found.Add((from, to, phrase.Label));
            }
        }
        return found;
    }

    /// <summary>True when <paramref name="word"/> is one of the hint words for <paramref name="label"/>.</summary>
    public bool IsHintWord(PiiLabel label, ReadOnlySpan<char> word)
        => _vocabulary.TryGetValue(label, out var vocabulary)
           && vocabulary.GetAlternateLookup<ReadOnlySpan<char>>().Contains(word);

    public bool HasWordsFor(PiiLabel label) => _vocabulary.ContainsKey(label);
}
