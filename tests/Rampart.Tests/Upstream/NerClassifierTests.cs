using System.Text;
using System.Text.RegularExpressions;
using Rampart.Ner;

namespace Rampart.Tests.Upstream;

/// <summary>
/// Port of <c>__tests__/ner-classifier.test.ts</c>: <see cref="NerDetector"/> driven by stand-in
/// classifiers, exactly as the reference drives <c>detectNer</c> — aggregated and raw BIO rows,
/// mocked tokenizers, windowing by a mocked token count.
/// </summary>
public sealed partial class NerClassifierTests
{
    private const int Budget = NerDetector.TokenBudget;

    [GeneratedRegex(@"\S+")]
    private static partial Regex Word();

    // One token per whitespace-delimited word: drives the windowing deterministically.
    private static int CountWords(string text) => Word().Count(text);

    private static TokenEntity Group(string group, double score, int start, int end, string word)
        => new() { EntityGroup = group, Score = score, Start = start, End = end, Word = word };

    private static TokenEntity Bio(string entity, double score, string word, int? index = null)
        => new() { Entity = entity, Score = score, Word = word, Index = index, Start = 0, End = 0 };

    private static NerDetector Detector(Func<string, IReadOnlyList<TokenEntity>> classify, Func<string, int>? countTokens = null, Func<string, IReadOnlyList<string>>? tokenize = null)
        => new(new DelegateTokenClassifier(classify, countTokens, tokenize));

    [Fact]
    public void RunsOnHyphenNormalizedTextAndRepairsHyphenatedNames()
    {
        string modelInput = "";
        var detector = Detector(text =>
        {
            modelInput = text;
            return
            [
                Group("GIVEN_NAME", 0.55, 11, 16, "Thanh"),
                Group("GIVEN_NAME", 0.22, 17, 23, "Nghiem"),
                Group("GIVEN_NAME", 0.45, 24, 28, "Quoc"),
                Group("GIVEN_NAME", 0.21, 29, 32, "Bao"),
            ];
        });

        var spans = detector.Detect("My name is Thanh-Nghiem Quoc-Bao.");

        Assert.Equal("My name is Thanh Nghiem Quoc Bao.", modelInput);
        Assert.Equal([new PiiSpan(11, 32, PiiLabel.GivenName, 0.55, SpanSource.Ner, "Thanh-Nghiem Quoc-Bao")], spans);
    }

    [Fact]
    public void RecoversRawOffsetsForFoldedBioTokensOverAccentedText()
    {
        var detector = Detector(_ => [Bio("B-GIVEN_NAME", 0.99, "genevieve"), Bio("I-GIVEN_NAME", 0.99, "muller")]);
        var spans = detector.Detect("Bonjour Geneviève Müller, votre dossier est prêt.");
        Assert.Equal([new PiiSpan(8, 24, PiiLabel.GivenName, 0.99, SpanSource.Ner, "Geneviève Müller")], spans);
    }

    [Fact]
    public void ExtendsPersonSpansOverShortCapitalizedParticles()
    {
        var detector = Detector(_ => [Group("GIVEN_NAME", 0.61, 16, 21, "Croix")]);
        Assert.Equal(["De La Croix"], detector.Detect("Applicant De La Croix applied.").Select(s => s.Text));
    }

    [Fact]
    public void JoinsInitialsAcrossPeriodsButNotWordsAcrossSentenceBoundaries()
    {
        var initials = Detector(_ => [Group("GIVEN_NAME", 0.9, 9, 16, "Tolkien")]);
        Assert.Equal(["J. R. R. Tolkien"], initials.Detect("J. R. R. Tolkien wrote the book.").Select(s => s.Text));

        var boundary = Detector(_ => [Group("GIVEN_NAME", 0.9, 0, 12, "Maria Garcia")]);
        Assert.Equal(["Maria Garcia"], boundary.Detect("Maria Garcia. I think she left.").Select(s => s.Text));
    }

    [Fact]
    public void DoesNotMergeTwoNamesAcrossASentenceEndingPeriod()
    {
        var detector = Detector(_ => [Group("GIVEN_NAME", 0.9, 11, 16, "Maria"), Group("GIVEN_NAME", 0.9, 18, 21, "Bob")]);
        Assert.Equal(["Maria", "Bob"], detector.Detect("We emailed Maria. Bob replied later.").Select(s => s.Text));
    }

    private static IReadOnlyList<TokenEntity> FindJordanLee(string text)
    {
        int at = text.IndexOf("Jordan Lee", StringComparison.Ordinal);
        return at < 0 ? [] : [Group("GIVEN_NAME", 0.95, at, at + "Jordan Lee".Length, "Jordan Lee")];
    }

    [Fact]
    public void WindowsLongInputByTokenBudgetAndKeepsTheTail()
    {
        var inputs = new List<string>();
        var detector = Detector(text =>
        {
            inputs.Add(text);
            return FindJordanLee(text);
        }, CountWords);

        var longText = string.Concat(Enumerable.Repeat("word ", Budget * 2)) + "Jordan Lee";
        var spans = detector.Detect(longText);

        Assert.True(inputs.Count > 1);
        Assert.All(inputs, input => Assert.True(CountWords(input) <= Budget));
        var jordan = Assert.Single(spans, s => s.Text == "Jordan Lee");
        Assert.Equal("Jordan Lee", longText[jordan.Start..jordan.End]);
    }

    [Fact]
    public void RecoversAnEntityStraddlingAWindowSeamExactlyOnce()
    {
        var detector = Detector(FindJordanLee, CountWords);
        var filler = string.Join(" ", Enumerable.Repeat("word", Budget - 1));
        var text = $"{filler} Jordan Lee and the rest of the message continues.";
        var jordan = Assert.Single(detector.Detect(text), s => s.Text == "Jordan Lee");
        Assert.Equal("Jordan Lee", text[jordan.Start..jordan.End]);
    }

    [Fact]
    public void HardSplitsAnOverBudgetUnbrokenToken()
    {
        var inputs = new List<string>();
        var detector = Detector(text =>
        {
            inputs.Add(text);
            return [];
        }, text => text.Length); // one token per char

        detector.Detect(new string('x', Budget * 3 + 17));

        Assert.True(inputs.Count > 1);
        Assert.All(inputs, input => Assert.True(input.Length <= Budget));
    }

    private static int IndexFrom(IReadOnlyList<string> tokens, string piece, int from)
    {
        for (int i = from; i < tokens.Count; i++) if (tokens[i] == piece) return i;
        return -1;
    }

    [Fact]
    public void AnchorsBioOffsetsToTokenIndexSoDuplicateDigitRunsMapCorrectly()
    {
        const string text = "or 555-123-4567. I live at 1234 Maple Street.";
        static IReadOnlyList<string> Tokenize(string input)
        {
            var pieces = new List<string>();
            foreach (var word in Regex.Split(input.Replace('-', ' '), @"\s+"))
            {
                if (word.Length == 0) continue;
                if (word == "1234") pieces.AddRange(["123", "##4"]);
                else pieces.Add(word);
            }
            return pieces;
        }
        var detector = Detector(input =>
        {
            var tokens = Tokenize(input);
            int at = IndexFrom(tokens, "at", 0);
            int b123 = IndexFrom(tokens, "123", at + 1);
            int b4 = IndexFrom(tokens, "##4", b123 + 1);
            int maple = IndexFrom(tokens, "maple", b4 + 1);
            int street = IndexFrom(tokens, "street", maple + 1);
            return
            [
                Bio("B-BUILDING_NUMBER", 0.99, "123", b123 + 1),
                Bio("B-BUILDING_NUMBER", 0.99, "##4", b4 + 1),
                Bio("B-STREET_NAME", 0.99, "maple", maple + 1),
                Bio("I-STREET_NAME", 0.99, "street", street + 1),
            ];
        }, tokenize: Tokenize);

        var spans = detector.Detect(text);
        var building = spans.Single(s => s.Label == PiiLabel.BuildingNumber);
        Assert.Equal("1234", building.Text);
        Assert.Equal("1234", text[building.Start..building.End]);
        // As upstream: the mock tokenizer does not lowercase, so "maple"/"street" are not among its pieces;
        // those rows carry index 0 (the [CLS] slot, no offsets) and are located by the word-search fallback.
        Assert.Equal("Maple Street", spans.Single(s => s.Label == PiiLabel.StreetName).Text);
    }

    private static string Fold(string s)
    {
        var nfkd = s.ToLowerInvariant().Normalize(NormalizationForm.FormKD);
        return new string(nfkd.Where(c => c is < '̀' or > 'ͯ').ToArray());
    }

    [Fact]
    public void RecoversRawOffsetsForAccentedWordsFromFoldedPieces()
    {
        const string text = "I live at 1234 Cárdenas Boulevard.";
        static IReadOnlyList<string> Tokenize(string input)
        {
            var pieces = new List<string>();
            foreach (var word in Regex.Split(Fold(input.Replace('-', ' ')), @"\s+"))
            {
                if (word.Length == 0) continue;
                if (word == "cardenas") pieces.AddRange(["car", "##denas"]);
                else if (word == "boulevard.") pieces.AddRange(["boulevard", "."]);
                else pieces.Add(word);
            }
            return pieces;
        }
        var detector = Detector(input =>
        {
            var tokens = Tokenize(input);
            int building = IndexFrom(tokens, "1234", 0);
            int car = IndexFrom(tokens, "car", building + 1);
            int denas = IndexFrom(tokens, "##denas", car + 1);
            int blvd = IndexFrom(tokens, "boulevard", denas + 1);
            return
            [
                Bio("B-BUILDING_NUMBER", 0.99, "1234", building + 1),
                Bio("B-STREET_NAME", 0.99, "car", car + 1),
                Bio("I-STREET_NAME", 0.99, "##denas", denas + 1),
                Bio("I-STREET_NAME", 0.99, "boulevard", blvd + 1),
            ];
        }, tokenize: Tokenize);

        var spans = detector.Detect(text);
        Assert.Equal("1234", spans.Single(s => s.Label == PiiLabel.BuildingNumber).Text);
        var street = spans.Single(s => s.Label == PiiLabel.StreetName);
        Assert.Equal("Cárdenas Boulevard", street.Text);
        Assert.Equal("Cárdenas Boulevard", text[street.Start..street.End]);
    }

    [Fact]
    public void KeepsAdjacentGivenNameAndSurnameSeparate()
    {
        const string text = "Please contact José Ångström today.";
        static IReadOnlyList<string> Tokenize(string input)
        {
            var pieces = new List<string>();
            foreach (var word in Regex.Split(Fold(input.Replace('-', ' ')), @"\s+"))
            {
                if (word.Length == 0) continue;
                if (word == "angstrom") pieces.AddRange(["ang", "##strom"]);
                else if (word == "today.") pieces.AddRange(["today", "."]);
                else pieces.Add(word);
            }
            return pieces;
        }
        var detector = Detector(input =>
        {
            var tokens = Tokenize(input);
            int jose = IndexFrom(tokens, "jose", 0);
            int ang = IndexFrom(tokens, "ang", jose + 1);
            int strom = IndexFrom(tokens, "##strom", ang + 1);
            return
            [
                Bio("B-GIVEN_NAME", 0.99, "jose", jose + 1),
                Bio("B-SURNAME", 0.99, "ang", ang + 1),
                Bio("B-SURNAME", 0.99, "##strom", strom + 1),
            ];
        }, tokenize: Tokenize);

        var spans = detector.Detect(text);
        Assert.Equal("José", spans.Single(s => s.Label == PiiLabel.GivenName).Text);
        Assert.Equal("Ångström", spans.Single(s => s.Label == PiiLabel.Surname).Text);
    }

    [Fact]
    public void DoesNotKeepIsolatedBelowAnchorSpans()
    {
        var detector = Detector(_ => [Group("GIVEN_NAME", 0.22, 11, 16, "Maybe")]);
        Assert.Empty(detector.Detect("The word is Maybe here."));
    }
}
