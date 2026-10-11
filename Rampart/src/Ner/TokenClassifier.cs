using System.Buffers;
using Catalyst.Rampart.Buffers;
using Catalyst.Rampart.Model;
using Catalyst.Rampart.Tokenization;

namespace Catalyst.Rampart.Ner;

/// <summary>
/// One token-classification row, shaped like a transformers.js <c>token-classification</c> result.
/// Two shapes are accepted, as in the reference:
/// <list type="bullet">
/// <item>raw BIO tokens — <see cref="Entity"/> (<c>B-GIVEN_NAME</c>), <see cref="Index"/> into
/// <c>[CLS] + content + [SEP]</c> and the WordPiece <see cref="Word"/>; this is what
/// <see cref="TokenClassifier"/> emits;</item>
/// <item>aggregated entities — <see cref="EntityGroup"/> with char <see cref="Start"/>/<see cref="End"/>
/// into the classified text, used as they are.</item>
/// </list>
/// </summary>
internal sealed record TokenEntity
{
    /// <summary>The BIO label of a raw token row (<c>B-GIVEN_NAME</c>).</summary>
    public string? Entity { get; init; }

    /// <summary>The entity group of an aggregated row (<c>GIVEN_NAME</c>).</summary>
    public string? EntityGroup { get; init; }

    /// <summary>Probability of the label.</summary>
    public double Score { get; init; }

    /// <summary>Token index in <c>[CLS] + content + [SEP]</c>; drives offset recovery for raw rows.</summary>
    public int? Index { get; init; }

    /// <summary>The token (<c>##</c>-prefixed for a WordPiece continuation) or the entity's text.</summary>
    public string Word { get; init; } = "";

    /// <summary>Char offsets into the classified text, for aggregated rows.</summary>
    public int? Start { get; init; }

    /// <inheritdoc cref="Start"/>
    public int? End { get; init; }
}

/// <summary>
/// The contract <see cref="NerDetector"/> runs a classifier through — the reference's
/// <c>TokenClassifier</c>: a call returning rows, plus an optional tokenizer.
/// </summary>
internal interface ITokenClassifier
{
    /// <summary>Classify <paramref name="text"/>; returns the non-<c>O</c> rows in order.</summary>
    IReadOnlyList<TokenEntity> Classify(string text);

    /// <summary>
    /// Model tokens in a string, excluding <c>[CLS]</c>/<c>[SEP]</c>. <c>null</c> when the classifier
    /// has no tokenizer: the detector then sends the whole input as a single window.
    /// </summary>
    Func<string, int>? CountTokens => null;

    /// <summary>
    /// WordPiece tokens of a string (no specials), used to map a row's <see cref="TokenEntity.Index"/>
    /// to char offsets. <c>null</c>: raw rows are located by searching for their <see cref="TokenEntity.Word"/>.
    /// </summary>
    Func<string, IReadOnlyList<string>>? Tokenize => null;
}

/// <summary>An <see cref="ITokenClassifier"/> made of delegates — for another model, or a test double.</summary>
internal sealed class DelegateTokenClassifier(
    Func<string, IReadOnlyList<TokenEntity>> classify,
    Func<string, int>? countTokens = null,
    Func<string, IReadOnlyList<string>>? tokenize = null) : ITokenClassifier
{
    public IReadOnlyList<TokenEntity> Classify(string text) => classify(text);

    public Func<string, int>? CountTokens => countTokens;

    public Func<string, IReadOnlyList<string>>? Tokenize => tokenize;
}

/// <summary>
/// The token-classification step over the Rampart model: tokenize, run the encoder, and keep each
/// token whose argmax label is not <c>O</c>. It mirrors the transformers.js pipeline the reference
/// runs, including its details — the first maximum wins a tie, special tokens (<c>[UNK]</c> included)
/// are skipped, and the softmax is evaluated in float32 like the typed array it runs on there.
/// </summary>
internal sealed class TokenClassifier(RampartModel model) : ITokenClassifier
{
    public RampartModel Model { get; } = model;

    public WordPieceTokenizer Tokenizer => Model.Tokenizer;

    public Func<string, int>? CountTokens => Tokenizer.CountTokens;

    public Func<string, IReadOnlyList<string>>? Tokenize => Tokenizer.Tokenize;

    /// <summary>Classify every token of <paramref name="text"/>; returns the non-<c>O</c> tokens in order.</summary>
    public IReadOnlyList<TokenEntity> Classify(string text)
    {
        var ids = Tokenizer.Encode(text);
        if (ids.Length > Model.MaxSequenceLength)
            throw new ArgumentException($"Text is {ids.Length} tokens; the model accepts {Model.MaxSequenceLength}. Use NerDetector, which windows long input.", nameof(text));
        return Classify(ids);
    }

    /// <summary>Classify a pre-tokenized sequence (already wrapped in <c>[CLS]</c> … <c>[SEP]</c>).</summary>
    public List<TokenEntity> Classify(ReadOnlySpan<int> inputIds)
    {
        var rows = new PooledList<NerRow>(16);
        try
        {
            AppendRows(inputIds, ref rows);
            var result = new List<TokenEntity>(rows.Count);
            foreach (var r in rows.AsSpan()) result.Add(new TokenEntity { Entity = r.Entity, Score = r.Score, Index = r.Index, Word = r.Word });
            return result;
        }
        finally
        {
            rows.Dispose();
        }
    }

    /// <summary>
    /// The allocation-free path <see cref="NerDetector"/> uses: tokenize <paramref name="text"/> once
    /// into <paramref name="ids"/> (with <c>[CLS]</c>/<c>[SEP]</c>; the caller disposes it), run the model
    /// into pooled logits, and append one row per non-<c>O</c>, non-special token. Labels and words are
    /// the model's own strings, so a row allocates nothing.
    /// </summary>
    internal void ClassifyWindow(ReadOnlySpan<char> text, ref PooledList<NerRow> rows, out PooledList<int> ids)
    {
        ids = new PooledList<int>(text.Length / 3 + 8);
        ids.Add(Tokenizer.ClsId);
        Tokenizer.EncodeTo(text, ref ids);
        ids.Add(Tokenizer.SepId);
        if (ids.Count > Model.MaxSequenceLength)
        {
            int count = ids.Count;
            ids.Dispose();
            throw new InvalidOperationException($"NER window of {count} tokens exceeds the model maximum of {Model.MaxSequenceLength}.");
        }
        AppendRows(ids.AsSpan(), ref rows);
    }

    private void AppendRows(ReadOnlySpan<int> inputIds, ref PooledList<NerRow> rows)
    {
        var labels = Model.Labels;
        int classes = labels.Count;
        var logits = ArrayPool<float>.Shared.Rent(Math.Max(1, inputIds.Length * classes));
        try
        {
            Model.ComputeLogits(inputIds, logits.AsSpan(0, inputIds.Length * classes));
            for (int j = 0; j < inputIds.Length; j++)
            {
                var row = logits.AsSpan(j * classes, classes);
                int best = ArgMax(row);
                var entity = labels[best];
                if (entity == "O") continue;
                if (Tokenizer.IsSpecial(inputIds[j])) continue;
                rows.Add(new NerRow(entity, null, Float32Softmax(row, best), j, Tokenizer.IdToToken(inputIds[j]), 0, 0, false));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(logits);
        }
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best]) best = i;
        }
        return best;
    }

    /// <summary>
    /// <c>softmax(Float32Array)[index]</c> as transformers.js computes it: each <c>exp(x - max)</c> is
    /// evaluated in double and stored to float32, summed in double, and the quotient stored to float32.
    /// </summary>
    internal static double Float32Softmax(ReadOnlySpan<float> row, int index)
    {
        float max = row[0];
        for (int i = 1; i < row.Length; i++) if (row[i] > max) max = row[i];
        double sum = 0;
        float target = 0;
        for (int i = 0; i < row.Length; i++)
        {
            float e = (float)Math.Exp((double)row[i] - max);
            sum += e;
            if (i == index) target = e;
        }
        return (float)(target / sum);
    }
}
