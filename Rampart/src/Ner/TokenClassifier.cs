using Rampart.Model;
using Rampart.Tokenization;

namespace Rampart.Ner;

/// <summary>
/// One token-classification row, shaped like a transformers.js <c>token-classification</c> result
/// (no aggregation): the argmax BIO label of a non-<c>O</c> token.
/// </summary>
/// <param name="Entity">The BIO label (<c>B-GIVEN_NAME</c>).</param>
/// <param name="Score">Softmax probability of that label.</param>
/// <param name="Index">Token index in <c>[CLS] + content + [SEP]</c>.</param>
/// <param name="Word">The WordPiece token (<c>##</c>-prefixed for a continuation).</param>
public readonly record struct TokenEntity(string Entity, double Score, int Index, string Word);

/// <summary>
/// The token-classification step: tokenize, run the encoder, and keep each token whose argmax label
/// is not <c>O</c>. It mirrors the transformers.js pipeline the reference runs, including its
/// details — the first maximum wins a tie, special tokens (<c>[UNK]</c> included) are skipped, and
/// the softmax is evaluated in float32 like the typed array it runs on there.
/// </summary>
public sealed class TokenClassifier(RampartModel model)
{
    public RampartModel Model { get; } = model;

    public WordPieceTokenizer Tokenizer => Model.Tokenizer;

    /// <summary>Classify every token of <paramref name="text"/>; returns the non-<c>O</c> tokens in order.</summary>
    public List<TokenEntity> Classify(string text)
    {
        var tokens = Tokenizer.Tokenize(text);
        var ids = new int[tokens.Count + 2];
        ids[0] = Tokenizer.ClsId;
        for (int i = 0; i < tokens.Count; i++) ids[i + 1] = Tokenizer.TokenToId(tokens[i]);
        ids[^1] = Tokenizer.SepId;
        return Classify(ids);
    }

    /// <summary>Classify a pre-tokenized sequence (already wrapped in <c>[CLS]</c> … <c>[SEP]</c>).</summary>
    public List<TokenEntity> Classify(ReadOnlySpan<int> inputIds)
    {
        var labels = Model.Labels;
        int classes = labels.Count;
        var logits = Model.ComputeLogits(inputIds);
        var result = new List<TokenEntity>();
        for (int j = 0; j < inputIds.Length; j++)
        {
            var row = logits.AsSpan(j * classes, classes);
            int best = ArgMax(row);
            var entity = labels[best];
            if (entity == "O") continue;
            if (Tokenizer.IsSpecial(inputIds[j])) continue;
            result.Add(new TokenEntity(entity, Float32Softmax(row, best), j, Tokenizer.IdToToken(inputIds[j])));
        }
        return result;
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
