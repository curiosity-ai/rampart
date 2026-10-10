using System.Buffers;
using System.Numerics.Tensors;
using System.Reflection;
using Rampart.Onnx;
using Rampart.Tokenization;

namespace Rampart.Model;

/// <summary>Loading options for <see cref="RampartModel"/>.</summary>
public sealed class RampartModelOptions
{
    /// <summary>
    /// Keep the linear weights in their shipped 4-bit form (~5 MB) and dequantize them tile by tile
    /// on every forward pass, instead of expanding them once at load into float32 (~42 MB). Results
    /// are identical; expanded weights are faster, markedly so for short inputs.
    /// </summary>
    public bool KeepWeightsQuantized { get; init; }
}

/// <summary>Receives every intermediate tensor of a forward pass, named as in the ONNX graph.</summary>
internal interface IForwardTrace
{
    void Record(string onnxName, ReadOnlySpan<float> data);
}

/// <summary>
/// The Rampart token classifier: a 6-layer, 384-wide MiniLM (BERT) encoder with a 35-way BIO head,
/// run natively from the published 4-bit ONNX weights (<c>onnx/model_q4.onnx</c>). No ONNX Runtime:
/// the graph is a fixed BERT, so this class evaluates it directly — int8 embedding lookup,
/// <c>MatMulNBits</c> linears on a register-blocked SIMD GEMM, exact-erf GELU and post-norm
/// residual blocks.
/// <para>Instances are immutable after load and safe to use from multiple threads.</para>
/// </summary>
public sealed class RampartModel
{
    private static readonly Lazy<RampartModel> _default = new(() => Load(), LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly RampartWeights _weights;

    /// <summary>The encoder configuration (from the shipped <c>config.json</c>).</summary>
    public RampartConfig Config => _weights.Config;

    /// <summary>The WordPiece tokenizer that matches this model's vocabulary.</summary>
    public WordPieceTokenizer Tokenizer { get; }

    /// <summary>BIO labels by class index.</summary>
    public IReadOnlyList<string> Labels => _weights.Config.Labels;

    /// <summary>Maximum sequence length, including <c>[CLS]</c> and <c>[SEP]</c>.</summary>
    public int MaxSequenceLength => _weights.Config.MaxPositionEmbeddings;

    private RampartModel(RampartWeights weights, WordPieceTokenizer tokenizer)
    {
        _weights = weights;
        Tokenizer = tokenizer;
    }

    /// <summary>The model embedded in this assembly, loaded once and shared.</summary>
    public static RampartModel Default => _default.Value;

    /// <summary>Loads the embedded model (a fresh instance; prefer <see cref="Default"/>).</summary>
    public static RampartModel Load(RampartModelOptions? options = null)
    {
        var assembly = typeof(RampartModel).Assembly;
        return Load(ReadResource(assembly, "Rampart.model_q4.onnx"), ReadResource(assembly, "Rampart.vocab.txt"), ReadResource(assembly, "Rampart.config.json"), options);
    }

    /// <summary>Loads a Rampart-compatible model from the three published files.</summary>
    public static RampartModel Load(ReadOnlySpan<byte> onnxModel, ReadOnlySpan<byte> vocabTxt, ReadOnlySpan<byte> configJson, RampartModelOptions? options = null)
    {
        var config = RampartConfig.Parse(configJson);
        var onnx = OnnxModel.Parse(onnxModel);
        var weights = RampartWeights.Bind(onnx, config);
        if (options?.KeepWeightsQuantized != true) weights.ExpandWeights();
        var tokenizer = WordPieceTokenizer.FromVocab(vocabTxt);
        if (tokenizer.VocabularySize != config.VocabSize) throw new InvalidDataException("vocab.txt does not match config vocab_size.");
        return new RampartModel(weights, tokenizer);
    }

    /// <summary>Loads a model from a directory laid out like the Hugging Face repository.</summary>
    public static RampartModel LoadFromDirectory(string directory, RampartModelOptions? options = null) => Load(
        File.ReadAllBytes(Path.Combine(directory, "onnx", "model_q4.onnx")),
        File.ReadAllBytes(Path.Combine(directory, "vocab.txt")),
        File.ReadAllBytes(Path.Combine(directory, "config.json")),
        options);

    internal static byte[] ReadResource(Assembly assembly, string name)
    {
        using var stream = assembly.GetManifestResourceStream(name) ?? throw new InvalidOperationException($"Embedded resource '{name}' is missing.");
        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    /// <summary>
    /// Runs the encoder over one unpadded sequence of token ids (already wrapped in <c>[CLS]</c> …
    /// <c>[SEP]</c>) and returns row-major logits of shape <c>[ids.Length, Labels.Count]</c>.
    /// </summary>
    public float[] ComputeLogits(ReadOnlySpan<int> inputIds) => ComputeLogits(inputIds, trace: null);

    internal float[] ComputeLogits(ReadOnlySpan<int> inputIds, IForwardTrace? trace)
    {
        var cfg = _weights.Config;
        int tokens = inputIds.Length;
        int hiddenSize = cfg.HiddenSize;
        int interSize = cfg.IntermediateSize;
        int labels = cfg.Labels.Count;
        if (tokens == 0) return [];
        if (tokens > cfg.MaxPositionEmbeddings)
            throw new ArgumentException($"Sequence of {tokens} tokens exceeds the model maximum of {cfg.MaxPositionEmbeddings}.", nameof(inputIds));

        var pool = ArrayPool<float>.Shared;
        int th = tokens * hiddenSize;
        var hidden = pool.Rent(th);
        var attnOut = pool.Rent(th);
        var q = pool.Rent(th);
        var k = pool.Rent(th);
        var v = pool.Rent(th);
        var context = pool.Rent(th);
        var dense = pool.Rent(th);
        var inter = pool.Rent(tokens * interSize);
        var probs = pool.Rent(cfg.NumAttentionHeads * tokens * tokens);
        try
        {
            var h = hidden.AsSpan(0, th);
            Embed(inputIds, h);
            trace?.Record(OnnxNames.EmbeddingSum, h);
            var en = _weights.EmbeddingNorm;
            Kernels.LayerNorm(h, tokens, en.Gamma, en.Beta, en.Epsilon);
            trace?.Record(OnnxNames.EmbeddingNorm, h);

            for (int l = 0; l < _weights.Layers.Length; l++)
            {
                var layer = _weights.Layers[l];
                var qs = q.AsSpan(0, th);
                var ks = k.AsSpan(0, th);
                var vs = v.AsSpan(0, th);
                layer.Query.Forward(h, tokens, qs);
                layer.Key.Forward(h, tokens, ks);
                layer.Value.Forward(h, tokens, vs);
                trace?.Record(OnnxNames.Query(l), qs);
                trace?.Record(OnnxNames.Key(l), ks);
                trace?.Record(OnnxNames.Value(l), vs);

                var ctx = context.AsSpan(0, th);
                var p = probs.AsSpan(0, cfg.NumAttentionHeads * tokens * tokens);
                Attention(qs, ks, vs, tokens, p, ctx);
                trace?.Record(OnnxNames.AttentionProbs(l), p);
                trace?.Record(OnnxNames.Context(l), ctx);

                // Post-norm residual: LayerNorm(dense(ctx) + h).
                var d = dense.AsSpan(0, th);
                layer.AttentionOutput.Forward(ctx, tokens, d);
                trace?.Record(OnnxNames.AttentionDense(l), d);
                var a = attnOut.AsSpan(0, th);
                TensorPrimitives.Add(d, h, a);
                Kernels.LayerNorm(a, tokens, layer.AttentionNorm.Gamma, layer.AttentionNorm.Beta, layer.AttentionNorm.Epsilon);
                trace?.Record(OnnxNames.AttentionNorm(l), a);

                var i = inter.AsSpan(0, tokens * interSize);
                layer.Intermediate.Forward(a, tokens, i);
                trace?.Record(OnnxNames.IntermediateDense(l), i);
                Kernels.Gelu(i);
                trace?.Record(OnnxNames.IntermediateGelu(l), i);

                layer.Output.Forward(i, tokens, d);
                trace?.Record(OnnxNames.OutputDense(l), d);
                TensorPrimitives.Add(d, a, h);
                Kernels.LayerNorm(h, tokens, layer.OutputNorm.Gamma, layer.OutputNorm.Beta, layer.OutputNorm.Epsilon);
                trace?.Record(OnnxNames.OutputNorm(l), h);
            }

            var logits = new float[tokens * labels];
            _weights.Classifier.Forward(h, tokens, logits);
            trace?.Record(OnnxNames.Logits, logits);
            return logits;
        }
        finally
        {
            pool.Return(hidden);
            pool.Return(attnOut);
            pool.Return(q);
            pool.Return(k);
            pool.Return(v);
            pool.Return(context);
            pool.Return(dense);
            pool.Return(inter);
            pool.Return(probs);
        }
    }

    /// <summary>word + token_type(0), then + position — the order the exported graph adds them in.</summary>
    private void Embed(ReadOnlySpan<int> ids, Span<float> destination)
    {
        int width = _weights.Config.HiddenSize;
        Span<float> typeRow = stackalloc float[width];
        Span<float> scratch = stackalloc float[width];
        _weights.TokenTypeEmbeddings.Dequantize(0, typeRow);
        for (int t = 0; t < ids.Length; t++)
        {
            var row = destination.Slice(t * width, width);
            _weights.WordEmbeddings.Dequantize(ids[t], row);
            TensorPrimitives.Add(row, typeRow, row);
            _weights.PositionEmbeddings.Dequantize(t, scratch);
            TensorPrimitives.Add(row, scratch, row);
        }
    }

    /// <summary>
    /// Multi-head self-attention over an unpadded sequence. Like the exported graph, Q and K are each
    /// scaled by <c>√(1/√d)</c> before their product, rather than the scores by <c>1/√d</c>.
    /// <paramref name="probs"/> receives the softmax weights laid out <c>[head, query, key]</c>.
    /// <para>
    /// Work is split into (head, block of query rows) tasks: a task's scores, softmax and weighted sum
    /// of V all happen while its slice of the probabilities is still in cache.
    /// </para>
    /// </summary>
    private unsafe void Attention(Span<float> q, Span<float> k, ReadOnlySpan<float> v, int tokens, Span<float> probs, Span<float> context)
    {
        var cfg = _weights.Config;
        int heads = cfg.NumAttentionHeads;
        int width = cfg.HiddenSize;
        int headDim = width / heads;
        float scale = MathF.Sqrt(1f / MathF.Sqrt(headDim));
        TensorPrimitives.Multiply(q, scale, q);
        TensorPrimitives.Multiply(k, scale, k);

        // K transposed per head: kt[h][d][j] = k[j][h·d + d], the B operand of scores = Q_h · K_hᵀ.
        var kt = ArrayPool<float>.Shared.Rent(tokens * width);
        try
        {
            for (int j = 0; j < tokens; j++)
            {
                var krow = k.Slice(j * width, width);
                for (int h = 0; h < heads; h++)
                {
                    var src = krow.Slice(h * headDim, headDim);
                    int dst = h * headDim * tokens + j;
                    for (int d = 0; d < headDim; d++) kt[dst + d * tokens] = src[d];
                }
            }

            int rowBlocks = (tokens + QueryBlock - 1) / QueryBlock;
            int tasks = heads * rowBlocks;
            fixed (float* pq = q)
            fixed (float* pv = v)
            fixed (float* pp = probs)
            fixed (float* pc = context)
            fixed (float* pk = kt)
            {
                nint qp = (nint)pq, vp = (nint)pv, ppp = (nint)pp, cp = (nint)pc, ktp = (nint)pk;
                int len = tokens * width;

                void Task(int task)
                {
                    int head = task / rowBlocks;
                    int i0 = task % rowBlocks * QueryBlock;
                    int rows = Math.Min(QueryBlock, tokens - i0);
                    int off = head * headDim;
                    var qs = new ReadOnlySpan<float>((void*)qp, len);
                    var vs = new ReadOnlySpan<float>((void*)vp, len);
                    var cs = new Span<float>((void*)cp, len);
                    var kth = new ReadOnlySpan<float>((void*)(ktp + (nint)head * headDim * tokens * sizeof(float)), headDim * tokens);
                    var ps = new Span<float>((void*)(ppp + ((nint)head * tokens + i0) * tokens * sizeof(float)), rows * tokens);

                    Gemm.Multiply(qs[(i0 * width + off)..], width, kth, tokens, ps, tokens, rows, tokens, headDim);
                    for (int i = 0; i < rows; i++) Kernels.SoftmaxInPlace(ps.Slice(i * tokens, tokens));
                    // context rows = P · V_h, reading V_h in place (row stride = hidden width).
                    Gemm.Multiply(ps, tokens, vs[off..], width, cs[(i0 * width + off)..], width, rows, headDim, tokens);
                }

                if ((long)tokens * tokens * width < 200_000)
                {
                    for (int task = 0; task < tasks; task++) Task(task);
                }
                else
                {
                    Parallel.For(0, tasks, Task);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(kt);
        }
    }

    /// <summary>Query rows per attention task (a multiple of the GEMM's 6-row block).</summary>
    private const int QueryBlock = 48;
}
