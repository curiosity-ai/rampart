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
    /// Cores one forward pass may use. Defaults to all of them; 1 runs every kernel on the calling
    /// thread, which suits a server that already runs many requests in parallel.
    /// </summary>
    public int MaxDegreeOfParallelism { get; init; } = Environment.ProcessorCount;
}

/// <summary>Receives every intermediate tensor of a forward pass, named as in the ONNX graph.</summary>
internal interface IForwardTrace
{
    void Record(string onnxName, ReadOnlySpan<float> data);

    /// <summary>
    /// False for a trace that only wants to know when each stage finishes (a profiler): the pass then
    /// runs exactly as untraced — fused GELU, no materialized attention probabilities — and the
    /// affected trace points receive empty spans.
    /// </summary>
    bool CapturesTensors => true;
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

    internal RampartWeights Weights => _weights;
    private readonly int _parallelism;

    /// <summary>The encoder configuration (from the shipped <c>config.json</c>).</summary>
    public RampartConfig Config => _weights.Config;

    /// <summary>The WordPiece tokenizer that matches this model's vocabulary.</summary>
    public WordPieceTokenizer Tokenizer { get; }

    /// <summary>BIO labels by class index.</summary>
    public IReadOnlyList<string> Labels => _weights.Config.Labels;

    /// <summary>Maximum sequence length, including <c>[CLS]</c> and <c>[SEP]</c>.</summary>
    public int MaxSequenceLength => _weights.Config.MaxPositionEmbeddings;

    private RampartModel(RampartWeights weights, WordPieceTokenizer tokenizer, int parallelism)
    {
        _parallelism = Math.Max(1, parallelism);
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
        var tokenizer = WordPieceTokenizer.FromVocab(vocabTxt);
        if (tokenizer.VocabularySize != config.VocabSize) throw new InvalidDataException("vocab.txt does not match config vocab_size.");
        return new RampartModel(weights, tokenizer, options?.MaxDegreeOfParallelism ?? Environment.ProcessorCount);
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
        // The full [heads, tokens, tokens] probabilities exist only for a trace; otherwise each attention
        // task keeps its slice in an L2-sized scratch buffer.
        bool capture = trace?.CapturesTensors == true;
        var probs = capture ? pool.Rent(cfg.NumAttentionHeads * tokens * tokens) : null;
        try
        {
            var h = hidden.AsSpan(0, th);
            Embed(inputIds, h);
            trace?.Record(OnnxNames.EmbeddingSum, h);
            var en = _weights.EmbeddingNorm;
            Kernels.LayerNorm(h, tokens, en.Gamma, en.Beta, en.Epsilon, _parallelism);
            trace?.Record(OnnxNames.EmbeddingNorm, h);

            for (int l = 0; l < _weights.Layers.Length; l++)
            {
                var layer = _weights.Layers[l];
                var qs = q.AsSpan(0, th);
                var ks = k.AsSpan(0, th);
                var vs = v.AsSpan(0, th);
                layer.Query.Forward(h, tokens, qs, _parallelism);
                layer.Key.Forward(h, tokens, ks, _parallelism);
                layer.Value.Forward(h, tokens, vs, _parallelism);
                trace?.Record(OnnxNames.Query(l), qs);
                trace?.Record(OnnxNames.Key(l), ks);
                trace?.Record(OnnxNames.Value(l), vs);

                var ctx = context.AsSpan(0, th);
                var p = probs is null ? default : probs.AsSpan(0, cfg.NumAttentionHeads * tokens * tokens);
                Attention(qs, ks, vs, tokens, p, ctx);
                trace?.Record(OnnxNames.AttentionProbs(l), p);
                trace?.Record(OnnxNames.Context(l), ctx);

                // Post-norm residual: LayerNorm(dense(ctx) + h).
                var d = dense.AsSpan(0, th);
                layer.AttentionOutput.Forward(ctx, tokens, d, _parallelism);
                trace?.Record(OnnxNames.AttentionDense(l), d);
                var a = attnOut.AsSpan(0, th);
                Kernels.AddLayerNorm(d, h, a, tokens, layer.AttentionNorm.Gamma, layer.AttentionNorm.Beta, layer.AttentionNorm.Epsilon);
                trace?.Record(OnnxNames.AttentionNorm(l), a);

                var i = inter.AsSpan(0, tokens * interSize);
                if (!capture)
                {
                    layer.Intermediate.Forward(a, tokens, i, _parallelism);
                    Kernels.Gelu(i, _parallelism);
                    trace?.Record(OnnxNames.IntermediateGelu(l), i);
                }
                else
                {
                    layer.Intermediate.Forward(a, tokens, i, _parallelism);
                    trace!.Record(OnnxNames.IntermediateDense(l), i);
                    Kernels.Gelu(i, _parallelism);
                    trace.Record(OnnxNames.IntermediateGelu(l), i);
                }

                layer.Output.Forward(i, tokens, d, _parallelism);
                trace?.Record(OnnxNames.OutputDense(l), d);
                Kernels.AddLayerNorm(d, a, h, tokens, layer.OutputNorm.Gamma, layer.OutputNorm.Beta, layer.OutputNorm.Epsilon);
                trace?.Record(OnnxNames.OutputNorm(l), h);
            }

            var logits = new float[tokens * labels];
            _weights.Classifier.Forward(h, tokens, logits, _parallelism);
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
            if (probs is not null) pool.Return(probs);
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
    /// <paramref name="probs"/>, when not empty, receives the softmax weights laid out <c>[head, query, key]</c>.
    /// <para>
    /// Work is split into (head, block of query rows) tasks: a task's scores, softmax and weighted sum
    /// of V all happen while its slice of the probabilities is still in cache.
    /// </para>
    /// </summary>
    private unsafe void Attention(Span<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int tokens, Span<float> probs, Span<float> context)
    {
        var cfg = _weights.Config;
        int heads = cfg.NumAttentionHeads;
        int width = cfg.HiddenSize;
        int headDim = width / heads;
        float scale = MathF.Sqrt(1f / MathF.Sqrt(headDim));
        TensorPrimitives.Multiply(q, scale, q);

        // Per-head panels, packed once per layer and shared by every query block of the head:
        // kt[h] = (K_h · scale)ᵀ, headDim × tokens — the B operand of scores = Q_h · K_hᵀ;
        // vh[h] = V_h, tokens × headDim contiguous — the B operand of context = P · V_h.
        int panel = tokens * headDim;
        var kt = ArrayPool<float>.Shared.Rent(heads * panel);
        var vh = ArrayPool<float>.Shared.Rent(heads * panel);
        try
        {
            for (int j = 0; j < tokens; j++)
            {
                var krow = k.Slice(j * width, width);
                var vrow = v.Slice(j * width, width);
                for (int h = 0; h < heads; h++)
                {
                    var src = krow.Slice(h * headDim, headDim);
                    int dst = h * panel + j;
                    for (int d = 0; d < headDim; d++) kt[dst + d * tokens] = src[d] * scale;
                    vrow.Slice(h * headDim, headDim).CopyTo(vh.AsSpan(h * panel + j * headDim, headDim));
                }
            }

            int block = _parallelism <= 1 ? QueryBlockSerial : QueryBlockParallel;
            int rowBlocks = (tokens + block - 1) / block;
            fixed (float* pq = q)
            fixed (float* pp = probs)
            fixed (float* pc = context)
            fixed (float* pk = kt)
            fixed (float* pv = vh)
            {
                var job = new AttentionJob((nint)pq, (nint)pk, (nint)pv, probs.IsEmpty ? 0 : (nint)pp, (nint)pc, tokens, width, headDim, block, rowBlocks);
                int degree = _parallelism <= 1 || (long)tokens * tokens * width < 200_000 ? 1 : _parallelism;
                Parallelism.For(heads * rowBlocks, degree, ref job);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(kt);
            ArrayPool<float>.Shared.Return(vh);
        }
    }

    /// <summary>Query rows per attention task (multiples of the GEMM's 12-row block): one core keeps a
    /// larger slice of probabilities in L2; several cores want more, smaller tasks.</summary>
    private const int QueryBlockSerial = 96, QueryBlockParallel = 48;

    /// <summary>One (head, block of queries) of attention: scores, softmax, weighted sum of V.</summary>
    private readonly unsafe struct AttentionJob(nint q, nint kt, nint vh, nint probs, nint context, int tokens, int width, int headDim, int block, int rowBlocks) : IParallelJob
    {
        public void Run(int from, int to)
        {
            int len = tokens * width, panel = tokens * headDim;
            var qs = new ReadOnlySpan<float>((void*)q, len);
            var cs = new Span<float>((void*)context, len);
            var scratch = probs == 0 ? ArrayPool<float>.Shared.Rent(block * tokens) : null;
            try
            {
            for (int task = from; task < to; task++)
            {
                int head = task / rowBlocks;
                int i0 = task % rowBlocks * block;
                int rows = Math.Min(block, tokens - i0);
                int off = head * headDim;
                var kth = new ReadOnlySpan<float>((void*)(kt + (nint)head * panel * sizeof(float)), panel);
                var vhh = new ReadOnlySpan<float>((void*)(vh + (nint)head * panel * sizeof(float)), panel);
                var ps = scratch is not null
                    ? scratch.AsSpan(0, rows * tokens)
                    : new Span<float>((void*)(probs + ((nint)head * tokens + i0) * tokens * sizeof(float)), rows * tokens);

                Gemm.Multiply(qs[(i0 * width + off)..], width, kth, tokens, ps, tokens, rows, tokens, headDim);
                for (int i = 0; i < rows; i++) Kernels.SoftmaxInPlace(ps.Slice(i * tokens, tokens));
                Gemm.Multiply(ps, tokens, vhh, headDim, cs[(i0 * width + off)..], width, rows, headDim, tokens);
            }
            }
            finally
            {
                if (scratch is not null) ArrayPool<float>.Shared.Return(scratch);
            }
        }
    }
}
