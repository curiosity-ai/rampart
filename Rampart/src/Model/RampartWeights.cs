using Catalyst.Rampart.Onnx;

namespace Catalyst.Rampart.Model;

/// <summary>A uint8 embedding table with one per-tensor scale and zero point (ONNX <c>DequantizeLinear</c>).</summary>
internal sealed class QuantizedEmbedding(byte[] data, int rows, int dim, float scale, byte zeroPoint)
{
    public int Rows { get; } = rows;
    public int Dim { get; } = dim;

    /// <summary>Adds <c>(q - zeroPoint) · scale</c> for row <paramref name="row"/> into <paramref name="destination"/>.</summary>
    public void Dequantize(int row, Span<float> destination)
    {
        if ((uint)row >= (uint)Rows) throw new ArgumentOutOfRangeException(nameof(row), $"Embedding row {row} is outside [0, {Rows}).");
        var q = data.AsSpan(row * Dim, Dim);
        for (int i = 0; i < q.Length; i++) destination[i] = (q[i] - zeroPoint) * scale;
    }
}

internal sealed record LayerNormWeights(float[] Gamma, float[] Beta, float Epsilon);

internal sealed record EncoderLayerWeights(
    Q4Linear Query,
    Q4Linear Key,
    Q4Linear Value,
    Q4Linear AttentionOutput,
    LayerNormWeights AttentionNorm,
    Q4Linear Intermediate,
    Q4Linear Output,
    LayerNormWeights OutputNorm);

/// <summary>
/// The weights of the shipped Rampart graph, bound by the tensor names the ONNX exporter gave each
/// node. Binding by graph structure keeps the quantized tensors exactly as published: nothing is
/// re-quantized or converted, so the bytes the C# engine reads are the bytes ONNX Runtime reads.
/// </summary>
internal sealed class RampartWeights
{
    public required RampartConfig Config { get; init; }
    public required QuantizedEmbedding WordEmbeddings { get; init; }
    public required QuantizedEmbedding PositionEmbeddings { get; init; }
    public required QuantizedEmbedding TokenTypeEmbeddings { get; init; }
    public required LayerNormWeights EmbeddingNorm { get; init; }
    public required EncoderLayerWeights[] Layers { get; init; }
    public required Q4Linear Classifier { get; init; }

    public static RampartWeights Bind(OnnxModel onnx, RampartConfig config)
    {
        var producers = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);
        var consumers = new Dictionary<string, List<OnnxNode>>(StringComparer.Ordinal);
        foreach (var node in onnx.Nodes)
        {
            foreach (var o in node.Outputs) producers[o] = node;
            foreach (var i in node.Inputs)
            {
                if (!consumers.TryGetValue(i, out var list)) consumers[i] = list = [];
                list.Add(node);
            }
        }

        OnnxNode Producer(string output, string opType)
        {
            if (!producers.TryGetValue(output, out var node) || node.OpType != opType)
                throw new InvalidDataException($"Expected a {opType} node producing '{output}' in the Rampart graph.");
            return node;
        }

        // A Linear is MatMulNBits(x, B, scales[, zero_points]) followed by Add(bias, ·).
        Q4Linear Linear(string matMulOutput)
        {
            var mm = Producer(matMulOutput, "MatMulNBits");
            if (mm.IntAttributes.GetValueOrDefault("bits", 4) != 4)
                throw new InvalidDataException($"Only 4-bit MatMulNBits is supported ('{matMulOutput}').");
            int k = checked((int)mm.IntAttributes["K"]);
            int n = checked((int)mm.IntAttributes["N"]);
            int blockSize = checked((int)mm.IntAttributes["block_size"]);
            var packed = onnx.GetTensor(mm.Inputs[1]).AsBytes();
            var scales = onnx.GetTensor(mm.Inputs[2]).AsFloats();
            byte[]? zeroPoints = mm.Inputs.Length > 3 && mm.Inputs[3].Length > 0 ? onnx.GetTensor(mm.Inputs[3]).AsBytes() : null;
            if (mm.Inputs.Length > 4 && mm.Inputs.Skip(4).Any(s => s.Length > 0))
                throw new InvalidDataException("MatMulNBits g_idx/bias inputs are not supported.");

            var add = consumers.GetValueOrDefault(matMulOutput)?.SingleOrDefault(c => c.OpType == "Add")
                ?? throw new InvalidDataException($"No bias Add after '{matMulOutput}'.");
            var biasName = add.Inputs.Single(i => i != matMulOutput);
            return new Q4Linear(n, k, blockSize, packed, scales, zeroPoints, onnx.GetTensor(biasName).AsFloats());
        }

        LayerNormWeights Norm(string output)
        {
            var ln = Producer(output, "LayerNormalization");
            float eps = ln.FloatAttributes.GetValueOrDefault("epsilon", 1e-5f);
            return new LayerNormWeights(onnx.GetTensor(ln.Inputs[1]).AsFloats(), onnx.GetTensor(ln.Inputs[2]).AsFloats(), eps);
        }

        // Embeddings: Gather(weight_quantized, ids) → DequantizeLinear(·, scale, zero_point).
        QuantizedEmbedding Embedding(string dequantOutput)
        {
            var dq = Producer(dequantOutput, "DequantizeLinear");
            var gather = Producer(dq.Inputs[0], "Gather");
            var table = onnx.GetTensor(gather.Inputs[0]);
            if (table.DataType != OnnxDataType.UInt8 || table.Dims.Length != 2)
                throw new InvalidDataException($"Embedding '{table.Name}' is not a 2-D uint8 table.");
            var scale = onnx.GetTensor(dq.Inputs[1]).ScalarFloat();
            var zero = onnx.GetTensor(dq.Inputs[2]).AsBytes()[0];
            return new QuantizedEmbedding(table.AsBytes(), (int)table.Dims[0], (int)table.Dims[1], scale, zero);
        }

        // The position ids are a constant arange sliced to the sequence length; the engine relies on that.
        var positionGather = Producer(OnnxNames.PositionGather, "Gather");
        var slice = Producer(positionGather.Inputs[1], "Slice");
        var positionIds = onnx.GetTensor(slice.Inputs[0]).AsInt64s();
        for (int i = 0; i < positionIds.Length; i++)
        {
            if (positionIds[i] != i) throw new InvalidDataException("Rampart graph position ids are not 0..n-1.");
        }

        var layers = new EncoderLayerWeights[config.NumHiddenLayers];
        for (int l = 0; l < layers.Length; l++)
        {
            layers[l] = new EncoderLayerWeights(
                Query: Linear(OnnxNames.MatMul(l, "attention/self/query")),
                Key: Linear(OnnxNames.MatMul(l, "attention/self/key")),
                Value: Linear(OnnxNames.MatMul(l, "attention/self/value")),
                AttentionOutput: Linear(OnnxNames.MatMul(l, "attention/output/dense")),
                AttentionNorm: Norm(OnnxNames.AttentionNorm(l)),
                Intermediate: Linear(OnnxNames.MatMul(l, "intermediate/dense")),
                Output: Linear(OnnxNames.MatMul(l, "output/dense")),
                OutputNorm: Norm(OnnxNames.OutputNorm(l)));
        }

        var weights = new RampartWeights
        {
            Config = config,
            WordEmbeddings = Embedding(OnnxNames.WordEmbeddings),
            PositionEmbeddings = Embedding(OnnxNames.PositionEmbeddings),
            TokenTypeEmbeddings = Embedding(OnnxNames.TokenTypeEmbeddings),
            EmbeddingNorm = Norm(OnnxNames.EmbeddingNorm),
            Layers = layers,
            Classifier = Linear(OnnxNames.ClassifierMatMul),
        };
        weights.Validate();
        return weights;
    }

    private void Validate()
    {
        int h = Config.HiddenSize;
        if (WordEmbeddings.Dim != h || PositionEmbeddings.Dim != h || TokenTypeEmbeddings.Dim != h)
            throw new InvalidDataException("Embedding width does not match hidden_size.");
        if (WordEmbeddings.Rows != Config.VocabSize) throw new InvalidDataException("Word embedding rows do not match vocab_size.");
        if (h % Config.NumAttentionHeads != 0) throw new InvalidDataException("hidden_size is not divisible by num_attention_heads.");
        foreach (var layer in Layers)
        {
            foreach (var lin in new[] { layer.Query, layer.Key, layer.Value, layer.AttentionOutput })
            {
                if (lin.K != h || lin.N != h) throw new InvalidDataException("Attention projection has the wrong shape.");
            }
            if (layer.Intermediate.K != h || layer.Intermediate.N != Config.IntermediateSize) throw new InvalidDataException("Intermediate projection has the wrong shape.");
            if (layer.Output.K != Config.IntermediateSize || layer.Output.N != h) throw new InvalidDataException("Output projection has the wrong shape.");
        }
        if (Classifier.K != h || Classifier.N != Config.Labels.Count) throw new InvalidDataException("Classifier does not match the label set.");
    }
}

/// <summary>Tensor names in the shipped graph; the engine's trace points use the same names.</summary>
internal static class OnnxNames
{
    public const string WordEmbeddings = "/bert/embeddings/word_embeddings/Gather_output_0";
    public const string PositionEmbeddings = "/bert/embeddings/position_embeddings/Gather_output_0";
    public const string TokenTypeEmbeddings = "/bert/embeddings/token_type_embeddings/Gather_output_0";
    public const string PositionGather = "/bert/embeddings/position_embeddings/Gather_output_0_quantized";
    public const string EmbeddingSum = "/bert/embeddings/Add_1_output_0";
    public const string EmbeddingNorm = "/bert/embeddings/LayerNorm/LayerNormalization_output_0";
    public const string ClassifierMatMul = "/classifier/MatMul_output_0";
    public const string Logits = "logits";

    private static string Layer(int l) => $"/bert/encoder/layer.{l}";
    public static string MatMul(int l, string module) => $"{Layer(l)}/{module}/MatMul_output_0";
    public static string Query(int l) => $"{Layer(l)}/attention/self/query/Add_output_0";
    public static string Key(int l) => $"{Layer(l)}/attention/self/key/Add_output_0";
    public static string Value(int l) => $"{Layer(l)}/attention/self/value/Add_output_0";
    public static string AttentionProbs(int l) => $"{Layer(l)}/attention/self/Softmax_output_0";
    public static string Context(int l) => $"{Layer(l)}/attention/self/Reshape_3_output_0";
    public static string AttentionDense(int l) => $"{Layer(l)}/attention/output/dense/Add_output_0";
    public static string AttentionNorm(int l) => $"{Layer(l)}/attention/output/LayerNorm/LayerNormalization_output_0";
    public static string IntermediateDense(int l) => $"{Layer(l)}/intermediate/dense/Add_output_0";
    public static string IntermediateGelu(int l) => $"{Layer(l)}/intermediate/intermediate_act_fn/Mul_1_output_0";
    public static string OutputDense(int l) => $"{Layer(l)}/output/dense/Add_output_0";
    public static string OutputNorm(int l) => $"{Layer(l)}/output/LayerNorm/LayerNormalization_output_0";
}
