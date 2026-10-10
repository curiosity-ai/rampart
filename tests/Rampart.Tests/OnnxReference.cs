using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Rampart.Model;
using Rampart.Onnx;

namespace Rampart.Tests;

/// <summary>
/// The reference implementation: the published <c>model_q4.onnx</c>, byte-for-byte the resource the
/// library embeds, executed by ONNX Runtime. Every intermediate tensor the C# engine traces is added
/// to the graph as an extra output, so the two can be compared layer by layer.
/// </summary>
public sealed class OnnxReference : IDisposable
{
    private static readonly Lazy<OnnxReference> _shared = new(() => new OnnxReference());
    public static OnnxReference Shared => _shared.Value;

    private readonly InferenceSession _literal;
    private readonly InferenceSession _optimized;

    /// <summary>Names of every traced tensor, in graph order.</summary>
    public IReadOnlyList<string> TracePoints { get; }

    private OnnxReference()
    {
        var bytes = RampartModel.ReadResource(typeof(RampartModel).Assembly, "Rampart.model_q4.onnx");
        var cfg = RampartModel.Default.Config;
        var names = new List<string> { OnnxNames.EmbeddingSum, OnnxNames.EmbeddingNorm };
        for (int l = 0; l < cfg.NumHiddenLayers; l++)
        {
            names.AddRange([
                OnnxNames.Query(l), OnnxNames.Key(l), OnnxNames.Value(l), OnnxNames.AttentionProbs(l), OnnxNames.Context(l),
                OnnxNames.AttentionDense(l), OnnxNames.AttentionNorm(l), OnnxNames.IntermediateDense(l), OnnxNames.IntermediateGelu(l),
                OnnxNames.OutputDense(l), OnnxNames.OutputNorm(l),
            ]);
        }
        names.Add(OnnxNames.Logits);
        TracePoints = names;

        // Graph exactly as written (no fusions), with every trace point exposed.
        var literal = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL, IntraOpNumThreads = 1 };
        _literal = new InferenceSession(AddGraphOutputs(bytes, names.Where(n => n != OnnxNames.Logits)), literal);
        // What a deployment runs: the stock graph with ORT's full optimization.
        var optimized = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        // Idle intra-op threads would otherwise spin after each run and steal cores from whatever runs next.
        optimized.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        _optimized = new InferenceSession(bytes, optimized);
    }

    /// <summary>Runs the literal graph and returns every trace point by name.</summary>
    public Dictionary<string, float[]> RunTraced(int[] ids)
    {
        using var results = _literal.Run(Inputs(ids), TracePoints);
        return results.ToDictionary(r => r.Name, r => r.AsTensor<float>().ToArray());
    }

    /// <summary>Logits from the optimized graph.</summary>
    public float[] RunLogits(int[] ids)
    {
        using var results = _optimized.Run(Inputs(ids));
        return results.First(r => r.Name == "logits").AsTensor<float>().ToArray();
    }

    private static List<NamedOnnxValue> Inputs(int[] ids)
    {
        var shape = new[] { 1, ids.Length };
        var inputIds = new DenseTensor<long>(ids.Select(i => (long)i).ToArray(), shape);
        var mask = new DenseTensor<long>(Enumerable.Repeat(1L, ids.Length).ToArray(), shape);
        var types = new DenseTensor<long>(new long[ids.Length], shape);
        return
        [
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", mask),
            NamedOnnxValue.CreateFromTensor("token_type_ids", types),
        ];
    }

    /// <summary>
    /// Re-encodes a <c>ModelProto</c> with extra float graph outputs appended to its <c>GraphProto</c>
    /// (field 7, outputs are field 12). Every other byte is copied verbatim.
    /// </summary>
    internal static byte[] AddGraphOutputs(byte[] model, IEnumerable<string> outputNames)
    {
        var extra = new MemoryStream();
        foreach (var name in outputNames)
        {
            var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
            var valueInfo = new MemoryStream();
            valueInfo.WriteByte(0x0A); WriteVarint(valueInfo, (ulong)nameBytes.Length); valueInfo.Write(nameBytes);
            // type { tensor_type { elem_type: FLOAT } }
            valueInfo.Write([0x12, 0x04, 0x0A, 0x02, 0x08, 0x01]);
            extra.WriteByte(0x62); WriteVarint(extra, (ulong)valueInfo.Length); valueInfo.WriteTo(extra);
        }

        var output = new MemoryStream();
        var reader = new ProtoReader(model);
        while (true)
        {
            int fieldStart = reader.Position;
            if (!reader.TryReadTag(out var field, out var wire)) break;
            if (field == 7 && wire == WireType.LengthDelimited)
            {
                var graph = reader.ReadBytes();
                output.WriteByte(0x3A);
                WriteVarint(output, (ulong)(graph.Length + extra.Length));
                output.Write(graph);
                extra.WriteTo(output);
            }
            else
            {
                reader.Skip(wire);
                output.Write(model, fieldStart, reader.Position - fieldStart);
            }
        }
        return output.ToArray();
    }

    private static void WriteVarint(Stream s, ulong v)
    {
        while (v >= 0x80)
        {
            s.WriteByte((byte)(v | 0x80));
            v >>= 7;
        }
        s.WriteByte((byte)v);
    }

    public void Dispose()
    {
        _literal.Dispose();
        _optimized.Dispose();
    }
}
