using Rampart.Model;

namespace Rampart.Tests;

/// <summary>
/// The C# encoder against ONNX Runtime running the published graph, compared at every intermediate
/// tensor: embeddings, Q/K/V, attention probabilities, context, both residual blocks, GELU, logits.
/// </summary>
public sealed class LayerParityTests
{
    // Observed worst case over the corpus is ~6e-6 (max, relative to the tensor's peak) and ~2.5e-6
    // (RMS): float32 rounding from a different summation order. The bounds leave ~10x headroom; a
    // wrong weight, nibble order, scale or op would be off by orders of magnitude more.
    private const double MaxRelative = 5e-5;
    private const double RmsRelative = 2e-5;

    public static TheoryData<string> Inputs()
    {
        var data = new TheoryData<string>();
        foreach (var s in Corpus.Sentences) data.Add(s);
        return data;
    }

    [Theory]
    [MemberData(nameof(Inputs))]
    public void EveryLayerMatchesOnnxRuntime(string text) => CompareLayers(RampartModel.Default.Tokenizer.Encode(text));

    [Fact]
    public void FullContextWindowMatchesOnnxRuntime()
    {
        // 512 tokens: the longest sequence the model accepts, and the parallel attention/matmul paths.
        var tokenizer = RampartModel.Default.Tokenizer;
        var content = tokenizer.Encode(Corpus.Long(), addSpecialTokens: false);
        int[] ids = [tokenizer.ClsId, .. content.AsSpan(0, 510), tokenizer.SepId];
        Assert.Equal(512, ids.Length);
        CompareLayers(ids);
    }

    [Fact]
    public void RejectsSequencesLongerThanTheModel()
    {
        var model = RampartModel.Default;
        Assert.Throws<ArgumentException>(() => model.ComputeLogits(new int[513]));
        Assert.Empty(model.ComputeLogits([]));
    }

    private static void CompareLayers(int[] ids)
    {
        var model = RampartModel.Default;
        var recorder = new TraceRecorder();
        model.ComputeLogits(ids, recorder);
        var reference = OnnxReference.Shared.RunTraced(ids);

        var output = TestContext.Current.TestOutputHelper;
        output?.WriteLine($"{ids.Length} tokens");
        foreach (var name in OnnxReference.Shared.TracePoints)
        {
            Assert.True(recorder.Tensors.ContainsKey(name), $"Engine did not trace {name}.");
            var diff = Diff.Of(reference[name], recorder.Tensors[name]);
            output?.WriteLine($"{name,-80} {diff}");
            Assert.True(diff.MaxRel < MaxRelative, $"{name}: {diff}");
            Assert.True(diff.RmsRel < RmsRelative, $"{name}: {diff}");
        }
    }
}
