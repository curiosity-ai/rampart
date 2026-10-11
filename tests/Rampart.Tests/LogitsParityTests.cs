using Catalyst.Rampart.Model;

namespace Catalyst.Rampart.Tests;

/// <summary>
/// Final logits against ONNX Runtime with full graph optimization (fused attention, LayerNorm and
/// GELU kernels): what a deployment of the original model computes. Covers every fixture case,
/// window by window for the long ones.
/// </summary>
public sealed class LogitsParityTests
{
    public static TheoryData<string> Cases() => ReferenceFixture.CaseIds();

    [Theory]
    [MemberData(nameof(Cases))]
    public void LogitsMatchOptimizedOnnxRuntime(string id)
    {
        var model = RampartModel.Default;
        var text = ReferenceFixture.Case(id).GetProperty("text").GetString()!;
        var content = model.Tokenizer.Encode(text, addSpecialTokens: false);
        int labels = model.Labels.Count;

        for (int offset = 0; offset == 0 || offset < content.Length; offset += 510)
        {
            int[] ids = [model.Tokenizer.ClsId, .. content.AsSpan(offset, Math.Min(510, content.Length - offset)), model.Tokenizer.SepId];
            var expected = OnnxReference.Shared.RunLogits(ids);
            var actual = model.ComputeLogits(ids);
            var diff = Diff.Of(expected, actual);
            Assert.True(diff.MaxAbs < 1e-3 && diff.RmsRel < 2e-5, $"{id}@{offset}: {diff}");

            // Same argmax label for every token whose top two logits are not a near-tie.
            for (int t = 0; t < ids.Length; t++)
            {
                var e = expected.AsSpan(t * labels, labels);
                var a = actual.AsSpan(t * labels, labels);
                int best = ArgMax(e);
                float runnerUp = float.MinValue;
                for (int k = 0; k < labels; k++) if (k != best) runnerUp = Math.Max(runnerUp, e[k]);
                if (e[best] - runnerUp > 1e-3f) Assert.Equal(best, ArgMax(a));
            }
        }
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }
}
