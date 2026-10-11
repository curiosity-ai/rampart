using Catalyst.Rampart.Model;

namespace Catalyst.Rampart.Tests;

internal sealed class TraceRecorder : IForwardTrace
{
    public Dictionary<string, float[]> Tensors { get; } = new(StringComparer.Ordinal);

    public void Record(string onnxName, ReadOnlySpan<float> data) => Tensors[onnxName] = data.ToArray();
}

/// <summary>Elementwise comparison of two tensors.</summary>
internal readonly record struct Diff(double MaxAbs, double MaxRel, double RmsRel, int Count)
{
    public static Diff Of(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException($"Length {actual.Length} != {expected.Length}.");
        double maxAbs = 0, sumSq = 0, refSq = 0, peak = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)expected[i] - actual[i]);
            if (double.IsNaN(d)) return new Diff(double.NaN, double.NaN, double.NaN, expected.Length);
            maxAbs = Math.Max(maxAbs, d);
            sumSq += d * d;
            refSq += (double)expected[i] * expected[i];
            peak = Math.Max(peak, Math.Abs(expected[i]));
        }
        // Max error relative to the tensor's peak magnitude; RMS error relative to its RMS.
        return new Diff(maxAbs, peak == 0 ? maxAbs : maxAbs / peak, refSq == 0 ? 0 : Math.Sqrt(sumSq / refSq), expected.Length);
    }

    public override string ToString() => $"maxAbs={MaxAbs:E2} maxRel={MaxRel:E2} rmsRel={RmsRel:E2}";
}
