using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Rampart.Model;

/// <summary>The elementwise and row-wise kernels of a BERT encoder, vectorized where it pays.</summary>
internal static class Kernels
{
    /// <summary>
    /// ONNX <c>LayerNormalization</c> over the last axis, in place, for <paramref name="rows"/> rows
    /// of width <paramref name="gamma"/>.Length.
    /// </summary>
    public static unsafe void LayerNorm(Span<float> data, int rows, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, float epsilon)
    {
        int width = gamma.Length;
        if (rows >= ParallelRows)
        {
            fixed (float* pd = data)
            fixed (float* pg = gamma)
            fixed (float* pb = beta)
            {
                nint dp = (nint)pd, gp = (nint)pg, bp = (nint)pb;
                int chunks = (rows + ParallelRows / 2 - 1) / (ParallelRows / 2);
                Parallel.For(0, chunks, chunk =>
                {
                    int r0 = chunk * (ParallelRows / 2);
                    int count = Math.Min(ParallelRows / 2, rows - r0);
                    var slice = new Span<float>((void*)(dp + (nint)r0 * width * sizeof(float)), count * width);
                    LayerNorm(slice, count, new ReadOnlySpan<float>((void*)gp, width), new ReadOnlySpan<float>((void*)bp, width), epsilon);
                });
            }
            return;
        }
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * width, width);
            float mean = TensorPrimitives.Sum(row) / width;
            // Two-pass variance: Σ (x - mean)² / width.
            TensorPrimitives.Subtract(row, mean, row);
            float variance = TensorPrimitives.SumOfSquares(row) / width;
            float inv = 1f / MathF.Sqrt(variance + epsilon);
            TensorPrimitives.Multiply(row, inv, row);
            TensorPrimitives.FusedMultiplyAdd(row, gamma, beta, row);
        }
    }

    /// <summary>
    /// Exact (erf-based) GELU, in place, evaluated the way the exported graph spells it:
    /// <c>x · (erf(x / √2) + 1) · 0.5</c>.
    /// </summary>
    public static unsafe void Gelu(Span<float> data)
    {
        if (data.Length >= ParallelElements)
        {
            fixed (float* pd = data)
            {
                nint dp = (nint)pd;
                int length = data.Length;
                int chunk = ParallelElements / 2;
                Parallel.For(0, (length + chunk - 1) / chunk, c =>
                {
                    int start = c * chunk;
                    GeluSerial(new Span<float>((void*)(dp + (nint)start * sizeof(float)), Math.Min(chunk, length - start)));
                });
            }
            return;
        }
        GeluSerial(data);
    }

    /// <summary>Row counts / element counts above which the elementwise kernels split across cores.</summary>
    private const int ParallelRows = 128;
    private const int ParallelElements = 1 << 17;

    private static void GeluSerial(Span<float> data)
    {
        const float invSqrt2 = 0.70710678118654752440f;
        ref float start = ref MemoryMarshal.GetReference(data);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var vInvSqrt2 = new Vector<float>(invSqrt2);
            var one = Vector<float>.One;
            var half = new Vector<float>(0.5f);
            for (; i <= data.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var x = Vector.LoadUnsafe(ref start, (nuint)i);
                var e = Erf(x * vInvSqrt2);
                (x * (e + one) * half).StoreUnsafe(ref start, (nuint)i);
            }
        }
        for (; i < data.Length; i++)
        {
            float x = data[i];
            data[i] = x * (Erf(x * invSqrt2) + 1f) * 0.5f;
        }
    }

    // Rational minimax approximation of erf on [-4, 4] (odd numerator / even denominator, the
    // form used by Eigen and XLA). Max absolute error ≈ 4e-7 over the real line; see ErfTests.
    private const float A1 = -1.60960333262415e-02f;
    private const float A3 = -2.95459980854025e-03f;
    private const float A5 = -7.34990630326855e-04f;
    private const float A7 = -5.69250639462346e-05f;
    private const float A9 = -2.10102402082508e-06f;
    private const float A11 = 2.77068142495902e-08f;
    private const float A13 = -2.72614225801306e-10f;
    private const float B0 = -1.42647390514189e-02f;
    private const float B2 = -7.37332916720468e-03f;
    private const float B4 = -1.68282697438203e-03f;
    private const float B6 = -2.13374055278905e-04f;
    private const float B8 = -1.45660718464996e-05f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Erf(float value)
    {
        float x = Math.Clamp(value, -4f, 4f);
        float x2 = x * x;
        float p = A13;
        p = p * x2 + A11;
        p = p * x2 + A9;
        p = p * x2 + A7;
        p = p * x2 + A5;
        p = p * x2 + A3;
        p = p * x2 + A1;
        p *= x;
        float q = B8;
        q = q * x2 + B6;
        q = q * x2 + B4;
        q = q * x2 + B2;
        q = q * x2 + B0;
        return p / q;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Erf(Vector<float> value)
    {
        var x = Vector.Min(Vector.Max(value, new Vector<float>(-4f)), new Vector<float>(4f));
        var x2 = x * x;
        var p = new Vector<float>(A13);
        p = p * x2 + new Vector<float>(A11);
        p = p * x2 + new Vector<float>(A9);
        p = p * x2 + new Vector<float>(A7);
        p = p * x2 + new Vector<float>(A5);
        p = p * x2 + new Vector<float>(A3);
        p = p * x2 + new Vector<float>(A1);
        p *= x;
        var q = new Vector<float>(B8);
        q = q * x2 + new Vector<float>(B6);
        q = q * x2 + new Vector<float>(B4);
        q = q * x2 + new Vector<float>(B2);
        q = q * x2 + new Vector<float>(B0);
        return p / q;
    }

    /// <summary>Numerically stable softmax of one row, in place.</summary>
    public static void SoftmaxInPlace(Span<float> row)
    {
        float max = TensorPrimitives.Max(row);
        TensorPrimitives.Subtract(row, max, row);
        TensorPrimitives.Exp(row, row);
        float sum = TensorPrimitives.Sum(row);
        TensorPrimitives.Divide(row, sum, row);
    }
}
