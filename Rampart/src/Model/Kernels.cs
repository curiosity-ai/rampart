using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Catalyst.Rampart.Model;

/// <summary>The elementwise and row-wise kernels of a BERT encoder, vectorized where it pays.</summary>
internal static class Kernels
{
    /// <summary>
    /// ONNX <c>LayerNormalization</c> over the last axis, in place, for <paramref name="rows"/> rows
    /// of width <paramref name="gamma"/>.Length.
    /// </summary>
    public static unsafe void LayerNorm(Span<float> data, int rows, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, float epsilon, int parallelism = 1)
    {
        int width = gamma.Length;
        if (parallelism > 1 && rows >= ParallelRows)
        {
            fixed (float* pd = data)
            fixed (float* pg = gamma)
            fixed (float* pb = beta)
            {
                var job = new LayerNormJob((nint)pd, (nint)pg, (nint)pb, rows, width, epsilon);
                Parallelism.For((rows + ParallelRows / 2 - 1) / (ParallelRows / 2), parallelism, ref job);
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
    /// <c>destination = LayerNorm(x + residual)</c> row by row: the residual add of a post-norm block
    /// fused into the normalization, three passes over each row while it is in L1 (sum while adding,
    /// then Σ(x - mean)², then scale and shift) — the same two-pass variance as <see cref="LayerNorm"/>.
    /// </summary>
    public static unsafe void AddLayerNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> residual, Span<float> destination, int rows, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, float epsilon, int parallelism = 1)
    {
        int width = gamma.Length;
        if (x.Length < rows * width || residual.Length < rows * width || destination.Length < rows * width)
            throw new ArgumentException("LayerNorm operand is smaller than rows x width.");
        if (parallelism > 1 && rows >= ParallelRows)
        {
            fixed (float* px = x)
            fixed (float* pr = residual)
            fixed (float* pd = destination)
            fixed (float* pg = gamma)
            fixed (float* pb = beta)
            {
                var job = new AddLayerNormJob((nint)px, (nint)pr, (nint)pd, (nint)pg, (nint)pb, rows, width, epsilon);
                Parallelism.For((rows + ParallelRows / 2 - 1) / (ParallelRows / 2), parallelism, ref job);
            }
            return;
        }
        if (width % 16 != 0 || !(Vector512.IsHardwareAccelerated || System.Runtime.Intrinsics.X86.Avx512F.IsSupported))
        {
            TensorPrimitives.Add(x[..(rows * width)], residual[..(rows * width)], destination);
            LayerNorm(destination, rows, gamma, beta, epsilon);
            return;
        }

        ref float rg = ref MemoryMarshal.GetReference(gamma);
        ref float rbe = ref MemoryMarshal.GetReference(beta);
        for (int r = 0; r < rows; r++)
        {
            ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetReference(x), r * width);
            ref float rr = ref Unsafe.Add(ref MemoryMarshal.GetReference(residual), r * width);
            ref float rd = ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), r * width);
            var sum = Vector512<float>.Zero;
            for (nuint i = 0; i < (nuint)width; i += 16)
            {
                var v = Vector512.LoadUnsafe(ref rx, i) + Vector512.LoadUnsafe(ref rr, i);
                v.StoreUnsafe(ref rd, i);
                sum += v;
            }
            var mean = Vector512.Create(Vector512.Sum(sum) / width);
            var sq = Vector512<float>.Zero;
            for (nuint i = 0; i < (nuint)width; i += 16)
            {
                var d = Vector512.LoadUnsafe(ref rd, i) - mean;
                sq = Vector512.FusedMultiplyAdd(d, d, sq);
            }
            var inv = Vector512.Create(1f / MathF.Sqrt(Vector512.Sum(sq) / width + epsilon));
            for (nuint i = 0; i < (nuint)width; i += 16)
            {
                var d = (Vector512.LoadUnsafe(ref rd, i) - mean) * inv;
                Vector512.FusedMultiplyAdd(d, Vector512.LoadUnsafe(ref rg, i), Vector512.LoadUnsafe(ref rbe, i)).StoreUnsafe(ref rd, i);
            }
        }
    }

    /// <summary>
    /// Exact (erf-based) GELU, in place, evaluated the way the exported graph spells it:
    /// <c>x · (erf(x / √2) + 1) · 0.5</c>.
    /// </summary>
    public static unsafe void Gelu(Span<float> data, int parallelism = 1)
    {
        if (parallelism > 1 && data.Length >= ParallelElements)
        {
            fixed (float* pd = data)
            {
                var job = new GeluJob((nint)pd, data.Length);
                Parallelism.For((data.Length + ParallelElements / 2 - 1) / (ParallelElements / 2), parallelism, ref job);
            }
            return;
        }
        GeluSerial(data);
    }

    private readonly unsafe struct LayerNormJob(nint data, nint gamma, nint beta, int rows, int width, float epsilon) : IParallelJob
    {
        public void Run(int from, int to)
        {
            for (int chunk = from; chunk < to; chunk++)
            {
                int r0 = chunk * (ParallelRows / 2);
                int count = Math.Min(ParallelRows / 2, rows - r0);
                var slice = new Span<float>((void*)(data + (nint)r0 * width * sizeof(float)), count * width);
                LayerNorm(slice, count, new ReadOnlySpan<float>((void*)gamma, width), new ReadOnlySpan<float>((void*)beta, width), epsilon);
            }
        }
    }

    private readonly unsafe struct AddLayerNormJob(nint x, nint residual, nint destination, nint gamma, nint beta, int rows, int width, float epsilon) : IParallelJob
    {
        public void Run(int from, int to)
        {
            for (int chunk = from; chunk < to; chunk++)
            {
                int r0 = chunk * (ParallelRows / 2);
                int count = Math.Min(ParallelRows / 2, rows - r0);
                nint at = (nint)r0 * width * sizeof(float);
                int len = count * width;
                AddLayerNorm(new ReadOnlySpan<float>((void*)(x + at), len), new ReadOnlySpan<float>((void*)(residual + at), len), new Span<float>((void*)(destination + at), len),
                    count, new ReadOnlySpan<float>((void*)gamma, width), new ReadOnlySpan<float>((void*)beta, width), epsilon);
            }
        }
    }

    private readonly unsafe struct GeluJob(nint data, int length) : IParallelJob
    {
        public void Run(int from, int to)
        {
            const int chunk = ParallelElements / 2;
            for (int c = from; c < to; c++)
            {
                int start = c * chunk;
                GeluSerial(new Span<float>((void*)(data + (nint)start * sizeof(float)), Math.Min(chunk, length - start)));
            }
        }
    }

    /// <summary>Row counts / element counts above which the elementwise kernels split across cores.</summary>
    private const int ParallelRows = 128;
    private const int ParallelElements = 1 << 17;

    private static void GeluSerial(Span<float> data)
    {
        ref float start = ref MemoryMarshal.GetReference(data);
        int i = 0;
        if (Vector512.IsHardwareAccelerated || System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
        {
            for (; i + 16 <= data.Length; i += 16) Gelu(Vector512.LoadUnsafe(ref start, (nuint)i)).StoreUnsafe(ref start, (nuint)i);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 8 <= data.Length; i += 8) Gelu(Vector256.LoadUnsafe(ref start, (nuint)i)).StoreUnsafe(ref start, (nuint)i);
        }
        for (; i < data.Length; i++) data[i] = Gelu(data[i]);
    }

    private const float InvSqrt2 = 0.70710678118654752440f;

    /// <summary><c>x · (erf(x / √2) + 1) · 0.5</c>, in the graph's order of operations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Gelu(Vector512<float> x) => x * (Erf(x * Vector512.Create(InvSqrt2)) + Vector512<float>.One) * Vector512.Create(0.5f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Gelu(Vector256<float> x) => x * (Erf(x * Vector256.Create(InvSqrt2)) + Vector256<float>.One) * Vector256.Create(0.5f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x) => x * (Erf(x * InvSqrt2) + 1f) * 0.5f;

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
    public static Vector512<float> Erf(Vector512<float> value)
    {
        var x = Vector512.Min(Vector512.Max(value, Vector512.Create(-4f)), Vector512.Create(4f));
        var x2 = x * x;
        var p = Vector512.Create(A13);
        p = p * x2 + Vector512.Create(A11);
        p = p * x2 + Vector512.Create(A9);
        p = p * x2 + Vector512.Create(A7);
        p = p * x2 + Vector512.Create(A5);
        p = p * x2 + Vector512.Create(A3);
        p = p * x2 + Vector512.Create(A1);
        p *= x;
        var q = Vector512.Create(B8);
        q = q * x2 + Vector512.Create(B6);
        q = q * x2 + Vector512.Create(B4);
        q = q * x2 + Vector512.Create(B2);
        q = q * x2 + Vector512.Create(B0);
        return p / q;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Erf(Vector256<float> value)
    {
        var x = Vector256.Min(Vector256.Max(value, Vector256.Create(-4f)), Vector256.Create(4f));
        var x2 = x * x;
        var p = Vector256.Create(A13);
        p = p * x2 + Vector256.Create(A11);
        p = p * x2 + Vector256.Create(A9);
        p = p * x2 + Vector256.Create(A7);
        p = p * x2 + Vector256.Create(A5);
        p = p * x2 + Vector256.Create(A3);
        p = p * x2 + Vector256.Create(A1);
        p *= x;
        var q = Vector256.Create(B8);
        q = q * x2 + Vector256.Create(B6);
        q = q * x2 + Vector256.Create(B4);
        q = q * x2 + Vector256.Create(B2);
        q = q * x2 + Vector256.Create(B0);
        return p / q;
    }

    /// <summary>
    /// Numerically stable softmax of one row, in place, in three passes as ONNX Runtime's MLAS does it:
    /// the maximum; <c>exp(x - max)</c> stored and summed; a multiply by <c>1 / sum</c>.
    /// </summary>
    public static void SoftmaxInPlace(Span<float> row)
    {
        if (row.IsEmpty) return;
        float max = Max(row);
        ref float r = ref MemoryMarshal.GetReference(row);
        int n = row.Length, i = 0;
        float sum = 0;
        if (Vector512.IsHardwareAccelerated || System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
        {
            var vmax = Vector512.Create(max);
            var acc = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var e = Exp(Vector512.LoadUnsafe(ref r, (nuint)i) - vmax);
                e.StoreUnsafe(ref r, (nuint)i);
                acc += e;
            }
            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vmax = Vector256.Create(max);
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var e = Exp(Vector256.LoadUnsafe(ref r, (nuint)i) - vmax);
                e.StoreUnsafe(ref r, (nuint)i);
                acc += e;
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < n; i++)
        {
            float e = Exp(Unsafe.Add(ref r, i) - max);
            Unsafe.Add(ref r, i) = e;
            sum += e;
        }
        TensorPrimitives.Multiply(row, 1f / sum, row);
    }

    /// <summary>Maximum of a non-empty row of finite values (no NaN propagation, unlike <c>TensorPrimitives.Max</c>).</summary>
    private static float Max(ReadOnlySpan<float> row)
    {
        ref float r = ref MemoryMarshal.GetReference(row);
        int n = row.Length, i = 0;
        float max = float.NegativeInfinity;
        if (n >= 64 && (Vector512.IsHardwareAccelerated || System.Runtime.Intrinsics.X86.Avx512F.IsSupported))
        {
            var m0 = Vector512.Create(float.NegativeInfinity);
            var m1 = m0;
            for (; i + 32 <= n; i += 32)
            {
                m0 = Vector512.Max(m0, Vector512.LoadUnsafe(ref r, (nuint)i));
                m1 = Vector512.Max(m1, Vector512.LoadUnsafe(ref r, (nuint)i + 16));
            }
            var m = Vector512.Max(m0, m1);
            for (int l = 0; l < 16; l++) max = Math.Max(max, m[l]);
        }
        else if (n >= 16 && Vector256.IsHardwareAccelerated)
        {
            var m = Vector256.Create(float.NegativeInfinity);
            for (; i + 8 <= n; i += 8) m = Vector256.Max(m, Vector256.LoadUnsafe(ref r, (nuint)i));
            for (int l = 0; l < 8; l++) max = Math.Max(max, m[l]);
        }
        for (; i < n; i++) max = Math.Max(max, Unsafe.Add(ref r, i));
        return max;
    }

    // exp(x) = 2^n · e^f with n = round(x / ln 2), f = x - n·ln 2 ∈ [-ln2/2, ln2/2] (Cody–Waite split of
    // ln 2), e^f by its degree-6 Taylor polynomial (truncation < 2e-7 relative), 2^n built in the exponent
    // bits. Inputs are clamped to [ln(FLT_MIN), ln(FLT_MAX)]; see KernelTests for the error bound.
    private const float ExpLow = -87.33654f, ExpHigh = 88.72283f;
    private const float Log2E = 1.44269504088896341f, Ln2Hi = 0.693359375f, Ln2Lo = -2.12194440e-4f;
    private const float P2 = 1f / 2, P3 = 1f / 6, P4 = 1f / 24, P5 = 1f / 120, P6 = 1f / 720;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(ExpLow)), Vector512.Create(ExpHigh));
        var n = Vector512.Round(x * Vector512.Create(Log2E));
        var f = Vector512.FusedMultiplyAdd(n, Vector512.Create(-Ln2Hi), x);
        f = Vector512.FusedMultiplyAdd(n, Vector512.Create(-Ln2Lo), f);
        var p = Vector512.Create(P6);
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(P5));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(P4));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(P3));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(P2));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512<float>.One);
        p = Vector512.FusedMultiplyAdd(p, f, Vector512<float>.One);
        var scale = Vector512.ShiftLeft(Vector512.ConvertToInt32(n) + Vector512.Create(127), 23).AsSingle();
        return p * scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Vector256.Min(Vector256.Max(x, Vector256.Create(ExpLow)), Vector256.Create(ExpHigh));
        var n = Vector256.Round(x * Vector256.Create(Log2E));
        var f = Vector256.FusedMultiplyAdd(n, Vector256.Create(-Ln2Hi), x);
        f = Vector256.FusedMultiplyAdd(n, Vector256.Create(-Ln2Lo), f);
        var p = Vector256.Create(P6);
        p = Vector256.FusedMultiplyAdd(p, f, Vector256.Create(P5));
        p = Vector256.FusedMultiplyAdd(p, f, Vector256.Create(P4));
        p = Vector256.FusedMultiplyAdd(p, f, Vector256.Create(P3));
        p = Vector256.FusedMultiplyAdd(p, f, Vector256.Create(P2));
        p = Vector256.FusedMultiplyAdd(p, f, Vector256<float>.One);
        p = Vector256.FusedMultiplyAdd(p, f, Vector256<float>.One);
        var scale = Vector256.ShiftLeft(Vector256.ConvertToInt32(n) + Vector256.Create(127), 23).AsSingle();
        return p * scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(float x)
    {
        x = Math.Clamp(x, ExpLow, ExpHigh);
        float n = MathF.Round(x * Log2E);
        float f = MathF.FusedMultiplyAdd(n, -Ln2Hi, x);
        f = MathF.FusedMultiplyAdd(n, -Ln2Lo, f);
        float p = P6;
        p = MathF.FusedMultiplyAdd(p, f, P5);
        p = MathF.FusedMultiplyAdd(p, f, P4);
        p = MathF.FusedMultiplyAdd(p, f, P3);
        p = MathF.FusedMultiplyAdd(p, f, P2);
        p = MathF.FusedMultiplyAdd(p, f, 1f);
        p = MathF.FusedMultiplyAdd(p, f, 1f);
        return p * BitConverter.Int32BitsToSingle(((int)n + 127) << 23);
    }
}
