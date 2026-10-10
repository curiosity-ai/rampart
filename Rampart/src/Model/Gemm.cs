using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Rampart.Model;

/// <summary>
/// Single-threaded <c>C = A · B (+ bias)</c> for row-major A (<c>m × k</c>, row stride <c>lda</c>),
/// B (<c>k × n</c>, row stride <c>ldb</c>) and C (<c>m × n</c>, row stride <c>ldc</c>). Strides let
/// callers multiply views in place: one attention head inside the hidden state, a column tile of C.
/// <para>
/// The micro-kernel holds a block of C in registers — 6 rows × 32 columns (two 512-bit vectors) on
/// AVX-512, 4 rows × 16 columns (two 256-bit vectors) elsewhere — and for every <c>p</c> broadcasts
/// one element of each A row against the vectors of B row <c>p</c>. Every FMA is independent, nothing
/// is reduced horizontally, and each loaded vector of B feeds four to six FMAs.
/// </para>
/// </summary>
internal static class Gemm
{
    /// <summary>
    /// 512-bit kernels whenever the CPU has AVX-512F. <see cref="Vector512.IsHardwareAccelerated"/> is
    /// deliberately conservative (false on parts that downclock under AVX-512), but for a dense FMA
    /// loop like this one the wider vectors win regardless.
    /// </summary>
    internal static bool UseVector512 { get; set; } = System.Runtime.Intrinsics.X86.Avx512F.IsSupported;

    public static void Multiply(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k,
        ReadOnlySpan<float> bias = default)
        => Multiply(a, lda, b, ldb, c, ldc, m, n, k, bias, UseVector512, Vector256.IsHardwareAccelerated);

    /// <summary>As above, with the kernel widths chosen explicitly (for tests).</summary>
    internal static void Multiply(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k,
        ReadOnlySpan<float> bias, bool useVector512, bool useVector256)
    {
        if (m == 0 || n == 0) return;
        if ((long)(m - 1) * lda + k > a.Length || (k > 0 && (long)(k - 1) * ldb + n > b.Length) || (long)(m - 1) * ldc + n > c.Length)
            throw new ArgumentException("GEMM operand is smaller than its declared shape.");
        if (!bias.IsEmpty && bias.Length < n) throw new ArgumentException("Bias is shorter than n.", nameof(bias));

        int j = 0;
        if (useVector512)
        {
            const int nr = 32;
            for (; j + nr <= n; j += nr)
            {
                int i = 0;
                for (; i + 6 <= m; i += 6) Kernel6x32(a, lda, b, ldb, c, ldc, i, j, k, bias);
                for (; i < m; i++) Kernel1x32(a, lda, b, ldb, c, ldc, i, j, k, bias);
            }
        }
        if (useVector256)
        {
            const int nr = 16;
            for (; j + nr <= n; j += nr)
            {
                int i = 0;
                for (; i + 4 <= m; i += 4) Kernel4x16(a, lda, b, ldb, c, ldc, i, j, k, bias);
                for (; i < m; i++) Kernel1x16(a, lda, b, ldb, c, ldc, i, j, k, bias);
            }
        }
        if (j < n) Scalar(a, lda, b, ldb, c, ldc, m, j, n, k, bias);
    }

    private static void Scalar(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int m, int j0, int n, int k, ReadOnlySpan<float> bias)
    {
        for (int i = 0; i < m; i++)
        {
            var crow = c.Slice(i * ldc, n);
            for (int j = j0; j < n; j++) crow[j] = 0;
            for (int p = 0; p < k; p++)
            {
                float x = a[i * lda + p];
                var brow = b.Slice(p * ldb, n);
                for (int j = j0; j < n; j++) crow[j] += x * brow[j];
            }
            if (!bias.IsEmpty) for (int j = j0; j < n; j++) crow[j] = bias[j] + crow[j];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x32(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int i, int j, int k, ReadOnlySpan<float> bias)
    {
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetReference(b), j);
        nuint a0 = (nuint)(i * lda), sa = (nuint)lda;

        Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        Vector512<float> c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;

        for (int p = 0; p < k; p++)
        {
            nuint bo = (nuint)(p * ldb);
            var b0 = Vector512.LoadUnsafe(ref rb, bo);
            var b1 = Vector512.LoadUnsafe(ref rb, bo + 16);
            nuint ap = a0 + (nuint)p;
            var x = Vector512.Create(Unsafe.Add(ref ra, ap));
            c00 = Vector512.FusedMultiplyAdd(x, b0, c00); c01 = Vector512.FusedMultiplyAdd(x, b1, c01);
            x = Vector512.Create(Unsafe.Add(ref ra, ap + sa));
            c10 = Vector512.FusedMultiplyAdd(x, b0, c10); c11 = Vector512.FusedMultiplyAdd(x, b1, c11);
            x = Vector512.Create(Unsafe.Add(ref ra, ap + 2 * sa));
            c20 = Vector512.FusedMultiplyAdd(x, b0, c20); c21 = Vector512.FusedMultiplyAdd(x, b1, c21);
            x = Vector512.Create(Unsafe.Add(ref ra, ap + 3 * sa));
            c30 = Vector512.FusedMultiplyAdd(x, b0, c30); c31 = Vector512.FusedMultiplyAdd(x, b1, c31);
            x = Vector512.Create(Unsafe.Add(ref ra, ap + 4 * sa));
            c40 = Vector512.FusedMultiplyAdd(x, b0, c40); c41 = Vector512.FusedMultiplyAdd(x, b1, c41);
            x = Vector512.Create(Unsafe.Add(ref ra, ap + 5 * sa));
            c50 = Vector512.FusedMultiplyAdd(x, b0, c50); c51 = Vector512.FusedMultiplyAdd(x, b1, c51);
        }

        ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetReference(c), i * ldc + j);
        nuint sc = (nuint)ldc;
        Vector512<float> bias0 = default, bias1 = default;
        if (!bias.IsEmpty)
        {
            bias0 = Vector512.Create(bias.Slice(j, 16));
            bias1 = Vector512.Create(bias.Slice(j + 16, 16));
        }
        (bias0 + c00).StoreUnsafe(ref rc); (bias1 + c01).StoreUnsafe(ref rc, 16);
        (bias0 + c10).StoreUnsafe(ref rc, sc); (bias1 + c11).StoreUnsafe(ref rc, sc + 16);
        (bias0 + c20).StoreUnsafe(ref rc, 2 * sc); (bias1 + c21).StoreUnsafe(ref rc, 2 * sc + 16);
        (bias0 + c30).StoreUnsafe(ref rc, 3 * sc); (bias1 + c31).StoreUnsafe(ref rc, 3 * sc + 16);
        (bias0 + c40).StoreUnsafe(ref rc, 4 * sc); (bias1 + c41).StoreUnsafe(ref rc, 4 * sc + 16);
        (bias0 + c50).StoreUnsafe(ref rc, 5 * sc); (bias1 + c51).StoreUnsafe(ref rc, 5 * sc + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel1x32(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int i, int j, int k, ReadOnlySpan<float> bias)
    {
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetReference(b), j);
        Vector512<float> c0 = default, c1 = default;
        for (int p = 0; p < k; p++)
        {
            nuint bo = (nuint)(p * ldb);
            var x = Vector512.Create(Unsafe.Add(ref ra, i * lda + p));
            c0 = Vector512.FusedMultiplyAdd(x, Vector512.LoadUnsafe(ref rb, bo), c0);
            c1 = Vector512.FusedMultiplyAdd(x, Vector512.LoadUnsafe(ref rb, bo + 16), c1);
        }
        ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetReference(c), i * ldc + j);
        if (!bias.IsEmpty)
        {
            c0 = Vector512.Create(bias.Slice(j, 16)) + c0;
            c1 = Vector512.Create(bias.Slice(j + 16, 16)) + c1;
        }
        c0.StoreUnsafe(ref rc);
        c1.StoreUnsafe(ref rc, 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel4x16(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int i, int j, int k, ReadOnlySpan<float> bias)
    {
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetReference(b), j);
        nuint a0 = (nuint)(i * lda), sa = (nuint)lda;

        Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector256<float> c20 = default, c21 = default, c30 = default, c31 = default;

        for (int p = 0; p < k; p++)
        {
            nuint bo = (nuint)(p * ldb);
            var b0 = Vector256.LoadUnsafe(ref rb, bo);
            var b1 = Vector256.LoadUnsafe(ref rb, bo + 8);
            nuint ap = a0 + (nuint)p;
            var x = Vector256.Create(Unsafe.Add(ref ra, ap));
            c00 = Vector256.FusedMultiplyAdd(x, b0, c00); c01 = Vector256.FusedMultiplyAdd(x, b1, c01);
            x = Vector256.Create(Unsafe.Add(ref ra, ap + sa));
            c10 = Vector256.FusedMultiplyAdd(x, b0, c10); c11 = Vector256.FusedMultiplyAdd(x, b1, c11);
            x = Vector256.Create(Unsafe.Add(ref ra, ap + 2 * sa));
            c20 = Vector256.FusedMultiplyAdd(x, b0, c20); c21 = Vector256.FusedMultiplyAdd(x, b1, c21);
            x = Vector256.Create(Unsafe.Add(ref ra, ap + 3 * sa));
            c30 = Vector256.FusedMultiplyAdd(x, b0, c30); c31 = Vector256.FusedMultiplyAdd(x, b1, c31);
        }

        ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetReference(c), i * ldc + j);
        nuint sc = (nuint)ldc;
        Vector256<float> bias0 = default, bias1 = default;
        if (!bias.IsEmpty)
        {
            bias0 = Vector256.Create(bias.Slice(j, 8));
            bias1 = Vector256.Create(bias.Slice(j + 8, 8));
        }
        (bias0 + c00).StoreUnsafe(ref rc); (bias1 + c01).StoreUnsafe(ref rc, 8);
        (bias0 + c10).StoreUnsafe(ref rc, sc); (bias1 + c11).StoreUnsafe(ref rc, sc + 8);
        (bias0 + c20).StoreUnsafe(ref rc, 2 * sc); (bias1 + c21).StoreUnsafe(ref rc, 2 * sc + 8);
        (bias0 + c30).StoreUnsafe(ref rc, 3 * sc); (bias1 + c31).StoreUnsafe(ref rc, 3 * sc + 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel1x16(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int i, int j, int k, ReadOnlySpan<float> bias)
    {
        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref Unsafe.Add(ref MemoryMarshal.GetReference(b), j);
        Vector256<float> c0 = default, c1 = default;
        for (int p = 0; p < k; p++)
        {
            nuint bo = (nuint)(p * ldb);
            var x = Vector256.Create(Unsafe.Add(ref ra, i * lda + p));
            c0 = Vector256.FusedMultiplyAdd(x, Vector256.LoadUnsafe(ref rb, bo), c0);
            c1 = Vector256.FusedMultiplyAdd(x, Vector256.LoadUnsafe(ref rb, bo + 8), c1);
        }
        ref float rc = ref Unsafe.Add(ref MemoryMarshal.GetReference(c), i * ldc + j);
        if (!bias.IsEmpty)
        {
            c0 = Vector256.Create(bias.Slice(j, 8)) + c0;
            c1 = Vector256.Create(bias.Slice(j + 8, 8)) + c1;
        }
        c0.StoreUnsafe(ref rc);
        c1.StoreUnsafe(ref rc, 8);
    }
}
