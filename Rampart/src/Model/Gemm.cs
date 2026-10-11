using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Catalyst.Rampart.Model;

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

        ref float ra = ref MemoryMarshal.GetReference(a);
        ref float rb = ref MemoryMarshal.GetReference(b);
        ref float rc = ref MemoryMarshal.GetReference(c);
        ref float rbias = ref MemoryMarshal.GetReference(bias);
        bool hasBias = !bias.IsEmpty;

        int j = 0;
        if (useVector512)
        {
            for (; j + 32 <= n; j += 32)
            {
                int i = 0;
                for (; i + 12 <= m; i += 12) Block512<R12>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias);
                if (i + 8 <= m) { Block512<R8>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias); i += 8; }
                if (i + 4 <= m) { Block512<R4>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias); i += 4; }
                if (i + 2 <= m) { Block512<R2>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias); i += 2; }
                if (i < m) Block512<R1>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias);
            }
        }
        if (useVector256)
        {
            for (; j + 16 <= n; j += 16)
            {
                int i = 0;
                for (; i + 6 <= m; i += 6) Block256<R6>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias);
                if (i + 4 <= m) { Block256<R4>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias); i += 4; }
                if (i + 2 <= m) { Block256<R2>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias); i += 2; }
                if (i < m) Block256<R1>(ref ra, lda, ref rb, ldb, ref rc, ldc, i, j, k, ref rbias, hasBias);
            }
        }
        if (j < n)
        {
            if (useVector512 || useVector256) Remainder(a, lda, b, ldb, c, ldc, m, j, n, k, bias, useVector512, useVector256);
            else Scalar(a, lda, b, ldb, c, ldc, m, j, n, k, bias);
        }
    }

    /// <summary>
    /// Columns <c>[j0, n)</c> that do not fill a vector block: copy that slice of B into a zero-padded
    /// panel, run the vector kernel into a scratch block of C, and copy back the valid columns. Short
    /// sequences hit this on every attention score (n = tokens) and the classifier (n = 35).
    /// </summary>
    private static void Remainder(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int m, int j0, int n, int k, ReadOnlySpan<float> bias, bool useVector512, bool useVector256)
    {
        int width = useVector512 ? 32 : 16;
        int cols = n - j0;
        var panel = ArrayPool<float>.Shared.Rent(k * width + width + m * width);
        try
        {
            var pb = panel.AsSpan(0, k * width);
            var pbias = panel.AsSpan(k * width, width);
            var pc = panel.AsSpan(k * width + width, m * width);
            pb.Clear();
            pbias.Clear();
            for (int p = 0; p < k; p++) b.Slice(p * ldb + j0, cols).CopyTo(pb.Slice(p * width, cols));
            if (!bias.IsEmpty) bias.Slice(j0, cols).CopyTo(pbias);
            Multiply(a, lda, pb, width, pc, width, m, width, k, bias.IsEmpty ? default : pbias, useVector512, useVector256);
            for (int i = 0; i < m; i++) pc.Slice(i * width, cols).CopyTo(c.Slice(i * ldc + j0, cols));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(panel);
        }
    }

    /// <summary>A compile-time row count: each struct instantiation of a block kernel is its own JIT
    /// specialization, so the rows a kernel does not have cost nothing.</summary>
    private interface IRows { static abstract int Count { get; } }
    private struct R1 : IRows { public static int Count => 1; }
    private struct R2 : IRows { public static int Count => 2; }
    private struct R4 : IRows { public static int Count => 4; }
    private struct R6 : IRows { public static int Count => 6; }
    private struct R8 : IRows { public static int Count => 8; }
    private struct R12 : IRows { public static int Count => 12; }

    /// <summary>
    /// <c>TRows × 32</c> block of C in up to 24 zmm accumulators. Per step: two vectors of B, one
    /// broadcast per row of A, two FMAs per row; row pointers are fixed so each A load is <c>[row + 4p]</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block512<TRows>(ref float a, int lda, ref float b, int ldb, ref float c, int ldc, int i, int j, int k, ref float bias, bool hasBias)
        where TRows : struct, IRows
    {
        int rows = TRows.Count;
        ref float a0 = ref Unsafe.Add(ref a, (nint)i * lda);
        ref float a1 = ref Unsafe.Add(ref a0, rows > 1 ? 1 * (nint)lda : 0);
        ref float a2 = ref Unsafe.Add(ref a0, rows > 2 ? 2 * (nint)lda : 0);
        ref float a3 = ref Unsafe.Add(ref a0, rows > 3 ? 3 * (nint)lda : 0);
        ref float a4 = ref Unsafe.Add(ref a0, rows > 4 ? 4 * (nint)lda : 0);
        ref float a5 = ref Unsafe.Add(ref a0, rows > 5 ? 5 * (nint)lda : 0);
        ref float a6 = ref Unsafe.Add(ref a0, rows > 6 ? 6 * (nint)lda : 0);
        ref float a7 = ref Unsafe.Add(ref a0, rows > 7 ? 7 * (nint)lda : 0);
        ref float a8 = ref Unsafe.Add(ref a0, rows > 8 ? 8 * (nint)lda : 0);
        ref float a9 = ref Unsafe.Add(ref a0, rows > 9 ? 9 * (nint)lda : 0);
        ref float a10 = ref Unsafe.Add(ref a0, rows > 10 ? 10 * (nint)lda : 0);
        ref float a11 = ref Unsafe.Add(ref a0, rows > 11 ? 11 * (nint)lda : 0);
        ref float bp = ref Unsafe.Add(ref b, j);

        Vector512<float> c00 = default, c01 = default;
        Vector512<float> c10 = default, c11 = default;
        Vector512<float> c20 = default, c21 = default;
        Vector512<float> c30 = default, c31 = default;
        Vector512<float> c40 = default, c41 = default;
        Vector512<float> c50 = default, c51 = default;
        Vector512<float> c60 = default, c61 = default;
        Vector512<float> c70 = default, c71 = default;
        Vector512<float> c80 = default, c81 = default;
        Vector512<float> c90 = default, c91 = default;
        Vector512<float> c100 = default, c101 = default;
        Vector512<float> c110 = default, c111 = default;

        for (nint p = 0; p < k; p++)
        {
            var b0 = Vector512.LoadUnsafe(ref bp);
            var b1 = Vector512.LoadUnsafe(ref bp, 16);
            bp = ref Unsafe.Add(ref bp, ldb);
            Vector512<float> x;
            { x = Vector512.Create(Unsafe.Add(ref a0, p)); c00 = Vector512.FusedMultiplyAdd(x, b0, c00); c01 = Vector512.FusedMultiplyAdd(x, b1, c01); }
            if (rows > 1) { x = Vector512.Create(Unsafe.Add(ref a1, p)); c10 = Vector512.FusedMultiplyAdd(x, b0, c10); c11 = Vector512.FusedMultiplyAdd(x, b1, c11); }
            if (rows > 2) { x = Vector512.Create(Unsafe.Add(ref a2, p)); c20 = Vector512.FusedMultiplyAdd(x, b0, c20); c21 = Vector512.FusedMultiplyAdd(x, b1, c21); }
            if (rows > 3) { x = Vector512.Create(Unsafe.Add(ref a3, p)); c30 = Vector512.FusedMultiplyAdd(x, b0, c30); c31 = Vector512.FusedMultiplyAdd(x, b1, c31); }
            if (rows > 4) { x = Vector512.Create(Unsafe.Add(ref a4, p)); c40 = Vector512.FusedMultiplyAdd(x, b0, c40); c41 = Vector512.FusedMultiplyAdd(x, b1, c41); }
            if (rows > 5) { x = Vector512.Create(Unsafe.Add(ref a5, p)); c50 = Vector512.FusedMultiplyAdd(x, b0, c50); c51 = Vector512.FusedMultiplyAdd(x, b1, c51); }
            if (rows > 6) { x = Vector512.Create(Unsafe.Add(ref a6, p)); c60 = Vector512.FusedMultiplyAdd(x, b0, c60); c61 = Vector512.FusedMultiplyAdd(x, b1, c61); }
            if (rows > 7) { x = Vector512.Create(Unsafe.Add(ref a7, p)); c70 = Vector512.FusedMultiplyAdd(x, b0, c70); c71 = Vector512.FusedMultiplyAdd(x, b1, c71); }
            if (rows > 8) { x = Vector512.Create(Unsafe.Add(ref a8, p)); c80 = Vector512.FusedMultiplyAdd(x, b0, c80); c81 = Vector512.FusedMultiplyAdd(x, b1, c81); }
            if (rows > 9) { x = Vector512.Create(Unsafe.Add(ref a9, p)); c90 = Vector512.FusedMultiplyAdd(x, b0, c90); c91 = Vector512.FusedMultiplyAdd(x, b1, c91); }
            if (rows > 10) { x = Vector512.Create(Unsafe.Add(ref a10, p)); c100 = Vector512.FusedMultiplyAdd(x, b0, c100); c101 = Vector512.FusedMultiplyAdd(x, b1, c101); }
            if (rows > 11) { x = Vector512.Create(Unsafe.Add(ref a11, p)); c110 = Vector512.FusedMultiplyAdd(x, b0, c110); c111 = Vector512.FusedMultiplyAdd(x, b1, c111); }
        }

        Vector512<float> bias0 = default, bias1 = default;
        if (hasBias)
        {
            bias0 = Vector512.LoadUnsafe(ref bias, (nuint)j);
            bias1 = Vector512.LoadUnsafe(ref bias, (nuint)j + 16);
        }
        ref float cr = ref Unsafe.Add(ref c, (nint)i * ldc + j);
        nuint s = (nuint)ldc;
        { (bias0 + c00).StoreUnsafe(ref cr, 0 * s); (bias1 + c01).StoreUnsafe(ref cr, 0 * s + 16); }
        if (rows > 1) { (bias0 + c10).StoreUnsafe(ref cr, 1 * s); (bias1 + c11).StoreUnsafe(ref cr, 1 * s + 16); }
        if (rows > 2) { (bias0 + c20).StoreUnsafe(ref cr, 2 * s); (bias1 + c21).StoreUnsafe(ref cr, 2 * s + 16); }
        if (rows > 3) { (bias0 + c30).StoreUnsafe(ref cr, 3 * s); (bias1 + c31).StoreUnsafe(ref cr, 3 * s + 16); }
        if (rows > 4) { (bias0 + c40).StoreUnsafe(ref cr, 4 * s); (bias1 + c41).StoreUnsafe(ref cr, 4 * s + 16); }
        if (rows > 5) { (bias0 + c50).StoreUnsafe(ref cr, 5 * s); (bias1 + c51).StoreUnsafe(ref cr, 5 * s + 16); }
        if (rows > 6) { (bias0 + c60).StoreUnsafe(ref cr, 6 * s); (bias1 + c61).StoreUnsafe(ref cr, 6 * s + 16); }
        if (rows > 7) { (bias0 + c70).StoreUnsafe(ref cr, 7 * s); (bias1 + c71).StoreUnsafe(ref cr, 7 * s + 16); }
        if (rows > 8) { (bias0 + c80).StoreUnsafe(ref cr, 8 * s); (bias1 + c81).StoreUnsafe(ref cr, 8 * s + 16); }
        if (rows > 9) { (bias0 + c90).StoreUnsafe(ref cr, 9 * s); (bias1 + c91).StoreUnsafe(ref cr, 9 * s + 16); }
        if (rows > 10) { (bias0 + c100).StoreUnsafe(ref cr, 10 * s); (bias1 + c101).StoreUnsafe(ref cr, 10 * s + 16); }
        if (rows > 11) { (bias0 + c110).StoreUnsafe(ref cr, 11 * s); (bias1 + c111).StoreUnsafe(ref cr, 11 * s + 16); }
    }

    /// <summary><c>TRows × 16</c> block in up to 12 ymm accumulators (fits the 16 registers of AVX2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block256<TRows>(ref float a, int lda, ref float b, int ldb, ref float c, int ldc, int i, int j, int k, ref float bias, bool hasBias)
        where TRows : struct, IRows
    {
        int rows = TRows.Count;
        ref float a0 = ref Unsafe.Add(ref a, (nint)i * lda);
        ref float a1 = ref Unsafe.Add(ref a0, rows > 1 ? lda : 0);
        ref float a2 = ref Unsafe.Add(ref a0, rows > 2 ? 2 * (nint)lda : 0);
        ref float a3 = ref Unsafe.Add(ref a0, rows > 3 ? 3 * (nint)lda : 0);
        ref float a4 = ref Unsafe.Add(ref a0, rows > 4 ? 4 * (nint)lda : 0);
        ref float a5 = ref Unsafe.Add(ref a0, rows > 5 ? 5 * (nint)lda : 0);
        ref float bp = ref Unsafe.Add(ref b, j);

        Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        Vector256<float> c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;

        for (nint p = 0; p < k; p++)
        {
            var b0 = Vector256.LoadUnsafe(ref bp);
            var b1 = Vector256.LoadUnsafe(ref bp, 8);
            bp = ref Unsafe.Add(ref bp, ldb);
            var x = Vector256.Create(Unsafe.Add(ref a0, p));
            c00 = Vector256.FusedMultiplyAdd(x, b0, c00); c01 = Vector256.FusedMultiplyAdd(x, b1, c01);
            if (rows > 1) { x = Vector256.Create(Unsafe.Add(ref a1, p)); c10 = Vector256.FusedMultiplyAdd(x, b0, c10); c11 = Vector256.FusedMultiplyAdd(x, b1, c11); }
            if (rows > 2) { x = Vector256.Create(Unsafe.Add(ref a2, p)); c20 = Vector256.FusedMultiplyAdd(x, b0, c20); c21 = Vector256.FusedMultiplyAdd(x, b1, c21); }
            if (rows > 3) { x = Vector256.Create(Unsafe.Add(ref a3, p)); c30 = Vector256.FusedMultiplyAdd(x, b0, c30); c31 = Vector256.FusedMultiplyAdd(x, b1, c31); }
            if (rows > 4) { x = Vector256.Create(Unsafe.Add(ref a4, p)); c40 = Vector256.FusedMultiplyAdd(x, b0, c40); c41 = Vector256.FusedMultiplyAdd(x, b1, c41); }
            if (rows > 5) { x = Vector256.Create(Unsafe.Add(ref a5, p)); c50 = Vector256.FusedMultiplyAdd(x, b0, c50); c51 = Vector256.FusedMultiplyAdd(x, b1, c51); }
        }

        Vector256<float> bias0 = default, bias1 = default;
        if (hasBias)
        {
            bias0 = Vector256.LoadUnsafe(ref bias, (nuint)j);
            bias1 = Vector256.LoadUnsafe(ref bias, (nuint)j + 8);
        }
        ref float cr = ref Unsafe.Add(ref c, (nint)i * ldc + j);
        nuint s = (nuint)ldc;
        (bias0 + c00).StoreUnsafe(ref cr); (bias1 + c01).StoreUnsafe(ref cr, 8);
        if (rows > 1) { (bias0 + c10).StoreUnsafe(ref cr, s); (bias1 + c11).StoreUnsafe(ref cr, s + 8); }
        if (rows > 2) { (bias0 + c20).StoreUnsafe(ref cr, 2 * s); (bias1 + c21).StoreUnsafe(ref cr, 2 * s + 8); }
        if (rows > 3) { (bias0 + c30).StoreUnsafe(ref cr, 3 * s); (bias1 + c31).StoreUnsafe(ref cr, 3 * s + 8); }
        if (rows > 4) { (bias0 + c40).StoreUnsafe(ref cr, 4 * s); (bias1 + c41).StoreUnsafe(ref cr, 4 * s + 8); }
        if (rows > 5) { (bias0 + c50).StoreUnsafe(ref cr, 5 * s); (bias1 + c51).StoreUnsafe(ref cr, 5 * s + 8); }
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
}
