using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Rampart.Model;

/// <summary>
/// A linear layer whose weight is stored in ONNX <c>com.microsoft.MatMulNBits</c> form: 4-bit
/// unsigned values packed two per byte (low nibble first), grouped along K into blocks of
/// <see cref="BlockSize"/> with one float scale (and an optional 4-bit zero point, default 8) per block.
/// <para>
/// The packed weights are kept as shipped (≈4.5 bits per weight). A forward pass dequantizes a tile of
/// output rows (SIMD nibble unpack) into a small scratch buffer once and reuses it for every token in
/// the sequence, so the dequantization cost is amortised over the sequence, and multiplies with the
/// register-blocked <see cref="Gemm"/> kernel. Tiles run in parallel.
/// </para>
/// </summary>
internal sealed class Q4Linear
{
    private const int RowTile = 32;

    /// <summary>Lookup from a packed byte to its two unsigned nibbles (low, high) as floats.</summary>
    private static readonly float[] NibblePairs = BuildNibblePairs();

    public int N { get; }
    public int K { get; }
    public int BlockSize { get; }
    public int BlocksPerRow { get; }
    private readonly byte[] _packed;      // [N, BlocksPerRow, BlockSize / 2]
    private readonly float[] _scales;     // [N, BlocksPerRow]
    private readonly byte[]? _zeroPoints; // [N, ceil(BlocksPerRow / 2)] packed 4-bit, or null -> 8
    private readonly float[] _bias;       // [N]
    private float[]? _panels;             // [tiles][K][RowTile]: dequantized, transposed; null → dequantize per call

    /// <summary>True when the weights have been expanded into float panels (see <see cref="ExpandWeights"/>).</summary>
    public bool IsExpanded => _panels is not null;

    /// <summary>
    /// Dequantizes every weight once into the transposed float panels the GEMM consumes, trading
    /// 8× the weight memory (4 bytes instead of ~4.5 bits per weight) for not dequantizing on every call.
    /// </summary>
    public void ExpandWeights()
    {
        if (_panels is not null) return;
        int tiles = (N + RowTile - 1) / RowTile;
        var panels = new float[tiles * K * RowTile];
        Parallel.For(0, tiles, tile =>
        {
            var row = new float[K];
            var panel = panels.AsSpan(tile * K * RowTile, K * RowTile);
            FillPanel(tile, panel, row);
        });
        _panels = panels;
    }

    private void FillPanel(int tile, Span<float> panel, Span<float> row)
    {
        int n0 = tile * RowTile;
        int rows = Math.Min(RowTile, N - n0);
        for (int r = 0; r < rows; r++)
        {
            DequantizeRow(n0 + r, row);
            for (int k = 0; k < K; k++) panel[k * RowTile + r] = row[k];
        }
    }

    public Q4Linear(int n, int k, int blockSize, byte[] packed, float[] scales, byte[]? zeroPoints, float[] bias)
    {
        if (blockSize <= 0 || blockSize % 2 != 0) throw new InvalidDataException($"Unsupported MatMulNBits block size {blockSize}.");
        N = n;
        K = k;
        BlockSize = blockSize;
        BlocksPerRow = (k + blockSize - 1) / blockSize;
        if (packed.Length != n * BlocksPerRow * (blockSize / 2)) throw new InvalidDataException("MatMulNBits weight has an unexpected size.");
        if (scales.Length != n * BlocksPerRow) throw new InvalidDataException("MatMulNBits scales have an unexpected size.");
        if (bias.Length != n) throw new InvalidDataException("Linear bias has an unexpected size.");
        _packed = packed;
        _scales = scales;
        _zeroPoints = zeroPoints;
        _bias = bias;
    }

    /// <summary>Dequantizes output row <paramref name="row"/> (K weights) into <paramref name="destination"/>.</summary>
    public void DequantizeRow(int row, Span<float> destination)
    {
        var bytesPerBlock = BlockSize / 2;
        var packedRow = _packed.AsSpan(row * BlocksPerRow * bytesPerBlock, BlocksPerRow * bytesPerBlock);
        var scales = _scales.AsSpan(row * BlocksPerRow, BlocksPerRow);
        bool simd = BlockSize == 32 && (Sse2.IsSupported || AdvSimd.Arm64.IsSupported);
        ref var pairs = ref MemoryMarshal.GetArrayDataReference(NibblePairs);

        for (int b = 0; b < BlocksPerRow; b++)
        {
            float scale = scales[b];
            float zero = ZeroPoint(row, b);
            var block = packedRow.Slice(b * bytesPerBlock, bytesPerBlock);
            int kBase = b * BlockSize;
            int count = Math.Min(BlockSize, K - kBase);
            var dst = destination.Slice(kBase, count);

            if (simd && count == 32)
            {
                DequantizeBlock32(block, zero, scale, dst);
                continue;
            }
            for (int i = 0; i < count; i += 2)
            {
                int p = block[i >> 1] * 2;
                dst[i] = (Unsafe.Add(ref pairs, p) - zero) * scale;
                if (i + 1 < count) dst[i + 1] = (Unsafe.Add(ref pairs, p + 1) - zero) * scale;
            }
        }
    }

    /// <summary>
    /// 16 packed bytes → 32 floats: split the nibbles, interleave them back into K order (low nibble
    /// first), widen to float and apply <c>(q - zero) · scale</c> — the same two roundings as the scalar path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeBlock32(ReadOnlySpan<byte> block, float zero, float scale, Span<float> dst)
    {
        var bytes = Vector128.Create(block);
        var lo = bytes & Vector128.Create((byte)0x0F);
        var hi = Vector128.ShiftRightLogical(bytes, 4);
        Vector128<byte> first, second;
        if (Sse2.IsSupported)
        {
            first = Sse2.UnpackLow(lo, hi);
            second = Sse2.UnpackHigh(lo, hi);
        }
        else
        {
            first = AdvSimd.Arm64.ZipLow(lo, hi);
            second = AdvSimd.Arm64.ZipHigh(lo, hi);
        }

        var z = Vector128.Create(zero);
        var s = Vector128.Create(scale);
        ref float d = ref MemoryMarshal.GetReference(dst);
        WidenStore(first, z, s, ref d, 0);
        WidenStore(second, z, s, ref d, 16);

        static void WidenStore(Vector128<byte> v, Vector128<float> z, Vector128<float> s, ref float d, nuint offset)
        {
            var (u0, u1) = Vector128.Widen(v);
            var (w0, w1) = Vector128.Widen(u0);
            var (w2, w3) = Vector128.Widen(u1);
            ((Vector128.ConvertToSingle(w0.AsInt32()) - z) * s).StoreUnsafe(ref d, offset);
            ((Vector128.ConvertToSingle(w1.AsInt32()) - z) * s).StoreUnsafe(ref d, offset + 4);
            ((Vector128.ConvertToSingle(w2.AsInt32()) - z) * s).StoreUnsafe(ref d, offset + 8);
            ((Vector128.ConvertToSingle(w3.AsInt32()) - z) * s).StoreUnsafe(ref d, offset + 12);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ZeroPoint(int row, int block)
    {
        if (_zeroPoints is null) return 8f;
        int stride = (BlocksPerRow + 1) / 2;
        byte packed = _zeroPoints[row * stride + (block >> 1)];
        return (block & 1) == 0 ? packed & 0x0F : packed >> 4;
    }

    /// <summary>
    /// <c>y[t, n] = bias[n] + Σ_k x[t, k] · W[n, k]</c> for a row-major <paramref name="x"/> of
    /// <paramref name="tokens"/> × <see cref="K"/>; writes <paramref name="tokens"/> × <see cref="N"/> into <paramref name="y"/>.
    /// </summary>
    public void Forward(ReadOnlySpan<float> x, int tokens, Span<float> y)
    {
        if (x.Length < tokens * K) throw new ArgumentException("Input is smaller than tokens x K.", nameof(x));
        if (y.Length < tokens * N) throw new ArgumentException("Output is smaller than tokens x N.", nameof(y));

        int tiles = (N + RowTile - 1) / RowTile;
        // Tasks are (token block, column tile) pairs, column tile varying fastest, so a worker that takes a
        // contiguous range of tasks keeps its block of activations in its own L2 while panels stream past.
        // With 4-bit weights a panel is dequantized per task, so the tokens are not split.
        int tokenBlock = _panels is null ? tokens : TokenBlock;
        int blocks = (tokens + tokenBlock - 1) / tokenBlock;
        int tasks = blocks * tiles;

        if ((long)tokens * N * K < 500_000 || tasks == 1)
        {
            var scratch = ArrayPool<float>.Shared.Rent((RowTile + 1) * K);
            try
            {
                for (int task = 0; task < tasks; task++) ForwardTask(x, tokens, y, task, tiles, tokenBlock, scratch);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scratch);
            }
            return;
        }

        // Spans cannot be captured by the parallel lambda; pin and pass raw pointers instead.
        unsafe
        {
            fixed (float* px = x)
            fixed (float* py = y)
            {
                var xp = (nint)px;
                var yp = (nint)py;
                int xLen = tokens * K, yLen = tokens * N;
                Parallel.For(0, tasks,
                    () => ArrayPool<float>.Shared.Rent((RowTile + 1) * K),
                    (task, _, scratch) =>
                    {
                        var xs = new ReadOnlySpan<float>((void*)xp, xLen);
                        var ys = new Span<float>((void*)yp, yLen);
                        ForwardTask(xs, tokens, ys, task, tiles, tokenBlock, scratch);
                        return scratch;
                    },
                    scratch => ArrayPool<float>.Shared.Return(scratch));
            }
        }
    }

    /// <summary>Tokens per task when the weights are expanded: a multiple of the GEMM's 6-row block.</summary>
    private const int TokenBlock = 96;

    /// <summary>
    /// Multiplies one block of tokens against the <c>K × RowTile</c> panel holding weight rows
    /// <c>[n0, n0 + RowTile)</c> transposed — precomputed, or dequantized into <paramref name="scratch"/> now.
    /// </summary>
    private void ForwardTask(ReadOnlySpan<float> x, int tokens, Span<float> y, int task, int tiles, int tokenBlock, float[] scratch)
    {
        int tile = task % tiles;
        int t0 = task / tiles * tokenBlock;
        int count = Math.Min(tokenBlock, tokens - t0);
        int n0 = tile * RowTile;
        int rows = Math.Min(RowTile, N - n0);
        ReadOnlySpan<float> panel;
        if (_panels is not null)
        {
            panel = _panels.AsSpan(tile * K * RowTile, K * RowTile);
        }
        else
        {
            var p = scratch.AsSpan(0, K * RowTile);
            FillPanel(tile, p, scratch.AsSpan(K * RowTile, K));
            panel = p;
        }
        Gemm.Multiply(x[(t0 * K)..], K, panel, RowTile, y[(t0 * N + n0)..], N, count, rows, K, _bias.AsSpan(n0, rows));
    }

    private static float[] BuildNibblePairs()
    {
        var table = new float[512];
        for (int b = 0; b < 256; b++)
        {
            table[2 * b] = b & 0x0F;
            table[2 * b + 1] = b >> 4;
        }
        return table;
    }
}
