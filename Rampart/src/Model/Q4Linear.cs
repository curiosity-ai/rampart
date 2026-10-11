using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Catalyst.Rampart.Model;

/// <summary>
/// A linear layer whose weight ships in ONNX <c>com.microsoft.MatMulNBits</c> form: 4-bit unsigned
/// values, grouped along K into blocks of <see cref="BlockSize"/> with one float scale (and an
/// optional 4-bit zero point, default 8) per block and output row.
/// <para>
/// At load the 4-bit values are <em>permuted</em> — not re-quantized — into panel order: for each tile
/// of 32 output rows and each k, the 32 nibbles sit in 16 consecutive bytes, and the scales and zero
/// points of a tile are stored transposed. A forward pass then dequantizes one tile at a time, a few
/// SIMD instructions per k, into a <c>K × 32</c> float panel (≤ 196 KB, cache-resident) and multiplies
/// every token against it with the <see cref="Gemm"/> kernel. The weights stay at ~4.5 bits each
/// (≈5 MB for the whole model, so they stay in the last-level cache between calls), and each panel is
/// dequantized once per call however long the sequence.
/// </para>
/// </summary>
internal sealed class Q4Linear
{
    /// <summary>Output rows per panel: one <see cref="Gemm"/> column block on 512-bit hardware.</summary>
    public const int Tile = 32;

    public int N { get; }
    public int K { get; }
    public int BlockSize { get; }
    public int BlocksPerRow { get; }
    public int Tiles { get; }

    private readonly byte[] _nibbles; // [Tiles][K][Tile / 2]: byte r/2 of (tile, k) holds row r (low) and r + 1 (high)
    private readonly float[] _scales; // [Tiles][BlocksPerRow][Tile], 0 for padding rows
    private readonly float[] _zeros;  // [Tiles][BlocksPerRow][Tile]
    private readonly float[] _bias;   // [N]

    public Q4Linear(int n, int k, int blockSize, byte[] packed, float[] scales, byte[]? zeroPoints, float[] bias)
    {
        if (blockSize <= 0 || blockSize % 2 != 0) throw new InvalidDataException($"Unsupported MatMulNBits block size {blockSize}.");
        N = n;
        K = k;
        BlockSize = blockSize;
        BlocksPerRow = (k + blockSize - 1) / blockSize;
        Tiles = (n + Tile - 1) / Tile;
        int bytesPerBlock = blockSize / 2;
        if (packed.Length != n * BlocksPerRow * bytesPerBlock) throw new InvalidDataException("MatMulNBits weight has an unexpected size.");
        if (scales.Length != n * BlocksPerRow) throw new InvalidDataException("MatMulNBits scales have an unexpected size.");
        if (bias.Length != n) throw new InvalidDataException("Linear bias has an unexpected size.");
        _bias = bias;

        // Permute into panel order. Padding rows of the last tile get scale 0, so they dequantize to 0.
        _nibbles = new byte[Tiles * K * (Tile / 2)];
        _scales = new float[Tiles * BlocksPerRow * Tile];
        _zeros = new float[Tiles * BlocksPerRow * Tile];
        _zeros.AsSpan().Fill(8f);
        int zeroStride = (BlocksPerRow + 1) / 2;
        for (int row = 0; row < n; row++)
        {
            int tile = row / Tile, r = row % Tile;
            for (int b = 0; b < BlocksPerRow; b++)
            {
                int at = (tile * BlocksPerRow + b) * Tile + r;
                _scales[at] = scales[row * BlocksPerRow + b];
                if (zeroPoints is not null)
                {
                    byte z = zeroPoints[row * zeroStride + (b >> 1)];
                    _zeros[at] = (b & 1) == 0 ? z & 0x0F : z >> 4;
                }
            }
            for (int kk = 0; kk < k; kk++)
            {
                int b = kk / blockSize, i = kk % blockSize;
                byte src = packed[(row * BlocksPerRow + b) * bytesPerBlock + (i >> 1)];
                int q = (i & 1) == 0 ? src & 0x0F : src >> 4;
                ref byte dst = ref _nibbles[(tile * K + kk) * (Tile / 2) + (r >> 1)];
                dst |= (byte)((r & 1) == 0 ? q : q << 4);
            }
        }
    }

    /// <summary>The dequantized weight <c>W[row, k]</c> as ONNX Runtime computes it: <c>(q - zero) · scale</c>.</summary>
    public float Weight(int row, int k)
    {
        int tile = row / Tile, r = row % Tile;
        byte packed = _nibbles[(tile * K + k) * (Tile / 2) + (r >> 1)];
        int q = (r & 1) == 0 ? packed & 0x0F : packed >> 4;
        int at = (tile * BlocksPerRow + k / BlockSize) * Tile + r;
        return (q - _zeros[at]) * _scales[at];
    }

    /// <summary>
    /// Dequantizes tile <paramref name="tile"/> into <paramref name="panel"/> laid out <c>[K][Tile]</c>:
    /// panel row k is <c>W[tile·32 .. tile·32 + 31, k]</c>, the B operand of <c>y = x · B</c>.
    /// </summary>
    public void DequantizeTile(int tile, Span<float> panel)
    {
        if (panel.Length < K * Tile) throw new ArgumentException("Panel is smaller than K x Tile.", nameof(panel));
        ref byte nib = ref _nibbles[tile * K * (Tile / 2)];
        ref float sc = ref _scales[tile * BlocksPerRow * Tile];
        ref float ze = ref _zeros[tile * BlocksPerRow * Tile];
        ref float dst = ref MemoryMarshal.GetReference(panel);
        bool x86 = Sse2.IsSupported, arm = AdvSimd.Arm64.IsSupported;

        for (int b = 0; b < BlocksPerRow; b++)
        {
            int k0 = b * BlockSize, k1 = Math.Min(K, k0 + BlockSize);
            nuint o = (nuint)(b * Tile);
            if (!x86 && !arm)
            {
                for (int k = k0; k < k1; k++)
                {
                    for (int r = 0; r < Tile; r++)
                    {
                        byte packed = Unsafe.Add(ref nib, k * (Tile / 2) + (r >> 1));
                        int q = (r & 1) == 0 ? packed & 0x0F : packed >> 4;
                        Unsafe.Add(ref dst, k * Tile + r) = (q - Unsafe.Add(ref ze, o + (nuint)r)) * Unsafe.Add(ref sc, o + (nuint)r);
                    }
                }
                continue;
            }

            if (Avx512F.IsSupported)
            {
                var z0 = Vector512.LoadUnsafe(ref ze, o);
                var z1 = Vector512.LoadUnsafe(ref ze, o + 16);
                var s0 = Vector512.LoadUnsafe(ref sc, o);
                var s1 = Vector512.LoadUnsafe(ref sc, o + 16);
                for (int k = k0; k < k1; k++)
                {
                    var (first, second) = Unpack(Vector128.LoadUnsafe(ref nib, (nuint)(k * (Tile / 2))), x86);
                    var w0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(first));
                    var w1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(second));
                    ((w0 - z0) * s0).StoreUnsafe(ref dst, (nuint)(k * Tile));
                    ((w1 - z1) * s1).StoreUnsafe(ref dst, (nuint)(k * Tile + 16));
                }
            }
            else
            {
                for (int k = k0; k < k1; k++)
                {
                    var (first, second) = Unpack(Vector128.LoadUnsafe(ref nib, (nuint)(k * (Tile / 2))), x86);
                    WidenStore(first, ref ze, ref sc, o, ref dst, (nuint)(k * Tile));
                    WidenStore(second, ref ze, ref sc, o + 16, ref dst, (nuint)(k * Tile + 16));
                }
            }
        }

        static void WidenStore(Vector128<byte> v, ref float ze, ref float sc, nuint o, ref float d, nuint at)
        {
            var (u0, u1) = Vector128.Widen(v);
            var (w0, w1) = Vector128.Widen(u0);
            var (w2, w3) = Vector128.Widen(u1);
            ((Vector128.ConvertToSingle(w0.AsInt32()) - Vector128.LoadUnsafe(ref ze, o)) * Vector128.LoadUnsafe(ref sc, o)).StoreUnsafe(ref d, at);
            ((Vector128.ConvertToSingle(w1.AsInt32()) - Vector128.LoadUnsafe(ref ze, o + 4)) * Vector128.LoadUnsafe(ref sc, o + 4)).StoreUnsafe(ref d, at + 4);
            ((Vector128.ConvertToSingle(w2.AsInt32()) - Vector128.LoadUnsafe(ref ze, o + 8)) * Vector128.LoadUnsafe(ref sc, o + 8)).StoreUnsafe(ref d, at + 8);
            ((Vector128.ConvertToSingle(w3.AsInt32()) - Vector128.LoadUnsafe(ref ze, o + 12)) * Vector128.LoadUnsafe(ref sc, o + 12)).StoreUnsafe(ref d, at + 12);
        }
    }

    /// <summary>16 packed bytes → the 32 nibbles in row order (low nibble first), as two byte vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector128<byte> First, Vector128<byte> Second) Unpack(Vector128<byte> bytes, bool x86)
    {
        var lo = bytes & Vector128.Create((byte)0x0F);
        var hi = Vector128.ShiftRightLogical(bytes, 4);
        return x86 ? (Sse2.UnpackLow(lo, hi), Sse2.UnpackHigh(lo, hi)) : (AdvSimd.Arm64.ZipLow(lo, hi), AdvSimd.Arm64.ZipHigh(lo, hi));
    }

    /// <summary>
    /// <c>y[t, n] = bias[n] + Σ_k x[t, k] · W[n, k]</c> for a row-major <paramref name="x"/> of
    /// <paramref name="tokens"/> × <see cref="K"/>; writes <paramref name="tokens"/> × <see cref="N"/> into
    /// <paramref name="y"/>. Each tile is dequantized once and multiplied against every token; tiles are
    /// spread over up to <paramref name="parallelism"/> cores.
    /// </summary>
    public void Forward(ReadOnlySpan<float> x, int tokens, Span<float> y, int parallelism = 1)
    {
        if (x.Length < tokens * K) throw new ArgumentException("Input is smaller than tokens x K.", nameof(x));
        if (y.Length < tokens * N) throw new ArgumentException("Output is smaller than tokens x N.", nameof(y));
        if (tokens == 0) return;

        if (parallelism <= 1 || Tiles == 1 || (long)tokens * N * K < 1_000_000)
        {
            var panel = ArrayPool<float>.Shared.Rent(K * Tile);
            try
            {
                for (int tile = 0; tile < Tiles; tile++) ForwardTile(x, tokens, y, tile, panel);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(panel);
            }
            return;
        }

        unsafe
        {
            fixed (float* px = x)
            fixed (float* py = y)
            {
                var job = new TileJob(this, (nint)px, (nint)py, tokens);
                Parallelism.For(Tiles, parallelism, ref job);
            }
        }
    }

    private void ForwardTile(ReadOnlySpan<float> x, int tokens, Span<float> y, int tile, float[] panel)
    {
        int n0 = tile * Tile;
        int rows = Math.Min(Tile, N - n0);
        DequantizeTile(tile, panel);
        Gemm.Multiply(x, K, panel, Tile, y[n0..], N, tokens, rows, K, _bias.AsSpan(n0, rows));
    }

    private readonly unsafe struct TileJob(Q4Linear linear, nint x, nint y, int tokens) : IParallelJob
    {
        public void Run(int from, int to)
        {
            var panel = ArrayPool<float>.Shared.Rent(linear.K * Tile);
            try
            {
                var xs = new ReadOnlySpan<float>((void*)x, tokens * linear.K);
                var ys = new Span<float>((void*)y, tokens * linear.N);
                for (int tile = from; tile < to; tile++) linear.ForwardTile(xs, tokens, ys, tile, panel);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(panel);
            }
        }
    }
}
