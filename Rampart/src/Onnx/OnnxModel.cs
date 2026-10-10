using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Rampart.Onnx;

/// <summary>ONNX TensorProto.DataType values this reader understands.</summary>
internal enum OnnxDataType
{
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    Int32 = 6,
    Int64 = 7,
}

/// <summary>An initializer or <c>Constant</c> node value read from an ONNX file.</summary>
internal sealed class OnnxTensor
{
    public required string Name { get; init; }
    public required OnnxDataType DataType { get; init; }
    public required long[] Dims { get; init; }
    /// <summary>Little-endian element bytes (raw_data, or the typed repeated field re-encoded).</summary>
    public required byte[] RawData { get; init; }

    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Dims) n *= d;
            return n;
        }
    }

    public float[] AsFloats()
    {
        Expect(OnnxDataType.Float);
        return MemoryMarshal.Cast<byte, float>(RawData).ToArray();
    }

    public byte[] AsBytes()
    {
        if (DataType is not (OnnxDataType.UInt8 or OnnxDataType.Int8)) Expect(OnnxDataType.UInt8);
        return RawData;
    }

    public long[] AsInt64s()
    {
        Expect(OnnxDataType.Int64);
        return MemoryMarshal.Cast<byte, long>(RawData).ToArray();
    }

    public float ScalarFloat() => AsFloats()[0];

    private void Expect(OnnxDataType type)
    {
        if (DataType != type) throw new InvalidDataException($"Tensor '{Name}' is {DataType}, expected {type}.");
    }
}

/// <summary>A graph node: just the parts the weight binder needs.</summary>
internal sealed class OnnxNode
{
    public required string OpType { get; init; }
    public required string Name { get; init; }
    public required string[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
    public required Dictionary<string, long> IntAttributes { get; init; }
    public required Dictionary<string, float> FloatAttributes { get; init; }
    public OnnxTensor? TensorAttribute { get; init; }
}

/// <summary>
/// A minimal, allocation-light protobuf reader for ONNX <c>ModelProto</c> files. It reads the
/// graph's nodes, initializers and <c>Constant</c> values — enough to bind the weights of the
/// shipped Rampart model without depending on ONNX Runtime or a protobuf library.
/// </summary>
internal sealed class OnnxModel
{
    public List<OnnxNode> Nodes { get; } = [];
    public Dictionary<string, OnnxTensor> Initializers { get; } = new(StringComparer.Ordinal);
    public List<string> GraphInputs { get; } = [];
    public List<string> GraphOutputs { get; } = [];

    public static OnnxModel Parse(ReadOnlySpan<byte> bytes)
    {
        var model = new OnnxModel();
        var reader = new ProtoReader(bytes);
        while (reader.TryReadTag(out var field, out var wire))
        {
            if (field == 7 && wire == WireType.LengthDelimited) model.ParseGraph(reader.ReadBytes());
            else reader.Skip(wire);
        }
        if (model.Nodes.Count == 0) throw new InvalidDataException("ONNX model has no graph nodes.");
        return model;
    }

    /// <summary>Initializer, or the value of the <c>Constant</c> node that outputs <paramref name="name"/>.</summary>
    public OnnxTensor GetTensor(string name)
    {
        if (Initializers.TryGetValue(name, out var t)) return t;
        throw new KeyNotFoundException($"ONNX tensor '{name}' not found.");
    }

    private void ParseGraph(ReadOnlySpan<byte> bytes)
    {
        var reader = new ProtoReader(bytes);
        while (reader.TryReadTag(out var field, out var wire))
        {
            switch (field)
            {
                case 1 when wire == WireType.LengthDelimited:
                    var node = ParseNode(reader.ReadBytes());
                    Nodes.Add(node);
                    if (node.OpType == "Constant" && node.TensorAttribute is { } value && node.Outputs.Length == 1)
                    {
                        Initializers[node.Outputs[0]] = new OnnxTensor { Name = node.Outputs[0], DataType = value.DataType, Dims = value.Dims, RawData = value.RawData };
                    }
                    break;
                case 5 when wire == WireType.LengthDelimited:
                    var tensor = ParseTensor(reader.ReadBytes());
                    Initializers[tensor.Name] = tensor;
                    break;
                case 11 when wire == WireType.LengthDelimited:
                    GraphInputs.Add(ParseValueInfoName(reader.ReadBytes()));
                    break;
                case 12 when wire == WireType.LengthDelimited:
                    GraphOutputs.Add(ParseValueInfoName(reader.ReadBytes()));
                    break;
                default:
                    reader.Skip(wire);
                    break;
            }
        }
    }

    private static string ParseValueInfoName(ReadOnlySpan<byte> bytes)
    {
        var reader = new ProtoReader(bytes);
        while (reader.TryReadTag(out var field, out var wire))
        {
            if (field == 1 && wire == WireType.LengthDelimited) return reader.ReadString();
            reader.Skip(wire);
        }
        return "";
    }

    private static OnnxNode ParseNode(ReadOnlySpan<byte> bytes)
    {
        var reader = new ProtoReader(bytes);
        var inputs = new List<string>();
        var outputs = new List<string>();
        string name = "", opType = "";
        var ints = new Dictionary<string, long>(StringComparer.Ordinal);
        var floats = new Dictionary<string, float>(StringComparer.Ordinal);
        OnnxTensor? tensorAttr = null;

        while (reader.TryReadTag(out var field, out var wire))
        {
            switch (field)
            {
                case 1 when wire == WireType.LengthDelimited: inputs.Add(reader.ReadString()); break;
                case 2 when wire == WireType.LengthDelimited: outputs.Add(reader.ReadString()); break;
                case 3 when wire == WireType.LengthDelimited: name = reader.ReadString(); break;
                case 4 when wire == WireType.LengthDelimited: opType = reader.ReadString(); break;
                case 5 when wire == WireType.LengthDelimited:
                    ParseAttribute(reader.ReadBytes(), ints, floats, ref tensorAttr);
                    break;
                default: reader.Skip(wire); break;
            }
        }

        return new OnnxNode
        {
            Name = name,
            OpType = opType,
            Inputs = [.. inputs],
            Outputs = [.. outputs],
            IntAttributes = ints,
            FloatAttributes = floats,
            TensorAttribute = tensorAttr,
        };
    }

    private static void ParseAttribute(ReadOnlySpan<byte> bytes, Dictionary<string, long> ints, Dictionary<string, float> floats, ref OnnxTensor? tensor)
    {
        var reader = new ProtoReader(bytes);
        string name = "";
        long? i = null;
        float? f = null;
        OnnxTensor? t = null;
        while (reader.TryReadTag(out var field, out var wire))
        {
            switch (field)
            {
                case 1 when wire == WireType.LengthDelimited: name = reader.ReadString(); break;
                case 2 when wire == WireType.Fixed32: f = reader.ReadFloat(); break;
                case 3 when wire == WireType.Varint: i = (long)reader.ReadVarint(); break;
                case 5 when wire == WireType.LengthDelimited: t = ParseTensor(reader.ReadBytes()); break;
                default: reader.Skip(wire); break;
            }
        }
        if (i is { } iv) ints[name] = iv;
        if (f is { } fv) floats[name] = fv;
        if (t is not null && name == "value") tensor = t;
    }

    private static OnnxTensor ParseTensor(ReadOnlySpan<byte> bytes)
    {
        var reader = new ProtoReader(bytes);
        var dims = new List<long>();
        var dataType = 0;
        string name = "";
        byte[]? raw = null;
        List<float>? floatData = null;
        List<long>? int64Data = null;
        List<int>? int32Data = null;

        while (reader.TryReadTag(out var field, out var wire))
        {
            switch (field)
            {
                case 1:
                    if (wire == WireType.LengthDelimited)
                    {
                        var packed = new ProtoReader(reader.ReadBytes());
                        while (!packed.End) dims.Add((long)packed.ReadVarint());
                    }
                    else dims.Add((long)reader.ReadVarint());
                    break;
                case 2: dataType = (int)reader.ReadVarint(); break;
                case 4:
                    floatData ??= [];
                    if (wire == WireType.LengthDelimited)
                    {
                        var packed = reader.ReadBytes();
                        for (int k = 0; k + 4 <= packed.Length; k += 4) floatData.Add(BinaryPrimitives.ReadSingleLittleEndian(packed[k..]));
                    }
                    else floatData.Add(reader.ReadFloat());
                    break;
                case 5:
                    int32Data ??= [];
                    if (wire == WireType.LengthDelimited)
                    {
                        var packed = new ProtoReader(reader.ReadBytes());
                        while (!packed.End) int32Data.Add((int)packed.ReadVarint());
                    }
                    else int32Data.Add((int)reader.ReadVarint());
                    break;
                case 7:
                    int64Data ??= [];
                    if (wire == WireType.LengthDelimited)
                    {
                        var packed = new ProtoReader(reader.ReadBytes());
                        while (!packed.End) int64Data.Add((long)packed.ReadVarint());
                    }
                    else int64Data.Add((long)reader.ReadVarint());
                    break;
                case 8 when wire == WireType.LengthDelimited: name = reader.ReadString(); break;
                case 9 when wire == WireType.LengthDelimited: raw = reader.ReadBytes().ToArray(); break;
                case 13: throw new NotSupportedException("ONNX external data is not supported.");
                default: reader.Skip(wire); break;
            }
        }

        var type = (OnnxDataType)dataType;
        raw ??= type switch
        {
            OnnxDataType.Float => MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(floatData ?? [])).ToArray(),
            OnnxDataType.Int64 => MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(int64Data ?? [])).ToArray(),
            OnnxDataType.Int32 => MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(int32Data ?? [])).ToArray(),
            // uint8/int8 typed data is stored widened in int32_data.
            OnnxDataType.UInt8 or OnnxDataType.Int8 => (int32Data ?? []).Select(v => (byte)v).ToArray(),
            _ => [],
        };

        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Big-endian platforms are not supported.");
        return new OnnxTensor { Name = name, DataType = type, Dims = [.. dims], RawData = raw };
    }
}

internal enum WireType
{
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    StartGroup = 3,
    EndGroup = 4,
    Fixed32 = 5,
}

/// <summary>Forward-only protobuf wire-format reader over a span.</summary>
internal ref struct ProtoReader(ReadOnlySpan<byte> data)
{
    private readonly ReadOnlySpan<byte> _data = data;
    private int _pos;

    public readonly bool End => _pos >= _data.Length;

    public readonly int Position => _pos;

    public bool TryReadTag(out int field, out WireType wire)
    {
        if (End)
        {
            field = 0;
            wire = 0;
            return false;
        }
        var tag = ReadVarint();
        field = (int)(tag >> 3);
        wire = (WireType)(tag & 7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            if (_pos >= _data.Length) throw new InvalidDataException("Truncated protobuf varint.");
            byte b = _data[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
        }
        throw new InvalidDataException("Malformed protobuf varint.");
    }

    public float ReadFloat()
    {
        var v = BinaryPrimitives.ReadSingleLittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return v;
    }

    public ReadOnlySpan<byte> ReadBytes()
    {
        var len = checked((int)ReadVarint());
        if (len < 0 || _pos + len > _data.Length) throw new InvalidDataException("Truncated protobuf field.");
        var slice = _data.Slice(_pos, len);
        _pos += len;
        return slice;
    }

    public string ReadString() => Encoding.UTF8.GetString(ReadBytes());

    public void Skip(WireType wire)
    {
        switch (wire)
        {
            case WireType.Varint: ReadVarint(); break;
            case WireType.Fixed64: _pos += 8; break;
            case WireType.LengthDelimited: ReadBytes(); break;
            case WireType.Fixed32: _pos += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }
}
