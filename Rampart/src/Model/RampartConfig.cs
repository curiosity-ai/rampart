using System.Text.Json;
using System.Text.Json.Serialization;

namespace Rampart.Model;

/// <summary>The subset of the Hugging Face <c>config.json</c> the encoder needs.</summary>
internal sealed class RampartConfig
{
    [JsonPropertyName("hidden_size")] public int HiddenSize { get; init; }
    [JsonPropertyName("num_hidden_layers")] public int NumHiddenLayers { get; init; }
    [JsonPropertyName("num_attention_heads")] public int NumAttentionHeads { get; init; }
    [JsonPropertyName("intermediate_size")] public int IntermediateSize { get; init; }
    [JsonPropertyName("max_position_embeddings")] public int MaxPositionEmbeddings { get; init; }
    [JsonPropertyName("vocab_size")] public int VocabSize { get; init; }
    [JsonPropertyName("layer_norm_eps")] public double LayerNormEps { get; init; }
    [JsonPropertyName("id2label")] public Dictionary<string, string> Id2Label { get; init; } = [];

    /// <summary>BIO labels by class index (<c>O</c>, <c>B-GIVEN_NAME</c>, …).</summary>
    [JsonIgnore] public IReadOnlyList<string> Labels { get; private set; } = [];

    public static RampartConfig Parse(ReadOnlySpan<byte> json)
    {
        var config = JsonSerializer.Deserialize(json, RampartConfigJsonContext.Default.RampartConfig)
            ?? throw new InvalidDataException("Empty Rampart config.json.");
        var labels = new string[config.Id2Label.Count];
        foreach (var (key, value) in config.Id2Label) labels[int.Parse(key, System.Globalization.CultureInfo.InvariantCulture)] = value;
        if (labels.Any(l => l is null)) throw new InvalidDataException("config.json id2label is not dense.");
        config.Labels = labels;
        return config;
    }
}

[JsonSerializable(typeof(RampartConfig))]
internal sealed partial class RampartConfigJsonContext : JsonSerializerContext;
