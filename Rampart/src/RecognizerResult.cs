namespace Rampart;

/// <summary>
/// One recognized PII entity, the shape Catalyst.Presidio uses: offsets into the analyzed text
/// (UTF-16 code units, <see cref="Start"/> and <see cref="End"/> both inclusive), the entity type
/// (<c>GIVEN_NAME</c>, <c>SSN</c>, …) and the recognizer's confidence.
/// </summary>
public readonly struct RecognizerResult
{
    public RecognizerResult(int start, int end, string entityType, float score)
    {
        Start = start;
        End = end;
        EntityType = entityType;
        Score = score;
    }

    /// <summary>The start index of the identified PII (inclusive).</summary>
    public int Start { get; }

    /// <summary>The end index of the identified PII (inclusive).</summary>
    public int End { get; }

    /// <summary>The confidence score of the recognition, in [0, 1].</summary>
    public float Score { get; }

    /// <summary>The type of the identified PII (e.g. <c>GIVEN_NAME</c>, <c>EMAIL</c>).</summary>
    public string EntityType { get; }

    /// <summary>The length of the identified PII.</summary>
    public int Length => End - Start + 1;

    public override string ToString() => $"{EntityType} [{Start}, {End}] {Score:0.00}";
}
