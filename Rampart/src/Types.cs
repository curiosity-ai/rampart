using System.Collections.Frozen;

namespace Rampart;

/// <summary>Every entity class the heuristics and the NER model can emit.</summary>
public enum PiiLabel
{
    // --- Structured (heuristic-detectable, premasked before the model) ---
    Ssn,
    CreditCard,
    IpAddress,
    // --- Contextual (NER model fine set) ---
    GivenName,
    Surname,
    Email,
    Phone,
    Url,
    TaxId,
    BankAccount,
    RoutingNumber,
    GovernmentId,
    Passport,
    DriversLicense,
    // --- Address components (NER) ---
    BuildingNumber,
    StreetName,
    SecondaryAddress,
    City,
    State,
    ZipCode,
}

/// <summary>Which layer produced a span; used for merge tie-breaks and audit.</summary>
internal enum SpanSource
{
    Heuristic,
    Ner,
}

/// <summary>
/// A detected entity over the <em>original</em> (raw) text. Offsets are UTF-16 code units, with
/// <see cref="Start"/> inclusive and <see cref="End"/> exclusive.
/// </summary>
/// <param name="Start">Inclusive start offset into the raw input.</param>
/// <param name="End">Exclusive end offset into the raw input.</param>
/// <param name="Label">The classified entity type.</param>
/// <param name="Score">Detector confidence in [0, 1]. Heuristics with validators report 1.</param>
/// <param name="Source">Which layer produced the span.</param>
/// <param name="Text">The raw substring covered, retained for placeholder rehydration.</param>
internal sealed record PiiSpan(int Start, int End, PiiLabel Label, double Score, SpanSource Source, string Text)
{
    public int Length => End - Start;
}

internal static class PiiLabels
{
    private static readonly FrozenDictionary<PiiLabel, string> Names = new Dictionary<PiiLabel, string>
    {
        [PiiLabel.Ssn] = "SSN",
        [PiiLabel.CreditCard] = "CREDIT_CARD",
        [PiiLabel.IpAddress] = "IP_ADDRESS",
        [PiiLabel.GivenName] = "GIVEN_NAME",
        [PiiLabel.Surname] = "SURNAME",
        [PiiLabel.Email] = "EMAIL",
        [PiiLabel.Phone] = "PHONE",
        [PiiLabel.Url] = "URL",
        [PiiLabel.TaxId] = "TAX_ID",
        [PiiLabel.BankAccount] = "BANK_ACCOUNT",
        [PiiLabel.RoutingNumber] = "ROUTING_NUMBER",
        [PiiLabel.GovernmentId] = "GOVERNMENT_ID",
        [PiiLabel.Passport] = "PASSPORT",
        [PiiLabel.DriversLicense] = "DRIVERS_LICENSE",
        [PiiLabel.BuildingNumber] = "BUILDING_NUMBER",
        [PiiLabel.StreetName] = "STREET_NAME",
        [PiiLabel.SecondaryAddress] = "SECONDARY_ADDRESS",
        [PiiLabel.City] = "CITY",
        [PiiLabel.State] = "STATE",
        [PiiLabel.ZipCode] = "ZIP_CODE",
    }.ToFrozenDictionary();

    private static readonly FrozenDictionary<string, PiiLabel> ByName = Names.ToFrozenDictionary(kv => kv.Value, kv => kv.Key, StringComparer.Ordinal);

    /// <summary>
    /// Labels that are intentionally preserved (classified but never redacted): broad geography
    /// (city / state / ZIP) is kept for eligibility questions while the street line is redacted.
    /// </summary>
    public static readonly FrozenSet<PiiLabel> DefaultKeepLabels = new[] { PiiLabel.City, PiiLabel.State, PiiLabel.ZipCode }.ToFrozenSet();

    /// <summary>The canonical upper-snake name used in sentinels and placeholders (<c>GIVEN_NAME</c>).</summary>
    public static string ToName(this PiiLabel label) => Names[label];

    /// <summary>Parses a canonical label name (<c>GIVEN_NAME</c>).</summary>
    public static bool TryParse(string name, out PiiLabel label) => ByName.TryGetValue(name, out label);

    /// <summary>Resolve a caller keep-set; <c>null</c> → <see cref="DefaultKeepLabels"/>.</summary>
    public static IReadOnlySet<PiiLabel> ResolveKeepLabels(IEnumerable<PiiLabel>? keepLabels)
        => keepLabels is null ? DefaultKeepLabels : keepLabels.ToFrozenSet();

    /// <summary>True when a label must be redacted under the default-deny policy.</summary>
    public static bool ShouldRedact(PiiLabel label, IReadOnlySet<PiiLabel>? keepLabels = null)
        => !(keepLabels ?? DefaultKeepLabels).Contains(label);
}
