namespace Catalyst.Rampart;

/// <summary>
/// The entity type names Rampart reports in <see cref="RecognizerResult.EntityType"/>, one constant
/// per <see cref="PiiLabel"/>, for comparing results without string literals.
/// <code>
/// var names = results.Where(r => r.EntityType is EntityTypes.GivenName or EntityTypes.Surname);
/// </code>
/// </summary>
public static class EntityTypes
{
    // Found by the deterministic recognizers, before the model.
    public const string Ssn = "SSN";
    public const string CreditCard = "CREDIT_CARD";
    public const string IpAddress = "IP_ADDRESS";
    public const string Email = "EMAIL";
    public const string Url = "URL";

    // Found by the model.
    public const string GivenName = "GIVEN_NAME";
    public const string Surname = "SURNAME";
    public const string Phone = "PHONE";
    public const string TaxId = "TAX_ID";
    public const string BankAccount = "BANK_ACCOUNT";
    public const string RoutingNumber = "ROUTING_NUMBER";
    public const string GovernmentId = "GOVERNMENT_ID";
    public const string Passport = "PASSPORT";
    public const string DriversLicense = "DRIVERS_LICENSE";
    public const string BuildingNumber = "BUILDING_NUMBER";
    public const string StreetName = "STREET_NAME";
    public const string SecondaryAddress = "SECONDARY_ADDRESS";

    // Found by the model, and kept in the text by default (RampartOptions.KeepLabels).
    public const string City = "CITY";
    public const string State = "STATE";
    public const string ZipCode = "ZIP_CODE";
}
