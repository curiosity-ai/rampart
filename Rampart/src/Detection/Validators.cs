namespace Catalyst.Rampart.Detection;

/// <summary>
/// Validators that turn "a run of digits" into "a <em>valid</em> SSN / card". They suppress false
/// positives so the filter does not mangle numbers the assistant is allowed to keep (income
/// figures, ages, years): a detector proposes a span, a validator decides whether it is real.
/// </summary>
internal static class Validators
{
    /// <summary>Luhn checksum — gates CREDIT_CARD so arbitrary 16-digit runs don't match.</summary>
    public static bool IsLuhnValid(ReadOnlySpan<char> digits)
    {
        int sum = 0;
        bool dbl = false;
        for (int i = digits.Length - 1; i >= 0; i--)
        {
            int d = digits[i] - '0';
            if (dbl)
            {
                d *= 2;
                if (d > 9) d -= 9;
            }
            sum += d;
            dbl = !dbl;
        }
        return sum % 10 == 0;
    }

    /// <summary>
    /// US SSN structural rules: area (first 3) cannot be 000, 666 or 900–999; group (middle 2)
    /// cannot be 00; serial (last 4) cannot be 0000.
    /// </summary>
    public static bool IsValidSsn(ReadOnlySpan<char> digits)
    {
        if (digits.Length != 9) return false;
        var area = digits[..3];
        if (area is "000" or "666") return false;
        if (area[0] == '9') return false;
        if (digits.Slice(3, 2) is "00") return false;
        if (digits[5..] is "0000") return false;
        return true;
    }
}
