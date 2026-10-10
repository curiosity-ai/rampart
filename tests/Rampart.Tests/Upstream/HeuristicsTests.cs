using Rampart.Detection;

namespace Rampart.Tests.Upstream;

/// <summary>Port of <c>__tests__/heuristics.test.ts</c> (github.com/nationaldesignstudio/rampart).</summary>
public sealed class HeuristicsTests
{
    [Theory]
    [InlineData("888-12-3456")]
    [InlineData("888 12 3456")]
    [InlineData("888.12.3456")]
    [InlineData("888123456")]
    public void DetectsSsnAcrossSeparatorVariants(string variant)
    {
        var ssn = Heuristics.Detect($"my ssn is {variant}.").Find(s => s.Label == PiiLabel.Ssn);
        Assert.NotNull(ssn);
        Assert.Equal(variant, ssn.Text);
    }

    [Fact]
    public void RejectsStructurallyInvalidSsns() => Assert.DoesNotContain(Heuristics.Detect("000-12-3456"), s => s.Label == PiiLabel.Ssn);

    [Fact]
    public void MatchesALuhnValidCardWithSpaces() => Assert.Contains(Heuristics.Detect("card 4111 1111 1111 1111 ok"), s => s.Label == PiiLabel.CreditCard);

    [Fact]
    public void IgnoresASixteenDigitRunThatFailsLuhn() => Assert.DoesNotContain(Heuristics.Detect("1234 5678 1234 5678"), s => s.Label == PiiLabel.CreditCard);

    private static string? First(string text, PiiLabel label) => Heuristics.Detect(text).Find(s => s.Label == label)?.Text;

    [Fact]
    public void DetectsIpv4() => Assert.Equal("10.0.0.1", First("server 10.0.0.1 ok", PiiLabel.IpAddress));

    [Theory]
    [InlineData("2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    [InlineData("2001:db8::1")]
    [InlineData("fe80::1ff:fe23:4567:890a")]
    [InlineData("::1")]
    public void DetectsFullAndCompressedIpv6(string form) => Assert.Equal(form, First($"host {form} up", PiiLabel.IpAddress));

    [Fact]
    public void DetectsMacAddresses()
    {
        Assert.Equal("00:1B:44:11:3A:B7", First("mac 00:1B:44:11:3A:B7", PiiLabel.IpAddress));
        Assert.Equal("00-1B-44-11-3A-B7", First("mac 00-1B-44-11-3A-B7", PiiLabel.IpAddress));
    }

    [Fact]
    public void DoesNotFireIpv6OnClockTimesOrShortHex()
    {
        Assert.DoesNotContain(Heuristics.Detect("meet at 12:34:56 sharp"), s => s.Label == PiiLabel.IpAddress);
        Assert.DoesNotContain(Heuristics.Detect("opcode ff:00 set"), s => s.Label == PiiLabel.IpAddress);
    }

    [Fact]
    public void DetectsEmails()
    {
        Assert.Equal("a@b.com", First("reach me at a@b.com ok", PiiLabel.Email));
        Assert.Equal("alex+housing@sub.example.gov", First("send to alex+housing@sub.example.gov today", PiiLabel.Email));
        Assert.Equal("maria.backup@example.org", First("mail maria.backup@example.org now", PiiLabel.Email));
    }

    [Fact]
    public void DetectsUrls()
    {
        Assert.Equal("https://files.example.com/private/alex", First("see https://files.example.com/private/alex for the file", PiiLabel.Url));
        Assert.Equal("www.example.org/private/maya", First("visit www.example.org/private/maya please", PiiLabel.Url));
        Assert.DoesNotContain(Heuristics.Detect("the e.g. case, see U.S. law"), s => s.Label == PiiLabel.Url);
    }

    [Fact]
    public void LeavesPhonesAddressesAndIdsToTheModel()
    {
        var labels = Heuristics.Detect("call 415-555-2671, 31 Birchwood Avenue Old Lyme CT 06371").Select(s => s.Label).ToHashSet();
        Assert.DoesNotContain(PiiLabel.Phone, labels);
        Assert.DoesNotContain(PiiLabel.StreetName, labels);

        labels = Heuristics.Detect("case AGY-2026-009871 passport X12345678 license D1234567 bank account number is 123456789012 token sk-test-1234567890abcdef")
            .Select(s => s.Label).ToHashSet();
        foreach (var absent in new[] { PiiLabel.GovernmentId, PiiLabel.Passport, PiiLabel.DriversLicense, PiiLabel.BankAccount }) Assert.DoesNotContain(absent, labels);
    }
}
