namespace Rampart.Tests;

/// <summary>Inputs shared by the parity tests: several languages, the premask sentinels, edge cases.</summary>
internal static class Corpus
{
    public static readonly string[] Sentences =
    [
        "My name is Alex Rivera and I live at 12 Main Street, Springfield.",
        "My name is Alex Rivera, my email is [EMAIL], and my monthly income is $1,950.",
        "My SSN is [SSN] and you can call me at (202) 555-0188.",
        "I live at 31 Birchwood Avenue Old Lyme CT 06371 and make $45,000.",
        "Please update the record for Jean Baptiste De La Croix.",
        "Hola, me llamo José Ángel García López y vivo en la Calle Mayor 5, 3º B, Madrid.",
        "Ich heiße Jürgen Müller, meine Steuer-ID ist 12 345 678 901 und ich wohne in der Hauptstraße 7.",
        "Je m'appelle Élodie Dubois, mon numéro de passeport est 18AV34567.",
        "Mijn naam is Sanne de Vries en mijn rekeningnummer is NL91ABNA0417164300.",
        "Driver's license D1234567, passport 912803456, routing 021000021.",
        "",
        "a",
        "[CLS] weird [SEP] tokens [MASK] and [UNK] inline",
        "ＦＵＬＬＷＩＤＴＨ ｔｅｘｔ and ligatures ﬁnance, emoji 🙂 and 中文字符 mixed in.",
    ];

    /// <summary>A long input (well past one 512-token window) for the windowing path.</summary>
    public static string Long()
    {
        var parts = new List<string>();
        for (int i = 0; i < 60; i++)
        {
            parts.Add($"Entry {i}: the applicant Maria Gonzalez{i % 3} called from (415) 555-01{i % 10}{i % 7} about her benefits at 4{i} Oak Street, Apt {i}B.");
        }
        return string.Join(" ", parts);
    }
}
