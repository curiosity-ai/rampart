/** Inputs beyond the package's public eval cases: languages, formats, offsets and edge cases. */
export const EXTRA_CASES: { id: string; text: string }[] = [
  { id: "x-basic", text: "My name is Alex Rivera and I live at 12 Main Street, Springfield." },
  { id: "x-es", text: "Hola, me llamo José Ángel García López y vivo en la Calle Mayor 5, 3º B, Madrid. Mi teléfono es 612 345 678." },
  { id: "x-de", text: "Ich heiße Jürgen Müller, meine Steuer-ID ist 12 345 678 901 und ich wohne in der Hauptstraße 7, 10115 Berlin." },
  { id: "x-fr", text: "Je m'appelle Élodie Dubois, mon numéro de passeport est 18AV34567 et j'habite 4 rue de la Paix à Paris." },
  { id: "x-it", text: "Mi chiamo Giulia Bianchi, codice fiscale BNCGLI85M41H501Z, abito in Via Roma 21, Milano." },
  { id: "x-pt", text: "Meu nome é João Pedro Gonçalves, CPF 123.456.789-09, moro na Rua das Flores 88, Lisboa." },
  { id: "x-nl", text: "Mijn naam is Sanne de Vries en mijn rekeningnummer is NL91ABNA0417164300." },
  { id: "x-ids", text: "Driver's license D1234567, passport 912803456, routing 021000021, account 000123456789." },
  { id: "x-email-url", text: "Write to jane.doe+benefits@mail.example.gov or see https://example.com/profile?id=42 and www.example.org/path." },
  { id: "x-ip-mac", text: "Login from 192.168.10.24 and 2001:db8::8a2e:370:7334, device 00:1A:2B:3C:4D:5E at 12:34." },
  { id: "x-card-forms", text: "Cards: 4111-1111-1111-1111, 378282246310005, and 4111 1111 1111 1112 (invalid)." },
  { id: "x-ssn-forms", text: "SSNs 472 81 0094, 472.81.0094, 472810094; not 000-12-3456 or 666-12-3456 or 912-34-5678." },
  { id: "x-initials", text: "Contact J. R. Smith or John F. Kennedy Jr. about the claim." },
  { id: "x-particles", text: "The tenants are Maria de los Santos, Pieter van der Berg and Ludwig von Beethoven." },
  { id: "x-apostrophe", text: "Seán O'Brien and D'Angelo Russell share apartment 4B at 77 Elm St." },
  { id: "x-sentence-boundary", text: "I spoke with Garcia. I think the case is closed. After that Chen called." },
  { id: "x-accents", text: "Ångström Øresund and Zoë Saldaña met Dvořák in Kraków." },
  { id: "x-unicode", text: "ＦＵＬＬＷＩＤＴＨ Ｊｏｈｎ, ligature ﬁnance, emoji 🙂 Ana 🙂 Silva, CJK 中文 王小明, tab\tand\nnewline." },
  { id: "x-specials", text: "[CLS] weird [SEP] tokens [MASK] for Mary [UNK] Jones [PAD]" },
  { id: "x-placeholder-like", text: "My name is [GIVEN_NAME_1] and my friend is Robert Brown." },
  { id: "x-sentinel-like", text: "I typed [SSN] myself, and my real one is 472-81-0094." },
  { id: "x-hyphens", text: "Anne-Marie Jean-Luc O'Neil-Smith lives at 12-14 Rue Saint-Honoré." },
  { id: "x-phone-forms", text: "Call +1 (415) 555-0132, 415.555.0198, or +44 20 7946 0958 after 5pm." },
  { id: "x-no-pii", text: "What documents do I need to apply for housing assistance if my income is $2,300 a month?" },
  { id: "x-empty", text: "" },
  { id: "x-space", text: "   " },
  { id: "x-single", text: "a" },
  { id: "x-long-word", text: "Supercalifragilisticexpialidocious" + "x".repeat(120) + " and then Emily Clark." },
  { id: "x-zero-width", text: "My name is Al​ex Ri‌vera and my SSN is 472‍-81-0094." },
  { id: "x-greek", text: "ΣΟΦΙΑ ΠΑΠΑΔΟΠΟΥΛΟΣ and Ὀδυσσεύς" },
  { id: "x-turkish", text: "İsmail Yılmaz lives in İzmir." },
  { id: "x-address-full", text: "Ship it to 1600 Pennsylvania Ave NW, Suite 200, Washington, DC 20500." },
  { id: "x-multi-person", text: "Household: Carlos Mendoza (father), Ana Mendoza (mother), and Luis Mendoza (son)." },
  { id: "x-tax", text: "My EIN is 12-3456789 and my ITIN is 912-70-1234." },
];

export const CONVERSATIONS = [
  {
    id: "conv-stable-placeholders",
    turns: [
      {
        user: "Hi, I'm Alex Rivera. My SSN is 472-81-0094.",
        reply: "Thanks [GIVEN_NAME_1]. I have noted the SSN [SSN_1].",
      },
      {
        user: "My wife is Maria Rivera and we live at 31 Birchwood Avenue, Old Lyme CT 06371.",
        reply: "Got it: [GIVEN_NAME_2] [SURNAME_1] at [BUILDING_NUMBER_1] [STREET_NAME_1]. Unknown [PHONE_9] stays.",
      },
      {
        user: "alex rivera again, call me at (202) 555-0188.",
        reply: "[GIVEN_NAME_1], we will call [PHONE_1].",
      },
    ],
  },
];

/** A long input, well past one 512-token window, for the windowing path. */
export function longText(): string {
  const parts: string[] = [];
  for (let i = 0; i < 60; i++) {
    parts.push(
      `Entry ${i}: the applicant Maria Gonzalez${i % 3} called from (415) 555-01${i % 10}${i % 7} about her benefits at 4${i} Oak Street, Apt ${i}B.`,
    );
  }
  return parts.join(" ");
}
