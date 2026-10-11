# Rampart for .NET

On-device PII detection, anonymization and reversible cloaking for .NET 10, with the same flow as
[Catalyst.Presidio](https://github.com/curiosity-ai/catalyst/tree/master/Catalyst.Presidio): an
**analyzer** finds the entities, an **anonymizer** masks them for good, and a **cloaker** swaps them
for placeholders around a call to an external service and puts them back in its answer.

Behind the analyzer is a native C# port of
[National Design Studio's Rampart](https://github.com/nationaldesignstudio/rampart): deterministic
recognizers plus the [Rampart model](https://huggingface.co/nationaldesignstudio/rampart), a 4-bit
MiniLM token classifier. The model is embedded in the package and runs in managed code, with no ONNX
Runtime, Python or network calls.

```
dotnet add package Rampart
```

## Quick start

```csharp
using Rampart;

var analyzer = new RampartAnalyzer();     // thread-safe, share one
var anonymizer = new RampartAnonymizer();
var cloaker = new RampartCloaker();       // one per conversation

var text = "My name is Alex Rivera and my SSN is 472-81-0094.";
var results = analyzer.Analyze(text);

anonymizer.Anonymize(text, results);
// → "My name is **** ****** and my SSN is ***********."

var answer = await cloaker.CloakAsync(text, results, async cloaked =>
{
    // cloaked: "My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1]."
    return await CallServiceAsync(cloaked);   // your LLM or API call; it never sees the real values
});
// The service answered "Thanks [GIVEN_NAME_1], your SSN [SSN_1] is on file."
// answer: "Thanks Alex, your SSN 472-81-0094 is on file."
```

The first call loads the embedded model, which takes about half a second. After that a chat-sized
message takes ~5 ms on a 4-core machine (~11 ms on one thread).

## Analyzer

`Analyze` returns the entities as a `List<RecognizerResult>`, ordered by position and never
overlapping:

```csharp
var text = "You can reach Maria Lopez at (415) 555-0132 or maria.lopez@example.com, she lives in Boston.";

foreach (var r in analyzer.Analyze(text))
    Console.WriteLine($"{r.EntityType,-10} {r.Score:0.00}  [{r.Start}, {r.End}]  \"{text.Substring(r.Start, r.Length)}\"");

// GIVEN_NAME 0.97  [14, 18]  "Maria"
// SURNAME    0.95  [20, 24]  "Lopez"
// PHONE      1.00  [29, 42]  "(415) 555-0132"
// EMAIL      1.00  [47, 69]  "maria.lopez@example.com"
```

`RecognizerResult` has the same shape as Catalyst.Presidio's:
- `Start` and `End` are both inclusive, so `Length` is `End - Start + 1`. Offsets are UTF-16 code
  units.
- `EntityType` is one of the names in [Entity types](#entity-types).
- `Score` is the confidence, from 0 to 1. The deterministic recognizers report 1.

"Boston" is missing on purpose: city, state and ZIP code are left in the text by default (see
[Options](#options)).

## Anonymizer

One-way: nothing can be restored afterwards. The operators are the same as Catalyst.Presidio's:

```csharp
var results = analyzer.Analyze(text);

anonymizer.Anonymize(text, results);                                   // "mask" is the default
// → "You can reach ***** ***** at ************** or ***********************, she lives in Boston."

anonymizer.Anonymize(text, results, "mask", maskChar: "#");
// → "You can reach ##### ##### at ############## or #######################, she lives in Boston."

anonymizer.Anonymize(text, results, "replace");
// → "You can reach <REDACTED> <REDACTED> at <REDACTED> or <REDACTED>, she lives in Boston."

anonymizer.Anonymize(text, results, "replace", replacement: "[PII]");
// → "You can reach [PII] [PII] at [PII] or [PII], she lives in Boston."

anonymizer.Anonymize(text, results, "redact");
// → "You can reach   at  or , she lives in Boston."

anonymizer.Anonymize("Ticket opened by maria.lopez@example.com", analyzer.Analyze("Ticket opened by maria.lopez@example.com"), "hash");
// → "Ticket opened by ceea7b686c43f044fffaab488a94d05dca6d2a7fb3b79b7c0adfb3e43574f7bf"   (SHA-256)
```

An unknown operator masks. Overlapping results keep the earliest one, and results outside the text
are skipped.

## Cloaker

`CloakAsync` replaces each result with a typed placeholder, calls your function with the cloaked
text, and returns its answer with the original values put back.

- **Stable placeholders.** A cloaker remembers what it has cloaked, so the same value gets the same
  placeholder on every call and the service can refer back to it. Use one cloaker per conversation.
  It is not thread-safe.
- **Case and spacing.** These are ignored when matching values: "Alex" and "alex" get the same
  placeholder.

```csharp
var conversation = new RampartCloaker();

// sends "Please call [GIVEN_NAME_1] [SURNAME_1] at [PHONE_1]."
var message = "Please call Jordan Smith at (202) 555-0188.";
await conversation.CloakAsync(message, analyzer.Analyze(message), CallServiceAsync);

// sends "[GIVEN_NAME_1] [SURNAME_1] lives at [BUILDING_NUMBER_1] [STREET_NAME_1], [SECONDARY_ADDRESS_1], Denver, CO 80202."
message = "Jordan Smith lives at 12 Elm Street, Apt 4B, Denver, CO 80202.";
await conversation.CloakAsync(message, analyzer.Analyze(message), CallServiceAsync);
```

### Streaming

For a service that streams its answer, pass a function that returns `IAsyncEnumerable<string>`.
The chunks come back with the original values restored. When a placeholder is split across chunks
(`"Hello [GIV"`, `"EN_NAME_1]"`), it is held back until it is complete, so the reader never sees a
fragment:

```csharp
await foreach (var chunk in cloaker.CloakAsync(text, analyzer.Analyze(text), cloaked => StreamServiceAsync(cloaked)))
    Console.Write(chunk);
// The service streamed "Hello [GIV", "EN_NAME_1] [SUR", "NAME_1], how can I help?"
// → Hello Alex Rivera, how can I help?
```

### What the cloaker accepts

- **Overlapping results** are merged, so no part of an entity is left in the cloaked text.
- **Errors instead of leaks.** An entity type the analyzer does not report throws `ArgumentException`.
  A result outside the text throws `ArgumentOutOfRangeException`. Either way nothing is sent.

## Options

```csharp
using Rampart.Model;

var analyzer = new RampartAnalyzer(new RampartOptions
{
    KeepLabels = [],     // report everything, including CITY, STATE and ZIP_CODE
    MinScore = 0.4,      // the model's confidence threshold (default 0.4)
    Model = RampartModel.Load(new RampartModelOptions { MaxDegreeOfParallelism = 1 }),
});
```

| Option | Default | |
|---|---|---|
| `KeepLabels` | `City`, `State`, `ZipCode` | Entity types the analyzer leaves in the text. Pass `[]` to report everything. |
| `MinScore` | `0.4` | Model results below this score are dropped, unless they join up with a confident neighbour. |
| `Model` | `RampartModel.Default` | The embedded model, loaded once and shared by every analyzer. |
| `RampartModelOptions.MaxDegreeOfParallelism` | all cores | Cores one analysis may use. 1 keeps each call on its own thread, the best choice when many analyses run at once. |

`RampartModel.LoadFromDirectory(path)` loads a local copy of the Hugging Face repository instead of
the embedded one (`onnx/model_q4.onnx`, `vocab.txt`, `config.json`).

The model is trained on English, Spanish, French, German, Italian, Portuguese and Dutch:

```csharp
var text = "Me llamo Lucía Fernández y vivo en la Calle Mayor 5, Madrid.";
anonymizer.Anonymize(text, analyzer.Analyze(text), "replace", replacement: "<PII>");
// → "Me llamo <PII> <PII> y vivo en la <PII> <PII>, Madrid."
```

## Per-call options

`Analyze` takes an optional `AnalyzeOptions` for that call. Without it, or with the defaults,
the result is exactly the original Rampart's. An instance can be reused; its hints are compiled
once.

### The capitalized-particle expansion

To keep names like "De La Croix" or "Van Der Berg" together, the original extends a detected name
over a short capitalized word next to it (a capital and up to three more letters). That also takes
in words like "I'm" or "Call":

```csharp
var text = "Hi, I'm Alex Rivera.";

anonymizer.Anonymize(text, analyzer.Analyze(text), "replace", replacement: "<PII>");
// → "Hi, <PII> <PII>."             ("I'm Alex" is one GIVEN_NAME)

var options = new AnalyzeOptions { ExpandCapitalizedParticles = false };
anonymizer.Anonymize(text, analyzer.Analyze(text, options), "replace", replacement: "<PII>");
// → "Hi, I'm <PII> <PII>."
```

Turning the expansion off trades extra redaction for possible leaks. A short part of a name that
the model did not tag is then left in the text:

```csharp
var text = "Juan De La Vega Ruiz signed.";
analyzer.Analyze(text, new AnalyzeOptions { ExpandCapitalizedParticles = false });
// → GIVEN_NAME "Juan", SURNAME "Ruiz"     ("De La Vega" is not reported)
```

Hints, below, close that gap for the names you know.

### Hints: values to always report

When you know values that must never get through, such as the name of the person sending the
message, pass them as hints by entity type:

```csharp
var sender = new AnalyzeOptions
{
    Hints = new Dictionary<string, string[]>
    {
        ["GIVEN_NAME"] = ["Alex"],
        ["SURNAME"] = ["Rivera"],
    },
};

var text = "Ask alex rivera about it, he will know.";
anonymizer.Anonymize(text, analyzer.Analyze(text), "replace", replacement: "<PII>");
// → "Ask alex rivera about it, he will know."     (the model finds nothing)
anonymizer.Anonymize(text, analyzer.Analyze(text, sender), "replace", replacement: "<PII>");
// → "Ask <PII> <PII> about it, he will know."
```

- **Always reported.** Every occurrence of a hinted value is reported as its entity type, with
  score 1. That holds even when the model misses it, and even when the type is in `KeepLabels`.
- **Whole words.** The value's words must match whole words, with any spacing between them: "Al"
  does not match "Also".
- **Case.** Matching ignores case by default. For a hint that is also a common word ("Will", "Mark",
  "May"), set `HintsIgnoreCase = false`.
- **Unknown types.** An entity type the analyzer does not report throws `ArgumentException`.

The words of a type's hints also **expand** spans of that type. A detected span grows over the
hinted words right before it, then over the ones right after it. Unlike the capitalized-particle
expansion, this works for any word length and any case, and it works with that expansion off. The
two name types share their words, because the model splits one person's name across `GIVEN_NAME`
and `SURNAME`:

```csharp
var text = "Juan De La Vega Ruiz signed.";
var options = new AnalyzeOptions
{
    ExpandCapitalizedParticles = false,
    Hints = new Dictionary<string, string[]> { ["SURNAME"] = ["De La Vega Ruiz"] },
};
analyzer.Analyze(text, options);
// → GIVEN_NAME "Juan", SURNAME "De La Vega Ruiz"
```

The expansion applies wherever those words stand next to a detected span of the type, even when
the whole value does not appear. For example, the same hint gives "Maria De La Vega Ruiz" as
GIVEN_NAME "Maria" and SURNAME "De La Vega Ruiz".

## Entity types

| Found by | Entity types |
|---|---|
| Deterministic recognizers, before the model | `SSN`, `CREDIT_CARD` (Luhn-checked), `EMAIL`, `URL`, `IP_ADDRESS` (IPv4, IPv6, MAC) |
| The model | `GIVEN_NAME`, `SURNAME`, `PHONE`, `TAX_ID`, `BANK_ACCOUNT`, `ROUTING_NUMBER`, `GOVERNMENT_ID`, `PASSPORT`, `DRIVERS_LICENSE`, `BUILDING_NUMBER`, `STREET_NAME`, `SECONDARY_ADDRESS` |
| The model, kept by default | `CITY`, `STATE`, `ZIP_CODE` |

## Limitations

Rampart reduces harm; it is not a security boundary. This package reproduces the original
implementation exactly, including its misses. Read the
[model card](https://huggingface.co/nationaldesignstudio/rampart) before relying on it. In
particular:

- **Names can be missed, or picked up with a neighbouring word.** In
  `"Ask Alex Rivera about it"` nothing is found. In `"Hi, I'm Alex Rivera"` the given name comes out
  as `"I'm Alex"`: the rule that keeps particles such as "De La" with a name also takes in any short
  capitalized word right before it. That redacts too much but leaks nothing. For names you know,
  [hints](#hints-values-to-always-report) fix both. The expansion can also be
  [turned off](#the-capitalized-particle-expansion).
- Recall on names in non-Latin scripts is low.
- Government IDs without a checksum are found only by the model.
- Adversarial input (zero-width characters, deliberate misspellings, prompt injection) can evade
  detection.

## Licenses

The code is MIT. The embedded model files (`model_q4.onnx`, `vocab.txt`, `config.json`) are
© National Design Studio under CC BY 4.0; the package carries both licenses and the third-party
notices. Source, validation and benchmarks: <https://github.com/curiosity-ai/rampart>.
