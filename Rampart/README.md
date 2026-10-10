# Rampart for .NET

On-device PII detection and reversible redaction for .NET 10. A native C# port of
[National Design Studio's Rampart](https://github.com/nationaldesignstudio/rampart) with the
[published model](https://huggingface.co/nationaldesignstudio/rampart) embedded in the package.

- **Before** a message goes to an LLM, names, phone numbers, IDs, card numbers, addresses and other
  PII are replaced with typed placeholders (`[GIVEN_NAME_1]`, `[SSN_1]`, …).
- **After**, the reply is turned back into the real values, also while it streams in.
- **On the device:** no ONNX Runtime, no Python and no network calls. The 4-bit MiniLM classifier
  runs in managed code with SIMD kernels, and a forward pass allocates nothing.

```
dotnet add package Rampart
```

## Quick start

```csharp
using Rampart;

var guard = new ChatGuard();   // one per conversation

var safe = guard.Protect("My name is Alex Rivera and my SSN is 472-81-0094.");
// safe.Text         → "My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1]."
// safe.Placeholders → [GIVEN_NAME_1], [SURNAME_1], [SSN_1]

// Send safe.Text to the model. The provider never sees the real values.

guard.Reveal("Thanks [GIVEN_NAME_1], your SSN [SSN_1] is on file.");
// → "Thanks Alex, your SSN 472-81-0094 is on file."
```

The first call loads the embedded model, which takes about half a second. After that a chat-sized
message takes ~5 ms on a 4-core machine (~11 ms on one thread).

## Samples

### Protecting an LLM call (`Microsoft.Extensions.AI`)

Any `IChatClient` (OpenAI, Azure OpenAI, Ollama, …) works the same way: protect the user's
message, send the placeholders, reveal the answer.

```csharp
using Microsoft.Extensions.AI;
using Rampart;

static async Task<string> AskAsync(IChatClient client, ChatGuard guard, string userMessage, CancellationToken ct = default)
{
    var safe = guard.Protect(userMessage);
    var response = await client.GetResponseAsync(safe.Text, cancellationToken: ct);
    return guard.Reveal(response.Text);
}
```

### Streaming replies

A placeholder can arrive split across chunks (`"Hello [GIV"`, `"EN_NAME_1]"`). `RevealAsync`
holds back a partial placeholder until it is complete, so the user never sees a fragment.

```csharp
static async Task StreamAsync(IChatClient client, ChatGuard guard, string userMessage, CancellationToken ct = default)
{
    var safe = guard.Protect(userMessage);
    var chunks = client.GetStreamingResponseAsync(safe.Text, cancellationToken: ct).Select(u => u.Text);

    await foreach (var chunk in guard.RevealAsync(chunks, ct))
        Console.Write(chunk);   // "Hello Alex Rivera!"
}
```

When you push chunks yourself, from a WebSocket or server-sent events for instance, use a
`StreamingReveal`:

```csharp
var reveal = guard.CreateStreamingReveal();
Send(reveal.Push("Hello [GIVEN_"));                          // "Hello "
Send(reveal.Push("NAME_1] [SURNAME_1], how can I help?"));    // "Alex Rivera, how can I help?"
Send(reveal.Flush());                                         // whatever was still held back
```

### Multi-turn conversations

A `ChatGuard` keeps one table per conversation, so the same value gets the same placeholder on
every turn and the model can refer back to it:

```csharp
var chat = new ChatGuard();

chat.Protect("Please call Jordan Smith at (202) 555-0188.").Text;
// → "Please call [GIVEN_NAME_1] [SURNAME_1] at [PHONE_1]."

chat.Protect("Jordan Smith lives at 12 Elm Street, Apt 4B, Denver, CO 80202.").Text;
// → "[GIVEN_NAME_1] [SURNAME_1] lives at [BUILDING_NUMBER_1] [STREET_NAME_1], [SECONDARY_ADDRESS_1], Denver, CO 80202."

foreach (var (placeholder, value) in chat.Table.Entries)
    Console.WriteLine($"{placeholder} = {value}");   // [PHONE_1] = (202) 555-0188, [SURNAME_1] = Smith, …
```

City, state and ZIP code are kept by default (see [Policy](#policy)). To also scrub what the
assistant writes, for example before storing a transcript, call `ProtectReply`. It shares the
same table:

```csharp
var guard = new ChatGuard();
guard.Protect("My name is Alex Rivera.");

guard.ProtectReply("I'll forward your request to Priya Shah, Alex Rivera.").Text;
// → "I'll forward your request to [GIVEN_NAME_2] [SURNAME_2], [GIVEN_NAME_1] [SURNAME_1]."
```

### Detection only

`RampartAnalyzer` is stateless and thread-safe; it returns the spans and does not rewrite anything.

```csharp
var analyzer = new RampartAnalyzer();
var text = "You can reach Maria Lopez at (415) 555-0132 or maria.lopez@example.com, she lives in Boston.";

foreach (var span in analyzer.Analyze(text))
    Console.WriteLine($"{span.Label.ToName(),-11} {span.Score:0.00}  {span.Source,-9}  \"{span.Text}\" [{span.Start}, {span.End})");

// GIVEN_NAME  0.97  Ner        "Maria" [14, 19)
// SURNAME     0.95  Ner        "Lopez" [20, 25)
// PHONE       1.00  Ner        "(415) 555-0132" [29, 43)
// EMAIL       1.00  Heuristic  "maria.lopez@example.com" [47, 70)
// CITY        0.96  Ner        "Boston" [85, 91)
```

`Analyze` returns the merged, non-overlapping spans, including the kept labels; `Detect` returns
the raw spans of both layers. Offsets are UTF-16 code units: `Start` is inclusive and `End`
exclusive.

### One-way redaction for logs

When nothing has to be restored (logs, analytics, traces), use `Redact`:

```csharp
analyzer.Redact(text);
// → "You can reach [GIVEN_NAME] [SURNAME] at [PHONE] or [EMAIL], she lives in Boston."

analyzer.Redact(text, RedactionMode.Mask);
// → "You can reach ***** ***** at ************** or ***********************, she lives in Boston."

analyzer.Redact("Ticket opened by maria.lopez@example.com", RedactionMode.Hash);
// → "Ticket opened by ceea7b686c43f044fffaab488a94d05dca6d2a7fb3b79b7c0adfb3e43574f7bf"
```

`RedactionMode.Remove` drops the spans. `Redactor.Redact(text, spans, …)` redacts spans you
already have.

For structured data only, skip the model. The deterministic recognizers find SSNs, card numbers
(Luhn-checked), emails, URLs and IP/MAC addresses in a few microseconds per message:

```csharp
var fast = new RampartAnalyzer(new RampartOptions { HeuristicsOnly = true });
fast.Redact("Server 10.0.0.12 logged in priya.shah@example.org, card 4111 1111 1111 1111");
// → "Server [IP_ADDRESS] logged in [EMAIL], card [CREDIT_CARD]"
```

### Policy

Everything detected is redacted except `CITY`, `STATE` and `ZIP_CODE`, which are useful context
and rarely identifying on their own. Change that, and rename the placeholders, with
`RampartOptions`:

```csharp
var strict = new ChatGuard(new RampartOptions
{
    KeepLabels = [],   // keep nothing
    Aliases = new Dictionary<PiiLabel, string>
    {
        [PiiLabel.GivenName] = "NAME",
        [PiiLabel.Surname] = "NAME",
    },
    MinScore = 0.4,    // the model's confidence threshold (default)
});

strict.Protect("My name is Maria Lopez and I live at 12 Elm Street, Denver, CO 80202.").Text;
// → "My name is [NAME_2] [NAME_1] and I live at [BUILDING_NUMBER_1] [STREET_NAME_1], [CITY_1], [STATE_1] [ZIP_CODE_1]."
```

Within a message, numbers are given out from right to left, as in the original. They identify a
value; they are not positions.

### Other languages

The model is trained on English, Spanish, French, German, Italian, Portuguese and Dutch:

```csharp
new ChatGuard().Protect("Me llamo Lucía Fernández y vivo en la Calle Mayor 5, Madrid.").Text;
// → "Me llamo [GIVEN_NAME_1] [SURNAME_1] y vivo en la [STREET_NAME_1] [BUILDING_NUMBER_1], Madrid."
```

### ASP.NET Core and dependency injection

Share one analyzer (the model is ~5 MB of weights, loaded once) and create a guard per
conversation:

```csharp
using Rampart;
using Rampart.Model;

builder.Services.AddSingleton(_ => new RampartAnalyzer(new RampartOptions
{
    // A server already runs many requests in parallel, so give each inference one thread.
    Model = RampartModel.Load(new RampartModelOptions { MaxDegreeOfParallelism = 1 }),
}));
builder.Services.AddScoped(sp => new ChatGuard(sp.GetRequiredService<RampartAnalyzer>()));
```

A `ChatGuard` is not thread-safe and holds the real values of its conversation. Keep it with the
conversation (scoped, or in your session store), never as a singleton shared between users.

By default `RampartModel.Default` uses every core for one inference, which is the right choice
for a desktop app or a CLI. With `MaxDegreeOfParallelism = 1` each request stays on its own thread,
which gives the best throughput under concurrent load.

### Loading the model from disk

The model ships inside the assembly. To use a local copy of the Hugging Face repository instead
(the same files, for example a newer revision), load it from a directory:

```csharp
var model = RampartModel.LoadFromDirectory("/models/rampart");   // onnx/model_q4.onnx, vocab.txt, config.json
var guard = new ChatGuard(new RampartOptions { Model = model });
```

## What gets detected

| Layer | Labels |
|---|---|
| Deterministic, before the model | `SSN`, `CREDIT_CARD`, `EMAIL`, `URL`, `IP_ADDRESS` |
| Model, redacted by default | `GIVEN_NAME`, `SURNAME`, `PHONE`, `TAX_ID`, `BANK_ACCOUNT`, `ROUTING_NUMBER`, `GOVERNMENT_ID`, `PASSPORT`, `DRIVERS_LICENSE`, `BUILDING_NUMBER`, `STREET_NAME`, `SECONDARY_ADDRESS` |
| Model, kept by default | `CITY`, `STATE`, `ZIP_CODE` |

## Limitations

Rampart reduces harm; it is not a security boundary. This package reproduces the original
implementation exactly, including its misses. Read the
[model card](https://huggingface.co/nationaldesignstudio/rampart) before relying on it. In
particular:

- **Names can be missed, or picked up with a neighbouring word.** In
  `"Ask Alex Rivera about it"` nothing is found. In `"Hi, I'm Alex Rivera"` the given name comes out
  as `"I'm Alex"`: the rule that keeps particles such as "De La" with a name also takes in any short
  capitalized word right before it. That redacts too much, but it does not leak anything.
- Recall on names in non-Latin scripts is low.
- Government IDs without a checksum are found only by the model.
- Adversarial input (zero-width characters, deliberate misspellings, prompt injection) can evade
  detection.

## Licenses

The code is MIT. The embedded model files (`model_q4.onnx`, `vocab.txt`, `config.json`) are
© National Design Studio under CC BY 4.0; the package carries both licenses and the third-party
notices. Source, validation and benchmarks: <https://github.com/curiosity-ai/rampart>.
