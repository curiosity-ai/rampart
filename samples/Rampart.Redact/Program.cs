using System.Diagnostics;
using Catalyst.Rampart;

// redact — an interactive terminal for the redaction engine, mirroring upstream's `bun run redact`
// (cli/redact.ts). Type a message; it prints what a cloaked call would send, the entities found and
// how long that took. Lines can also be piped in:
//   echo "My name is Jane Doe" | dotnet run --project samples/Rampart.Redact
const string Dim = "\x1b[2m", Bold = "\x1b[1m", Reset = "\x1b[0m";
static string Ms(double ms) => ms >= 100 ? $"{ms:F0} ms" : $"{ms:F1} ms";

bool interactive = !Console.IsInputRedirected;
var started = Stopwatch.StartNew();
var analyzer = new RampartAnalyzer();
analyzer.Analyze("warm up");
var cloaker = new RampartCloaker(); // one per session: the same value keeps its placeholder
if (interactive)
{
    Console.WriteLine($"{Bold}rampart{Reset}{Dim} · interactive redactor{Reset}");
    Console.WriteLine($"{Dim}ready in {Ms(started.Elapsed.TotalMilliseconds)} · Ctrl-C to quit{Reset}\n");
}

while (true)
{
    if (interactive) Console.Write("> ");
    var line = Console.ReadLine();
    if (line is null) break;
    var input = line.Trim();
    if (input.Length == 0) continue;

    var timer = Stopwatch.StartNew();
    var results = analyzer.Analyze(input);
    var elapsed = timer.Elapsed.TotalMilliseconds;

    // The "service" here just echoes what it was sent: the cloaked text.
    string sent = input;
    await cloaker.CloakAsync(input, results, cloaked => Task.FromResult(sent = cloaked));

    Console.WriteLine(sent);
    if (interactive)
    {
        foreach (var r in results)
            Console.WriteLine($"{Dim}  {r.EntityType,-17} {r.Score:0.00}  \"{input.Substring(r.Start, r.Length)}\"{Reset}");
        Console.WriteLine($"{Dim}analyzed in {Ms(elapsed)} · {results.Count} entit{(results.Count == 1 ? "y" : "ies")}{Reset}\n");
    }
}
