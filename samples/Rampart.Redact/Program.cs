using System.Diagnostics;
using Rampart;
using Rampart.Model;

// redact — an interactive terminal for the redaction engine, mirroring upstream's `bun run redact`
// (cli/redact.ts). Type a message; it prints the locally redacted text and how long that took.
// Lines can also be piped in: echo "My name is Jane Doe" | dotnet run --project samples/Rampart.Redact
const string Dim = "\x1b[2m", Bold = "\x1b[1m", Reset = "\x1b[0m";
static string Ms(double ms) => ms >= 100 ? $"{ms:F0} ms" : $"{ms:F1} ms";

bool interactive = !Console.IsInputRedirected;
var started = Stopwatch.StartNew();
var guard = new ChatGuard(new RampartOptions { Model = RampartModel.Default });
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
    var result = guard.Protect(input);
    var elapsed = timer.Elapsed.TotalMilliseconds;

    Console.WriteLine(result.Text);
    int count = result.Placeholders.Count;
    if (interactive) Console.WriteLine($"{Dim}redacted in {Ms(elapsed)} · {count} placeholder{(count == 1 ? "" : "s")}{Reset}\n");
}
