using Rampart.Session;

namespace Rampart;

/// <summary>
/// The reference's <c>ChatGuard</c> (upstream <c>src/guard.ts</c>), kept internal: detection, the
/// default-deny policy and the session table in one object, as the ported upstream tests and the
/// parity tests drive it. The public API is <see cref="RampartAnalyzer"/>, <see cref="RampartAnonymizer"/>
/// and <see cref="RampartCloaker"/>. Not thread-safe.
/// </summary>
internal sealed class ChatGuard
{
    private readonly RampartAnalyzer _analyzer;

    /// <summary>The conversation's placeholder table.</summary>
    public SessionEntityTable Table { get; }

    public ChatGuard(RampartOptions? options = null) : this(new RampartAnalyzer(options), options) { }

    /// <summary>A guard over a shared analyzer (cheap: one per conversation).</summary>
    public ChatGuard(RampartAnalyzer analyzer, RampartOptions? options = null)
    {
        _analyzer = analyzer;
        Table = new SessionEntityTable(options?.Aliases, PiiLabels.ResolveKeepLabels(options?.KeepLabels));
    }

    /// <summary>
    /// Run this on the user's text before handing it to the model provider. Returns the placeholdered
    /// text to send plus the placeholders it contains.
    /// </summary>
    public ScrubResult Protect(string text) => Table.Scrub(text, _analyzer.Detect(text));

    /// <summary>Restore real values in a complete (non-streaming) assistant reply.</summary>
    public string Reveal(string reply) => Table.Rehydrate(reply);

    /// <summary>A stateful reveal for a reply that arrives in chunks.</summary>
    public StreamingReveal CreateStreamingReveal() => new(Table.Resolve);

    /// <summary>Reveal placeholders in a streamed reply, handling placeholders split across chunks.</summary>
    public IAsyncEnumerable<string> RevealAsync(IAsyncEnumerable<string> chunks, CancellationToken cancellationToken = default)
        => StreamingReveal.RevealAsync(chunks, Table.Resolve, cancellationToken);

    /// <summary>
    /// Defense in depth: scrub the model's <em>output</em> before logging or persisting it, since a
    /// model can emit PII the user never typed.
    /// </summary>
    public ScrubResult ProtectReply(string reply) => Table.Scrub(reply, _analyzer.Detect(reply));
}
