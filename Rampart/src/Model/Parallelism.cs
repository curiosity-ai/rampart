using System.Runtime.ExceptionServices;

namespace Rampart.Model;

/// <summary>A unit of parallel work over the index range <c>[from, to)</c>.</summary>
internal interface IParallelJob
{
    void Run(int from, int to);
}

/// <summary>
/// Fork/join for the encoder's kernels over a dedicated pool of <c>cores - 1</c> worker threads.
/// <para>
/// A forward pass has ~40 parallel regions, many only tens of microseconds long, so dispatch latency
/// decides whether parallelism pays at all: waking parked thread-pool threads costs more than a short
/// region. Workers here spin for a short while after each region before they park (as ONNX Runtime's
/// pool does), so back-to-back regions find them awake; the calling thread always works too.
/// </para>
/// <para>
/// Nothing is allocated per call: each job type has one reusable batch. One region runs on the pool
/// at a time; a call that finds the pool busy (another inference on another thread) runs inline on
/// its own thread instead of queueing — on a busy server the cores are already taken anyway.
/// </para>
/// </summary>
internal static class Parallelism
{
    /// <summary>Iterations a worker spins for new work before parking (a few hundred µs: longer than the gap between regions).</summary>
    private const int SpinIterations = 500;

    private static readonly object Gate = new();
    private static readonly Worker[] Workers = CreateWorkers();
    private static volatile Batch? _current;
    private static long _generation;

    private static Worker[] CreateWorkers()
    {
        var workers = new Worker[Math.Max(0, Environment.ProcessorCount - 1)];
        for (int i = 0; i < workers.Length; i++) workers[i] = new Worker(i);
        return workers;
    }

    public static void For<TJob>(int count, int degree, ref TJob job) where TJob : struct, IParallelJob
    {
        int helpers = Math.Min(Math.Min(degree, count) - 1, Workers.Length);
        if (helpers <= 0 || !Monitor.TryEnter(Gate))
        {
            job.Run(0, count);
            return;
        }

        try
        {
            var batch = Batch<TJob>.Instance;
            batch.Start(job, count, helpers);
            _current = batch;
            Interlocked.Increment(ref _generation);
            for (int i = 0; i < helpers; i++) Workers[i].Wake();
            batch.Work();
            batch.Close();
            batch.Clear();
        }
        finally
        {
            Monitor.Exit(Gate);
        }
    }

    /// <summary>The type-erased face of a batch, as the workers see it.</summary>
    private abstract class Batch
    {
        private int _next;
        private int _count;
        private int _helpers;
        private int _joined;
        private int _active;
        private int _closed = 1;
        private Exception? _error;

        protected void Reset(int count, int helpers)
        {
            _count = count;
            _helpers = helpers;
            _next = -1;
            _joined = 0;
            _error = null;
            Volatile.Write(ref _closed, 0); // last: a helper that sees the batch open sees it complete
        }

        /// <summary>A helper joins if the batch is still open and has room for it.</summary>
        public void Help()
        {
            Interlocked.Increment(ref _active);
            try
            {
                if (Volatile.Read(ref _closed) == 0 && Interlocked.Increment(ref _joined) <= _helpers) Work();
            }
            finally
            {
                Interlocked.Decrement(ref _active);
            }
        }

        public void Work()
        {
            try
            {
                int i;
                while ((i = Interlocked.Increment(ref _next)) < _count) RunItem(i);
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref _error, e, null);
                Volatile.Write(ref _next, _count); // stop handing out work
            }
        }

        /// <summary>The owner, out of work: close the batch and wait for helpers still inside it.</summary>
        public void Close()
        {
            Volatile.Write(ref _closed, 1);
            var spin = default(SpinWait);
            while (Volatile.Read(ref _active) != 0) spin.SpinOnce(sleep1Threshold: -1);
            if (_error is { } e) ExceptionDispatchInfo.Throw(e);
        }

        protected abstract void RunItem(int index);
    }

    private sealed class Batch<TJob> : Batch where TJob : struct, IParallelJob
    {
        public static readonly Batch<TJob> Instance = new();

        private TJob _job;

        public void Start(TJob job, int count, int helpers)
        {
            _job = job;
            Reset(count, helpers);
        }

        /// <summary>Drop the job (it holds pointers into the caller's pinned buffers).</summary>
        public void Clear() => _job = default;

        protected override void RunItem(int index) => _job.Run(index, index + 1);
    }

    private sealed class Worker
    {
        private readonly SemaphoreSlim _signal = new(0);
        private int _sleeping;

        public Worker(int index)
        {
            var thread = new Thread(Loop) { IsBackground = true, Name = $"Rampart worker {index}" };
            thread.Start();
        }

        public void Wake()
        {
            if (Interlocked.Exchange(ref _sleeping, 0) == 1) _signal.Release();
        }

        private void Loop()
        {
            long seen = Interlocked.Read(ref _generation);
            while (true)
            {
                // Spin briefly for the next region, then park until woken.
                int spins = 0;
                while (Interlocked.Read(ref _generation) == seen && spins++ < SpinIterations) Thread.SpinWait(20);
                if (Interlocked.Read(ref _generation) == seen)
                {
                    Volatile.Write(ref _sleeping, 1);
                    // Re-check after announcing sleep, so a dispatch between the check and the write is not missed.
                    if (Interlocked.Read(ref _generation) == seen) _signal.Wait();
                    else if (Interlocked.Exchange(ref _sleeping, 0) == 0) _signal.Wait(); // a Wake raced us: consume its release
                }
                seen = Interlocked.Read(ref _generation);
                _current?.Help();
            }
        }
    }
}
