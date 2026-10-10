namespace Rampart.Model;

/// <summary>A unit of parallel work over the index range <c>[from, to)</c>.</summary>
internal interface IParallelJob
{
    void Run(int from, int to);
}

/// <summary>
/// A minimal fork/join over the thread pool, for the encoder's kernels. Compared with
/// <c>Parallel.For</c> it allocates one small object per call (no delegates, closures or partitioner
/// state), the calling thread does work instead of only waiting, and it never waits for a helper that
/// has not started: helpers queued behind other work find the batch closed and return.
/// </summary>
internal static class Parallelism
{
    public static void For<TJob>(int count, int degree, ref TJob job) where TJob : struct, IParallelJob
    {
        int workers = Math.Min(degree, count);
        if (workers <= 1)
        {
            job.Run(0, count);
            return;
        }

        var batch = new Batch<TJob>(job, count);
        for (int i = 1; i < workers; i++) ThreadPool.UnsafeQueueUserWorkItem(batch, preferLocal: false);
        batch.Work();
        batch.Close();
    }

    private sealed class Batch<TJob>(TJob job, int count) : IThreadPoolWorkItem where TJob : struct, IParallelJob
    {
        private TJob _job = job;
        private int _next = -1;
        private int _active;
        private int _closed;
        private Exception? _error;

        /// <summary>Helper entry point: join unless the caller has already finished.</summary>
        public void Execute()
        {
            Interlocked.Increment(ref _active);
            try
            {
                if (Volatile.Read(ref _closed) == 0) Work();
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
                while ((i = Interlocked.Increment(ref _next)) < count) _job.Run(i, i + 1);
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref _error, e, null);
                Volatile.Write(ref _next, count); // stop handing out work
            }
        }

        /// <summary>Called by the owner once it runs out of work: waits for helpers still running.</summary>
        public void Close()
        {
            Volatile.Write(ref _closed, 1);
            var spin = default(SpinWait);
            while (Volatile.Read(ref _active) != 0) spin.SpinOnce();
            if (_error is { } e) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Throw(e);
        }
    }
}
