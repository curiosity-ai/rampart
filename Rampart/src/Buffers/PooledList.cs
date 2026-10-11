using System.Buffers;
using System.Runtime.CompilerServices;

namespace Catalyst.Rampart.Buffers;

/// <summary>
/// A growable list over <see cref="ArrayPool{T}.Shared"/>: the scratch storage of the hot paths, so a
/// call allocates nothing once the pool is warm. A mutable struct — pass it by <c>ref</c>, never copy
/// it, and <see cref="Dispose"/> it exactly once.
/// </summary>
internal struct PooledList<T> : IDisposable
{
    private T[] _items;
    private int _count;

    public PooledList(int capacity)
    {
        _items = ArrayPool<T>.Shared.Rent(Math.Max(capacity, 16));
        _count = 0;
    }

    public readonly int Count => _count;

    public readonly Span<T> AsSpan() => _items.AsSpan(0, _count);

    public readonly ref T this[int index] => ref _items[index];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(T item)
    {
        if (_count == _items.Length) Grow(_count + 1);
        _items[_count++] = item;
    }

    public void AddRange(ReadOnlySpan<T> items)
    {
        if (_count + items.Length > _items.Length) Grow(_count + items.Length);
        items.CopyTo(_items.AsSpan(_count));
        _count += items.Length;
    }

    public void Clear() => _count = 0;

    public void RemoveAt(int index)
    {
        _items.AsSpan(index + 1, _count - index - 1).CopyTo(_items.AsSpan(index));
        _count--;
    }

    private void Grow(int needed)
    {
        var next = ArrayPool<T>.Shared.Rent(Math.Max(needed, _items.Length * 2));
        _items.AsSpan(0, _count).CopyTo(next);
        ArrayPool<T>.Shared.Return(_items, RuntimeHelpers.IsReferenceOrContainsReferences<T>());
        _items = next;
    }

    public void Dispose()
    {
        if (_items is null) return;
        ArrayPool<T>.Shared.Return(_items, RuntimeHelpers.IsReferenceOrContainsReferences<T>());
        _items = null!;
        _count = 0;
    }
}
