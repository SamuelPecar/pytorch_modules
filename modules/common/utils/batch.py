def sort_by_lengths(lengths):
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    _, reversed_idx = sorted_idx.sort(descending=True)

    def sort(iterable):
        return iterable[sorted_idx]

    def unsort(iterable):
        return iterable[reversed_idx][sorted_idx][reversed_idx]

    return sorted_lengths, sort, unsort
