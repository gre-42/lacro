# -*- coding: utf-8 -*-
import itertools
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Sequence, Set

from lacro.assertions import set_assert_subset
from lacro.collections._value import IdAndValue
from lacro.string.misc import trunc
from lacro.typing import SizedIterable

# ---------------
# - get element -
# ---------------


def first_element(lst):
    try:
        return next(iter(lst))
    except StopIteration:
        raise ValueError('Iterator did not contain any element')


def first_element_finish(lst):
    """
    returns the first element and iterates until the end
    """
    try:
        v = next(lst)
    except StopIteration:
        raise ValueError('Iterator did not contain any element')
    for _ in lst:
        pass
    return v


def last_element(lst):
    """
    returns the last element and iterates until the end
    """
    for v in lst:
        pass
    try:
        # noinspection PyUnboundLocalVariable
        return v  # noqa pylint: disable=W0631
    except UnboundLocalError:
        raise ValueError('Iterator did not contain any element')


def single_element(lst, error=None, to_str=lambda v: trunc(str(v), 2000),
                   msg=''):
    if len(lst) != 1:
        if error is not None:
            raise ValueError(error(lst) if callable(error) else error)
        else:
            raise ValueError('%sList does not contain exactly one element: '
                             '%s\n' % (msg, to_str(lst)))
    return first_element(lst)


def unique_value(lst, error=None, to_str=lambda v: trunc(str(v), 2000),
                 msg=''):
    return single_element(set(lst), error=error, to_str=to_str, msg=msg)


# --------
# - misc -
# --------

def is_iterable(v) -> bool:
    try:
        iter(v)
        return True
    except TypeError:
        return False


# -------
# - zip -
# -------


def anyziplist(anyzip, lst, sublen=None, types=None, msg=''):
    assert (sublen is not None) + (types is not None) == 1
    assert str not in (type(l) for l in lst)
    if sublen is not None:
        types = [tuple] * sublen
    else:
        sublen = len(types)
    if len(lst) > 0:
        res = [typ(v) for typ, v in eqzip(types, list(anyzip(*lst)),
                                          msg=msg + ('Types(sublen=%s): ' %
                                                     sublen))]
        assert len(res) == sublen
        return res
    else:
        return [typ() for typ in types]


def ziplist(lst, sublen=None, types=None, msg=''):
    return anyziplist(zip, lst, sublen, types, msg=msg)


def eqziplist(lst, sublen=None, types=None, msg=''):
    return anyziplist(lambda *args: eqzip(*args, msg=msg), lst, sublen, types,
                      msg=msg)


def eqzip(*args: SizedIterable, msg=''):
    if len(set(len(a) for a in args)) > 1:
        raise ValueError('%sNot all lists have equal length %s' %
                         (msg, tuple(len(a) for a in args)))
    return zip(*args)


# -------------
# - iterables -
# -------------


def iterables_concat(iterables):
    return (v for l in iterables for v in l)


def iterable_ids_of_unique(ite):
    return (iv.i for iv in set(IdAndValue(i, v) for i, v in enumerate(ite)))


def iterable_ids_of_change(ite):
    old = None

    def changed(v, i):
        nonlocal old
        res = (i == 0 or v != old)
        old = v
        return res

    return (i for i, v in enumerate(ite) if changed(v, i))


def iterable_ordered_group(ite, available_groups=None, reverse=False,
                           sort_keys=True):
    groups = itertools.groupby(sorted((IdAndValue(i, v)
                                       for i, v in enumerate(ite)),
                                      reverse=reverse))
    # print([[k,list(vs)] for k,vs in groups])
    # may not yet sort here because advancing the group iterator destroys
    # previous groups
    res = [(giv, [iv.i for iv in ivs]) for giv, ivs in groups]
    if available_groups is None:
        return [(g_v[0].v, g_v[1])
                for g_v in (res if sort_keys else
                            sorted(res, key=lambda g_v: g_v[0].i))]
    else:
        dres = OrderedDict((g_v[0].v, g_v[1]) for g_v in res)
        set_assert_subset(available_groups, dres.keys())
        return [(v, dres.get(v, [])) for v in available_groups]


def iterables_merge(iterables):
    from lacro.collections import list_removed_duplicates
    iterables = [list_removed_duplicates(it) for it in iterables]
    v2id = OrderedDict()
    for it in iterables:
        for i, c in enumerate(it):
            v2id[c] = max(v2id.get(c, 0), i)
    return [c for i, c in sorted((i, c) for c, i in v2id.items())]


def prod(iterable):
    """
    Like the builtin "sum" function, but for multiplication.
    has fewer restrictions on the shape of the arguments than numpy.multiply.
    """
    from functools import reduce
    from operator import mul

    return reduce(mul, iterable, 1)


def products(iterable, start, repeats, cat=lambda a, b: a + b,
             commutative=False):
    """
    Like itertools.product, but with all intermediate repetitions smaller
    or equal repeats.
    """
    last = [start]
    yield from last
    for i in range(repeats):
        last = [cat(v, l) for j, v in enumerate(iterable)
                for l in last[(j * i if commutative else None):]]
        yield from last


def ids_of_longest_univariate_products(nitems, repeats, commutative=False):
    """
    Indices of all univariate products of maximum length.
    """
    res = products([[i] for i in range(nitems)], [], repeats,
                   commutative=commutative)
    return [i for i, l in enumerate(res)
            if (len(set(l)) == 1) and (len(l) == repeats)]


def blocked_iterator(iterable, block_size, totuple=False):
    """
    http://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    """
    it = iter(iterable)
    while True:
        if totuple:
            chunk = tuple(itertools.islice(it, block_size))
            if len(chunk) == 0:
                return
            yield chunk
        else:
            chunk_it = itertools.islice(it, block_size)
            try:
                first_el = next(chunk_it)
            except StopIteration:
                return
            yield itertools.chain((first_el,), chunk_it)


class RewindingGenerator:

    def __init__(self, generator_function, len_seq):
        self.generator_function = generator_function
        self.len_seq = len_seq

    def __iter__(self):
        return self.generator_function()

    def __len__(self):
        return self.len_seq


def ProgressBarIterator(seq, len_seq=None, print_at_verbosity=1,
                        remove_old_speeders=True):
    len_seq = len(seq) if len_seq is None else len_seq
    return RewindingGenerator(
        lambda: ProgressBarIterator_(seq, len_seq, print_at_verbosity,
                                     remove_old_speeders), len_seq)


def ProgressBarIterator_(seq, len_seq, print_at_verbosity=1,
                         remove_old_speeders=True):
    import lacro.stdext as se
    p = ProgressBar(len_seq)
    # this assumes that time passes after the "yield s", and not in the "seq"
    # iterator
    for s in seq:
        p.notify_work_item_starting()
        if remove_old_speeders:
            p.remove_old_speeders()
        if se.verbosity.value >= print_at_verbosity:
            print(p.progress_info_string(), flush=True)
        yield s


class ProgressBar:
    """
    usage
    p = ProgressBar(work_count)
    for ....
        do_work
        p.work_item_completed()
    """

    def __init__(self, work_count: int,
                 history_length: Optional[int] = None) -> None:
        self.work_count = work_count
        self.work_started = 0
        self.work_completed = 0
        self.history_length = history_length
        self.local = threading.local()
        self.workers: Set[int] = set()
        self.data = None

    def notify_work_item_starting(self):
        self.work_started += 1
        if not hasattr(self.local, 'times'):
            self.local.times = []
        else:
            self.work_completed += 1
        self.local.times.append(time.time())
        self.local.times = self.local.times[-10:]
        self.workers |= {threading.current_thread().ident}

    # can be used to remove extremely fast previous runs that resulted from
    # loop skips the time estimates directly after a speeder are bogus,
    # because the last measured time delta is always used as a reference,
    # not the maximum measured time delta
    def remove_old_speeders(self):
        # note that '1234'[-1] = '4'
        # note that '1234'[:-1] = '123'
        if len(self.local.times) >= 2:
            T = (self.local.times[-1] - self.local.times[-2])
            self.local.times = [t for t, tn in zip(self.local.times[0:-2],
                                                   self.local.times[1:-1])
                                if tn - t > 0.1 * T] + self.local.times[-2:]
            self.local.times = self.local.times[(-self.history_length  # noqa pylint: disable=E1130
                                                 if (self.history_length
                                                     is not None) else None):]

    def remaining_time(self):
        remaining_work = self.work_count - self.work_completed
        time_per_work = ((self.local.times[-1] - self.local.times[0]) /
                         (len(self.local.times) - 1))
        return remaining_work * time_per_work / len(self.workers)

    def progress_info_string(self):
        if not hasattr(self.local, 'times'):
            raise ValueError('No iteration before "progress_info_string"')
        assert len(self.local.times) > 0

        from lacro.string.numeric import i_maxnum_2_str
        res = '%3d %% | Beginning %s of %d' % (
            (100 * (self.work_started - 1)) // self.work_count,
            i_maxnum_2_str(self.work_started, self.work_count),
            self.work_count)
        if len(self.local.times) == 1:
            res += ' | Now: %s' % datetime.now()
        else:
            rem_td = timedelta(seconds=self.remaining_time())
            res += ' | Remaining: %15s | End: %s' % (
                rem_td,
                datetime.now() + rem_td)
        return res


def blocked_indices_seq(nindices: Sequence[int],
                        block_size: Optional[Sequence[int]] = None,
                        nblocks: Optional[Sequence[int]] = None,
                        interleaved=False,
                        slices=False,
                        overlap=None,
                        return_valid=False):
    """
    see also: numpy.array_split
    """
    assert (block_size is None) + (nblocks is None) == 1

    if block_size is not None:
        assert len(nindices) == len(block_size)
    if nblocks is not None:
        assert len(nindices) == len(nblocks)
    if overlap is not None:
        assert len(nindices) == len(overlap)
    res = itertools.product(*[blocked_indices_num(
        nindices[i],
        None if block_size is None else block_size[i],
        None if nblocks is None else nblocks[i],
        interleaved,
        slices,
        None if overlap is None else overlap[i],
        return_valid
    ) for i in range(len(nindices))])
    if return_valid:
        return [zip(*r) for r in res]
    else:
        return res


def blocked_indices_num(nindices: int,
                        block_size: Optional[int] = None,
                        nblocks: Optional[int] = None,
                        interleaved=False,
                        slices=False,
                        overlap=None,
                        return_valid=False):
    if block_size is None:
        block_size = (nindices - 1) // nblocks + 1
    else:
        nblocks = (nindices - 1) // block_size + 1

    if overlap is None:
        overlap = 0

    def sl0(b, e, s=None):
        return (slice(b, e, s) if slices else
                list(range(b, e, 1 if s is None else s)))

    def sl1(b, e):
        b1 = max(0, b - overlap)
        e1 = min(e + overlap, nindices)
        ids = sl0(b1, e1)
        if return_valid:
            # print(overlap, b, b1, max(0,b-b1))
            vb1 = max(0, b - b1)
            return ids, sl0(b, e), sl0(vb1, vb1 + e - b)
        else:
            return ids

    if interleaved:
        assert overlap == 0
        # zz = list(zip(list(range(nblocks)) * block_size,
        #               list(range(nindices))))
        # print(zz)
        # return [[vv for kk,vv in v] for k,v in
        # itertools.groupby(sorted(zz, key=lambda bi: bi[0]), key=lambda
        # bi:bi[0])]
        return [sl0(i, nindices, nblocks) for i in range(nblocks)]
    else:
        return [sl1(i, min(i + block_size, nindices))
                for i in range(0, nindices, block_size)]
