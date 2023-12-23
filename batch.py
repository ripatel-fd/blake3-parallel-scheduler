"""
batch.py provides an example scheduler for BLAKE3.
"""

from dataclasses import dataclass
import os
import struct
from typing import Optional

from pure_blake3 import ChunkState, IV, PARENT, ROOT, Hasher, compress, BLOCK_LEN

# Logging

VERBOSE = (os.getenv("VERBOSE") or "") not in {"", "0"}


def debug(str):
    if VERBOSE:
        print(str)


# Integer support routines


def msb(x):
    if x == 0:
        return 0
    i = -1
    while x > 0:
        x >>= 1
        i += 1
    return i


def pow2_up(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    x += 1
    return x


# Constants

COL_MAX_CNT = 8  # max number of elements per batch

CHUNK_LG_SZ = 10
CHUNK_SZ = 2**CHUNK_LG_SZ  # number of bytes per row

OUTCHAIN_LG_SZ = 5
OUTCHAIN_SZ = 2**5

INPUT_MAX_SZ = 10 * 1024 * 1024  # max supported input size

ROW_MAX_CNT = msb(INPUT_MAX_SZ // CHUNK_SZ) + 2


@dataclass
class ChunkOp:
    """ChunkOp is a scheduled BLAKE3 chunk hash operation"""

    msg: memoryview  # length in [0,CHUNK_SZ)
    out: memoryview  # 32 bytes
    counter: int
    flags: int

    def __str__(self):
        if (self.flags & ROOT) == ROOT:
            name = "root"
        elif self.flags & PARENT:
            name = "parent"
        elif self.flags == 0:
            name = "data"
        else:
            assert False, f"invalid flags: {hex(self.flags)}"
        return f"chunk(counter={self.counter} type={name} len(msg)={len(self.msg)}) => {self.out.hex()}"

    def process(self):
        if self.flags & PARENT:
            final = compress(
                IV, list(struct.unpack("<16I", self.msg)), 0, BLOCK_LEN, self.flags
            )
        else:
            state = ChunkState(IV, self.counter, 0)
            state.update(self.msg)
            last = state.output()
            final = compress(
                last.input_chaining_value,
                last.block_words,
                last.counter,
                last.block_len,
                last.flags | self.flags,
            )
        self.out[:] = struct.pack("<8I", *final[:8])
        return self.out


@dataclass
class HashState:
    """HashState generates an instruction stream of BLAKE3 chunk hash
    operations.  For each input size, there is a particular binary tree
    (see BLAKE3 spec Section 2.1).  This class iterates each tree in
    a particular way that maximizes parallelism."""

    # input is the actual input data to the hash operation
    input: memoryview

    # leaf_cnt is the number of leaf nodes in the input
    # The minimum leaf_cnt is one.  Each CHUNK_SZ bytes of input create
    # one leaf node.
    leaf_cnt: int

    # slots is the working memory buffering intermediate hash values
    # that are recursively merged; eventually arriving at the root
    # internally is a 2D array where columns are contiguous and rows
    # have a stride of (COL_MAX_CNT * OUT_CHAIN_SZ).  The columns in
    # each row are allocated left-to-right.
    slots: bytearray

    # rows holds the number of nodes buffered.  Once a row reaches
    # "COL_MAX_CNT", it needs to be flushed to the next layer (via pair-
    # wise joining).  progress holds the number of nodes already joined
    # for that layer.  Invariant: for each i in [0,ROW_MAX_CNT) :
    # progress[i] <= rows[i]
    # If progress[i] == rows[i] == COL_MAX_CNT, they both reset to 0.
    progress: list[int]
    rows: list[int]

    # head is the next leaf node to process
    head = 0

    # layer is the tree layer where we are currently adding nodes
    layer = 0

    # wait is the number of active hash operations
    wait = 0

    # live is the total number of unmerged nodes
    live = 0

    def __init__(self, input):
        self.input = input
        self.leaf_cnt = max(1, (len(self.input) + CHUNK_SZ - 1) >> CHUNK_LG_SZ)
        self.slots = memoryview(bytearray(ROW_MAX_CNT * COL_MAX_CNT * OUTCHAIN_SZ))
        self.rows = [0] * ROW_MAX_CNT
        self.progress = [0] * ROW_MAX_CNT

    # Some algorithm notes:
    #
    # Consider the following tree of three leaf nodes
    #
    #    2        [E]
    #            /   \
    #    1     [D]    |
    #         /   \   |
    #    0   [A] [B] [C]
    #
    # Leaf nodes are all in layer 0.  There are no gaps in layers.
    # Each adjacent 2-aligned pair is joined into the next layer.
    #
    # Layers with odd node count which form two special cases:
    # (a) the tree root (b) node joining across layers.
    # The latter occurs when there are two layers with uneven node
    # count (e.g. a tree with three leaves).  These are joined bottom-
    # to-top by moving the lower lonely node to the upper lonely node.
    #
    # I.e. the tree looks as follows after doing the uneven join.
    #
    #    2        [E]
    #            /   \
    #    1     [D]   [C]
    #         /   \
    #    0   [A] [B]

    # poll requests the next BLAKE3 hash instruction.  Returns ChunkOp
    # on success.  Returns None if the state is blocked on the next hash
    # operation (notified via submit).  Must not be called if self.done()
    def poll(self) -> Optional[ChunkOp]:
        if self.layer > len(self.rows):
            # Finished
            return None

        debug(
            f"      rows={self.rows} progress={self.progress} next_layer={self.layer} head={self.head}/{self.leaf_cnt} live={self.live}"
        )
        if self.layer == 0:
            # Leaf node
            if self.rows[0] >= COL_MAX_CNT:
                # Waiting for leaf hashes
                return None

            if self.head >= self.leaf_cnt:
                # Reached EOF
                return None

            # Schedule a new leaf node
            return self._prepare_leaf()

        else:
            # Merge complete batch
            return self._prepare_branch()

    def _seek_branch(self):
        # Happy case: Still hashing leaf nodes, thus attempt to build a
        # complete binary tree.
        if self.head < self.leaf_cnt:
            # Nothing to merge? Then go back to the leaf layer
            if self.progress[self.layer - 1] + 1 >= self.rows[self.layer - 1]:
                return False
            # Continue on current layer
            return True

        # Special case: We hashed all leaf nodes and need to finish the
        # tree.
        diff = [r - l for l, r in zip(self.progress, self.rows)]

        # Find an adjacent pair of nodes to hash
        for layer in range(0, len(diff)):
            nodes = diff[layer]
            if nodes >= 2:
                self.layer = layer + 1
                return True

        # Reached the following invariants:
        # - There are at least two unjoined nodes in the tree.
        # - Every unjoined node is in a different layer
        # (This happens for all nodes with leaf count not in 2^n.)
        for layer in diff:
            assert layer <= 1

        # We reached a state where there are no adjacent nodes.
        # Forcibly create a pair by moving a lonely node from the
        # lowest layer to the second-to-lowest.
        layer = 0
        while True:
            if diff[layer]:
                lo = layer
                break
            layer += 1
        while True:
            layer += 1
            if diff[layer]:
                hi = layer
                break
        self._slot(hi, self.rows[hi])[:] = self._slot(lo, self.progress[lo])
        self.rows[lo] -= 1
        self.rows[hi] += 1

        # Now merge the newly found pair
        self.layer = hi + 1
        return True

    def _prepare_branch(self):
        if self.live == 1:
            return None
        if not self._seek_branch():
            return None
        msg = self._slot_pair(self.layer - 1, self.progress[self.layer - 1])
        out = self._slot(self.layer, self.rows[self.layer])
        self.rows[self.layer] += 1
        self.progress[self.layer - 1] += 2
        self.wait += 1
        self.live -= 1
        flags = PARENT
        if self.live == 1:
            flags |= ROOT
        return ChunkOp(msg, out, 0, flags)

    def _prepare_leaf(self):
        offset = self.head << CHUNK_LG_SZ
        msg = memoryview(self.input)[offset:]
        msg = msg[: min(CHUNK_SZ, len(msg))]
        out = self._slot(0, self.head % COL_MAX_CNT)
        counter = self.head
        self.rows[0] += 1
        self.head += 1
        self.wait += 1
        self.live += 1
        flags = 0
        if self.leaf_cnt == 1:
            flags = ROOT
        return ChunkOp(msg, out, counter, flags)

    # submit notifies the scheduler that one or more operations completed.
    def submit(self, n):
        assert self.wait > 0

        self.wait -= n
        if self.wait > 0:
            return

        for i in range(ROW_MAX_CNT):
            if self.rows[i] == self.progress[i]:
                self.rows[i] = self.progress[i] = 0

        # All scheduled operations completed at this point.
        #
        # Decide whether we want to schedule another operation in this
        # layer, move up a layer, or go back to the leaf layer.

        # If the current layer is full, we should merge it into the
        # next layer
        if self.rows[self.layer] == COL_MAX_CNT:
            self.layer += 1
        # If we can merge more, do that
        elif (
            self.layer > 0 and self.progress[self.layer - 1] < self.rows[self.layer - 1]
        ):
            pass
        # If we reached there are no more leaf nodes, merge into the
        # next layer
        elif self.head == self.leaf_cnt:
            self.layer += 1
            if self.live == 1:
                self.layer = None
        # If we don't have anything to merge, go back to leaf layer
        elif self.layer > 0:
            self.layer = 0

    # Returns True if the root of the tree has been calculated.
    def done(self) -> bool:
        return self.layer is None

    # Returns a byte slice holding a single output chaining value.
    def _slot(self, layer, idx) -> memoryview:
        assert idx < COL_MAX_CNT, idx
        offset = ((layer * COL_MAX_CNT) + (idx % COL_MAX_CNT)) << OUTCHAIN_LG_SZ
        mem = self.slots[offset : offset + OUTCHAIN_SZ]
        debug(f"      slot({layer}, {idx}) => {mem.hex()} @ {hex(offset)}")
        return mem

    # Returns a byte slice holding an adjacent pair of output chaining
    # values.
    def _slot_pair(self, layer, idx) -> memoryview:
        assert idx < COL_MAX_CNT - 1, idx
        offset = ((layer * COL_MAX_CNT) + (idx % COL_MAX_CNT)) << OUTCHAIN_LG_SZ
        mem = self.slots[offset : offset + 2 * OUTCHAIN_SZ]
        debug(f"      slot({layer}, {idx}) => {mem.hex()} @ {hex(offset)}")
        return mem


def wide_hash(input) -> list[bytes]:
    state = HashState(input)
    print(f"Wide hashing {len(state.input)} bytes")
    attempt = 0
    while not state.done():
        debug(f"  Schedule {attempt}")
        op_cnt = 0
        while not state.done():
            op = state.poll()
            if op is None:
                break
            hash = op.process()
            debug(f"    {op}")
            op_cnt += 1
        state.submit(op_cnt)
        assert op_cnt > 0, "done"
        attempt += 1
    assert state.live == 1

    actual = hash

    ref = Hasher()
    ref.update(state.input)
    expected = ref.finalize()

    assert actual == expected, f"{actual.hex()} != {expected.hex()}"


def seq_hash(input: bytes) -> list[bytes]:
    state = HashState(input)
    print(f"Seq hashing {len(state.input)} bytes")
    attempt = 0
    while not state.done():
        op = state.poll()
        if op is None:
            break
        hash = op.process()
        debug(f"    {op}")
        state.submit(1)
        attempt += 1
    assert state.live == 1

    actual = hash

    ref = Hasher()
    ref.update(state.input)
    expected = ref.finalize()

    assert actual == expected, f"{actual.hex()} != {expected.hex()}"


if __name__ == "__main__":
    max_sz = 1024 * 12
    zeros = b"0" * max_sz
    for sz in range(0, max_sz, 1024):
        wide_hash(zeros[:sz])
        seq_hash(zeros[:sz])
    wide_hash(b"0" * (10 * 1024 * 1024))
    seq_hash(b"0" * (10 * 1024 * 1024))
