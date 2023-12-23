This repository is a fork of the [BLAKE3 Pure Python implementation](https://github.com/oconnor663/pure_python_blake3).

It adds [batch.py](./batch.py), an alternative non-recursive scheduler
that allows for data parallelism across independent hash operations.
