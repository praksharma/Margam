# Burgers observations

A deterministic subset of the original authors' published reference field:
2,000 unique `(x,t,u)` rows, stored as ordinary CSV. `x` is space and `t` is physical
time. No parameter answers are encoded in the CSV or supplied as loss targets.

See source.json for the pinned commit, original MAT path/hash, selection convention
and CSV hash. The upstream MIT notice is retained in UPSTREAM-LICENSE.txt. The
original MAT file is not bundled; runtime training reads only this CSV using NumPy.
This is an implementation fixture, not a new accuracy benchmark.
