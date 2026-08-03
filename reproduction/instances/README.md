# Bundled experiment instances

These immutable inputs make the fresh reviewer commands independent of
machine-specific MIPLIB and MATPOWER paths. Section 3.1 defaults to the bundled
MPS file, and Section 3.3 defaults to the bundled MATPOWER directory. The
`--mps` and `--matpower-dir` options remain available as explicit overrides.

Verify every input from the repository root with:

```bash
sha256sum --check reproduction/instances/SHA256SUMS
```

## MIPLIB input

`miplib/enlight_hard.mps.gz` is the byte-exact official compressed download
used for the experiment. Its compressed SHA256 is
`942168c2126a2a91ae3ec1ededea59bc1af0cad55f94223edf4c03d20e831f66`;
the decompressed MPS content SHA256 is
`572ca23c17d0ad734895e8338af458525a753ee76bdb117d2917e4069c6b65b0`.
The public source is the
[MIPLIB instance page](https://miplib.zib.de/instance_details_enlight_hard.html)
and its linked download. The file is preserved unchanged, including its lack of
an embedded license notice.

## MATPOWER inputs

The six `matpower/*.m` files are preserved byte-for-byte from MATPOWER 8.1,
tag `8.1`, commit `1a828c7af590714499284e36ee9c81273388c594`. Their exact
hashes are in `SHA256SUMS`; the reproduction preflight rejects changed files
before starting Julia.

The author-side internal PDMO working tree did not contain the complete input
set: only `case30.m` and `case118.m` existed in its older Git history, and
those blobs match the bundled MATPOWER 8.1 files. The complete six-file set was
copied from the retained MATPOWER 8.1 checkout adjacent to that internal
repository. The MPS was copied from a retained official MIPLIB download because
no MPS blob existed in the internal PDMO checkout.

MATPOWER's package license explicitly says its case files are not covered by
the package's BSD license. The original per-file comments and attribution
notices are therefore retained as the authoritative notices. In particular,
`case89pegase.m` and `case1888rte.m` carry Creative Commons Attribution 4.0
notices and requested citations. `matpower/MATPOWER-PACKAGE-LICENSE` and
`matpower/MATPOWER-CITATION` preserve the surrounding MATPOWER package
license statement and requested software citations; they do not replace the
case-specific notices.
