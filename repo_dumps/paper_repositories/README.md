# Paper Repository Dumps

This directory is the default output location for `scripts/dump_paper_repositories.py`.

Run:

```bash
python scripts/dump_paper_repositories.py
```

The script creates dated snapshots under `repo_dumps/paper_repositories/YYYY-MM-DD/`.
Each snapshot contains Git mirrors, portable `*.bundle` files, and `manifest.json` /
`manifest.csv` with source URLs, experiment membership, default branches, and HEAD SHAs.

Generated mirrors and bundles are intentionally ignored by Git because they can be large.
