# Notebooks in Git

Working `.ipynb` files retain plots, tables and execution results. During
`git add`, the `nbstripout` filter passes a copy without outputs or execution
counts to Git. Files on disk are not overwritten. Diffs also use the stripped
representation so computed results do not obscure code changes.

## Setup after cloning

From the repository root:

```bash
uv sync
uv run nbstripout --install
uv run nbstripout --status
```

`nbstripout` is a development dependency. Rules for `.ipynb` files live in
`.gitattributes`; filter commands are configured locally in `.git/config`.
Git does not copy these commands when cloning, so install the filter in each
clone. The configured filter is required: failure stops notebook staging
instead of storing an unstripped copy.

After installation, use regular Git commands:

```bash
git add notebooks/
git diff --cached --stat
```

Inspect a particular staged notebook with `git show :notebooks/uzk/gru.ipynb`.
Its cells will have empty `outputs` and `null` `execution_count`; the local
notebook will retain its results.

Do not run `nbstripout file.ipynb` on working files: that command strips them
in place. Use the Git filter to preserve local plots. The filter applies to
newly staged content and does not remove outputs from earlier commits.
A fresh clone will not contain plots: they remain in the author's working copy.

See the [nbstripout documentation](https://github.com/kynan/nbstripout) for details.
