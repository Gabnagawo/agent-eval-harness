# Raw results

Output of the real A vs. B run reported in the main README: `qwen3:8b` through Ollama, 32 tasks, k = 3, at the commit in `commit.txt`.

| File | Content |
|---|---|
| `baseline.json`, `guarded.json` | Full JSON reports, with every trajectory |
| `baseline.log`, `guarded.log` | Console output of each run |
| `compare.json`, `compare.md` | A vs. B comparison, per metric, family and task |
| `compare.html` | Self-contained HTML report of both runs, side by side |
| `commit.txt` | Commit the runs were made at |

`guarded.log` is the output of the resumed session: lines 1 to 21 replay the trials restored from the checkpoint of the interrupted session, and the duration in its header covers trials 22 to 96 only.
