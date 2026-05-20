# validation/data

* `smoke_5.ns.jsonl` — 5 hand-written NS-format problems for wiring smoke.
  Cheap-to-judge questions (single-word answers); the goal is to exercise
  the 3-run pipeline end-to-end, not to measure agent quality.
* `full100` (not committed) — produced by `ns prepare_data`:

  ```bash
  ns prepare_data --benchmarks frontierscience-olympiad --data_dir /alaptev/data
  # → /alaptev/data/frontierscience-olympiad/all.jsonl  (100 problems)
  ```

  We don't commit the full dataset because it's already hosted on
  Hugging Face (`openai/frontierscience`, config `olympiad`, split
  `test`) and re-downloaded with `ns prepare_data`.
