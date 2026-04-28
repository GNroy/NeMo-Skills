# Agent Integration — Implementation Notes

Nuances and non-obvious failures encountered while building and running the
agentic pipeline on NeMo-Skills.

---

## 1. SSH host-key conflict blocks NeMo-Run rsync

**Symptom**
```
rsync error: unexplained error (code 255)
Stderr: Host key verification failed.
```

**Cause**
NeMo-Run's rsync call does not pass `-o StrictHostKeyChecking=no`.
If `~/.ssh/known_hosts` contains a stale entry for the cluster's *IP address*
that contradicts the entry for its *hostname*, SSH rejects the connection even
though the hostname entry is correct.

This happens when you previously connected with `StrictHostKeyChecking=no` and
it auto-added the IP, and then the cluster's IP key was rotated.

**Fix**
```bash
ssh-keygen -R <stale-ip>   # e.g. ssh-keygen -R 10.65.34.137
```
SSH will re-add the correct IP key on the next successful connection.

---

## 2. `nested_dataclass` does not walk the MRO for field annotations

**Symptom**
```
omegaconf.errors.ConfigKeyError: Key 'chat_template_kwargs' is not in struct
    full_key: inference.extra_body.chat_template_kwargs
```

**Cause**
`nested_dataclass` (in `nemo_skills/utils.py`) uses `check_class.__annotations__`
to decide which fields to recursively instantiate when `_init_nested=True`.
In Python, `__annotations__` is per-class — it does not include fields inherited
from parent classes.

Consequence for `AgentTaskConfig(GenerationTaskConfig)`: the inherited
`inference: InferenceConfig` field was invisible to the recursion guard, so it
stayed as an OmegaConf `DictConfig` instead of being converted to a Python
`InferenceConfig` dataclass. Later, when `GenerationTask.__init__` replaced
`self.cfg.inference.extra_body` with a plain Python dict, the assignment went
through OmegaConf's `__setattr__`, which re-wrapped the empty dict as a
*struct* `DictConfig` (zero allowed keys). Writing a new key to it then raised
`ConfigKeyError`.

The same bug affects any `nested_dataclass` subclass that has dataclass-typed
fields defined only in a parent class.

**Fix**
Walk the full MRO when building the annotation map:
```python
all_annotations = {}
for cls in reversed(inspect.getmro(check_class)):
    if hasattr(cls, '__annotations__'):
        all_annotations.update(cls.__annotations__)
```

---

## 3. `prompt_format='openai'` without `prompt_config` fails on standard benchmarks

**Symptom**
```
KeyError: 'messages'
```
in `generate.py :: fill_prompt`.

**Cause**
When `prompt_format='openai'` and `prompt_config=None`, `fill_prompt` takes
the "pure openai path" which expects the data point to already contain a
`messages` list. Standard benchmarks (physics, frontierscience, etc.) provide
a `problem` string field, not a pre-built messages list.

`AgentTaskConfig` overrides `prompt_format` to `"openai"` but inherits
`prompt_config=None`, so the pure-openai path was taken every time.

**Fix**
`AgentTaskConfig` now sets `prompt_config: str = "agents/default_agent"` as
its default. The default template has `user: '{problem}'`, which `fill_prompt`
uses to build the messages list from the data point's `problem` field.

---

## 4. Curly-brace placeholders in field comments break `get_help_message`

**Symptom**
```
KeyError: 'problem'
```
raised at module import time (before Hydra even starts).

**Cause**
`get_help_message` reads the dataclass source with `inspect.getsource` and
calls `str.format(**kwargs)` on the collected field comments. The regex that
pre-escapes braces only handles *multi-word* patterns like `{word space word}`
— it does not escape single-word patterns like `{problem}`.

Writing `{problem}` in a comment attached to a field in any `nested_dataclass`
will therefore be mistaken for a missing format argument and raise `KeyError`.

**Rule**: never write `{word}` in comments inside `nested_dataclass` classes.
Use prose like `the 'problem' key` instead.

---

## 5. Pushing to `origin` (GNroy/NeMo-Skills)

The `origin` remote uses SSH (`git@github.com:GNroy/NeMo-Skills.git`) with the
key at `~/.ssh/gitlab` (mapped via `~/.ssh/config`).  There is no persistent
SSH agent in this environment, so the key must be added each session.

```bash
eval $(ssh-agent -s)
echo "<passphrase>" | SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/bin/cat ssh-add ~/.ssh/gitlab
git push origin <branch>
```

Notes:
- `ssh -T git@github.com` exits with code 1 even on success ("no shell access"),
  so do not use it as a gate in an `&&` chain before `git push`.
- GPG commit signing is not available in this environment; use
  `git -c commit.gpgsign=false commit ...` when creating commits.

---

## 6. General: `rerun_done=False` means output files accumulate across resubmits

Each time the script is resubmitted without changing `--output-dir`, jobs whose
chunk output files already exist are skipped (`skip_filled=True`). This is
intentional but means partial results from a broken run are silently reused on
the next attempt. If the failure corrupted some chunk files, pass `--rerun-done`
or wipe the output directory before resubmitting.
