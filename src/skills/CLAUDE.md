# src/skills/ - Procedural Skill Memory (SKILL substrate)

## Purpose
Replaces the flat "Divine Codex" (stored VALUES) + the superficial regex "playbook" with a
structured, PROCEDURAL, PINNED, PERSISTENT memory that is injected into the **Coder** (not just
the Manager).

A skill is a PROCEDURE, not a value: *"read max_episode_steps from env.spec; set learning_starts =
that + margin"* generalizes to every goal env; *"learning_starts=200"* poisons the next one.

## skill_store.py

### `Skill` (dataclass)
`{id, name, when_to_use, procedure, pitfalls, verification, status, confidence, source_env,
created_iter, last_validated_iter, tags[]}`
- `status`: `proposed | verified | deprecated`
- `embeddable_text()` (for future semantic search) · `render()` (compact prompt block) · `to_markdown()`

### `SkillStore`
- `load()` / `save()` — canonical `skills.json` + per-skill `<id>.SKILL.md` export (browsable, git-friendly, presentable).
- `seed_if_empty(initial_skills, initial_rules)` — seed a fresh store from config (prefers structured skills, falls back to wrapping flat rules).
- `add / improve / verify / deprecate` — mutation. **Pinning**: a `verified` skill cannot be clobbered by a weaker (non-verified) write; `deprecate` only demotes confidence on a verified skill.
- `relevant(env_name, tags, query)` — retrieval. Now: env/tag match + general skills, verified-first. Built for future embedding/semantic search (the `query` arg + `embeddable_text()`), no caller change needed.
- `render_for_coder(env_name, tags)` — the SKILL block injected into the Coder prompt.
- `apply_ops(ops, iteration)` — apply SHODAN's `skill_ops` `{add, improve, verify, remove}`; saves on change.

## Wiring
- Instantiated in `main.py` (loaded from `config.skills_dir`, seeded from `config.initial_skills` / `initial_codex_rules`), placed in `AgentState["skill_store"]` (a live instance, mutated in place like `stats`/`conversation_logger`).
- **Coder** (`coder.py`) renders relevant skills (+ any legacy Codex notes) into its prompt.
- **Reviewer/SHODAN** (`reviewer.py`) applies `skill_ops` and sees the skill summary (`render_summary()`).
- **Manager** (`manager.py`) distils a verified procedural skill from winning code on env-solve (`_skill_from_winning_code`).

## Config
- `skills_dir` (default `"skills"`) — where the store persists. Point different runs at different dirs (e.g. `skills/robot_seeded` vs `skills/robot_blind`) to keep an A/B comparison clean.
- `initial_skills` — list of structured procedural skills to seed (each: name/when_to_use/procedure/pitfalls/verification/tags/status; status defaults to verified for a deliberate seed).
