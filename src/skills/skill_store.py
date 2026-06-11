"""Procedural skill memory — the SKILL substrate that replaces the flat Divine Codex
and the superficial regex "playbook".

Why this exists (from the PandaPush post-mortem):
  - The old Codex stored VALUES ("learning_starts=200") — fragile, env-specific,
    re-fumbled, and silently overwritten by later wrong guesses.
  - The old playbook captured only `algo/steps/device` via regex and was injected to
    the Manager only — so a solved env's real recipe (SAC + MultiInputPolicy + HER +
    learning_starts-from-spec + chunked-resume) never reached the next env's Coder.

A SKILL is a PROCEDURE, not a value: "read max_episode_steps from env.spec; set
learning_starts = that + margin; print both" generalises to every goal env. Skills are:
  - STRUCTURED      (when_to_use / procedure / pitfalls / verification)
  - PINNED          (a `verified` skill cannot be clobbered by a weaker `proposed` one)
  - PERSISTENT      (skills.json on disk -> learning accumulates across runs)
  - CROSS-TASK      (injected to the CODER, carried between environments)
  - SEMANTIC-READY  (embeddable_text() + relevant(query) for future embedding retrieval)

The canonical store is skills.json; a human-readable `<name>.SKILL.md` is exported per
skill (browsable library, git-friendly, presentable).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional

VALID_STATUS = ("proposed", "verified", "deprecated")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "skill").strip().lower()).strip("-")
    return (s or "skill")[:48]


@dataclass
class Skill:
    id: str
    name: str
    when_to_use: str = ""
    procedure: str = ""
    pitfalls: str = ""
    verification: str = ""
    status: str = "proposed"            # proposed | verified | deprecated
    confidence: float = 0.5
    source_env: str = ""
    created_iter: int = 0
    last_validated_iter: int = -1
    tags: List[str] = field(default_factory=list)

    def embeddable_text(self) -> str:
        """Text blob for future semantic (embedding) retrieval."""
        return f"{self.name}. When: {self.when_to_use}. Procedure: {self.procedure}. Tags: {', '.join(self.tags)}"

    def render(self) -> str:
        """Compact SKILL block for prompt injection (the Coder follows this)."""
        lines = [f"### [{self.id}] {self.name}  ({self.status})"]
        if self.when_to_use:
            lines.append(f"- When to use: {self.when_to_use}")
        if self.procedure:
            lines.append(f"- Procedure: {self.procedure}")
        if self.pitfalls:
            lines.append(f"- Pitfalls: {self.pitfalls}")
        if self.verification:
            lines.append(f"- Verify: {self.verification}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Full human-readable SKILL.md document."""
        return (
            f"# {self.name}\n\n"
            f"- **id**: `{self.id}`\n"
            f"- **status**: {self.status} (confidence {self.confidence:.2f})\n"
            f"- **source env**: {self.source_env or '-'}\n"
            f"- **tags**: {', '.join(self.tags) or '-'}\n\n"
            f"## When to use\n{self.when_to_use or '-'}\n\n"
            f"## Procedure\n{self.procedure or '-'}\n\n"
            f"## Pitfalls\n{self.pitfalls or '-'}\n\n"
            f"## Verification\n{self.verification or '-'}\n"
        )


class SkillStore:
    """Loads/saves procedural skills and renders the relevant ones for the Coder."""

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.json_path = os.path.join(skills_dir, "skills.json")
        self.skills: Dict[str, Skill] = {}

    # ---------- persistence ----------
    def load(self) -> "SkillStore":
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                valid = {f_.name for f_ in fields(Skill)}
                for d in data.get("skills", []):
                    s = Skill(**{k: v for k, v in d.items() if k in valid})
                    self.skills[s.id] = s
            except Exception as e:  # never let a corrupt store crash a run
                print(f"[SkillStore] load failed ({e}) - starting empty")
        return self

    def save(self) -> None:
        try:
            os.makedirs(self.skills_dir, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"skills": [asdict(s) for s in self.skills.values()]},
                          f, indent=2, ensure_ascii=False)
            self._export_markdown()
        except Exception as e:
            print(f"[SkillStore] save failed: {e}")

    def _export_markdown(self) -> None:
        try:
            for s in self.skills.values():
                with open(os.path.join(self.skills_dir, f"{s.id}.SKILL.md"), "w", encoding="utf-8") as f:
                    f.write(s.to_markdown())
        except Exception:
            pass

    def seed_if_empty(self, initial_skills: Optional[List[Dict[str, Any]]] = None,
                      initial_rules: Optional[List[str]] = None) -> None:
        """Seed a fresh store. Prefers structured `initial_skills` dicts; falls back to
        wrapping plain `initial_codex_rules` strings as procedure-only skills."""
        if self.skills:
            return
        for sk in (initial_skills or []):
            if isinstance(sk, dict) and sk.get("name"):
                self.add(status=sk.get("status", "verified"), **{k: v for k, v in sk.items() if k != "status"})
        if not self.skills:
            for rule in (initial_rules or []):
                if rule and rule.strip():
                    self.add(name=rule.strip()[:48], procedure=rule.strip(), tags=["general"], status="proposed")

    # ---------- mutation ----------
    def add(self, name: str, when_to_use: str = "", procedure: str = "", pitfalls: str = "",
            verification: str = "", source_env: str = "", created_iter: int = 0,
            tags: Optional[List[str]] = None, status: str = "proposed",
            confidence: float = 0.5) -> str:
        sid = _slug(name)
        if status not in VALID_STATUS:
            status = "proposed"
        existing = self.skills.get(sid)
        # Pinning: never let a weaker (non-verified) write clobber a verified skill.
        if existing and existing.status == "verified" and status != "verified":
            return sid
        self.skills[sid] = Skill(
            id=sid, name=name, when_to_use=when_to_use, procedure=procedure,
            pitfalls=pitfalls, verification=verification, status=status,
            confidence=confidence, source_env=source_env, created_iter=created_iter,
            tags=list(tags or []),
        )
        return sid

    def improve(self, sid: str, **changes: Any) -> bool:
        s = self.skills.get(sid) or self.skills.get(_slug(sid))
        if not s:
            return False
        valid = {f_.name for f_ in fields(Skill)}
        for k, v in changes.items():
            if k not in valid or k == "id" or v is None:
                continue
            if k == "confidence":
                try:
                    v = float(v)  # reject garbage like "high" (would crash :.1f render + persist)
                except (TypeError, ValueError):
                    continue
            elif k == "status":
                if v not in VALID_STATUS:
                    continue
                if s.status == "verified" and v != "verified":
                    continue  # pinning: improve() must not un-verify a verified skill
            setattr(s, k, v)
        return True

    def verify(self, sid: str, iteration: int = 0) -> bool:
        s = self.skills.get(sid) or self.skills.get(_slug(sid))
        if not s:
            return False
        s.status = "verified"
        s.confidence = min(1.0, s.confidence + 0.3)
        s.last_validated_iter = iteration
        return True

    def deprecate(self, sid: str) -> bool:
        s = self.skills.get(sid) or self.skills.get(_slug(sid))
        if not s:
            return False
        # Verified skills are pinned: demote confidence first; only drop when it bottoms out.
        if s.status == "verified":
            s.confidence = max(0.0, s.confidence - 0.3)
            if s.confidence <= 0.1:
                s.status = "deprecated"
        else:
            s.status = "deprecated"
        return True

    # ---------- retrieval ----------
    def active(self) -> List[Skill]:
        return [s for s in self.skills.values() if s.status != "deprecated"]

    def relevant(self, env_name: str = "", tags: Optional[List[str]] = None,
                 query: str = "") -> List[Skill]:
        """Skills relevant to the current task. For now: env/tag match + general skills.
        The signature is built for future embedding retrieval (rank `query` against each
        skill's embeddable_text()) without changing callers."""
        want = set(t.lower() for t in (tags or []))
        out: List[Skill] = []
        for s in self.active():
            stags = set(t.lower() for t in s.tags)
            is_general = ("general" in stags) or (not stags)
            env_match = bool(env_name) and (env_name == s.source_env or env_name.lower() in stags)
            tag_match = bool(want & stags)
            if is_general or env_match or tag_match or (not env_name and not want):
                out.append(s)
        out.sort(key=lambda s: (s.status != "verified", -s.confidence))
        return out

    def render_for_coder(self, env_name: str = "", tags: Optional[List[str]] = None,
                         max_skills: int = 12) -> str:
        rel = self.relevant(env_name=env_name, tags=tags)[:max_skills]
        if not rel:
            return ""
        blocks = ["", "=== LEARNED SKILLS (procedures hard-won from past runs - apply the ones that fit) ==="]
        for s in rel:
            blocks.append(s.render())
        blocks.append("=== end skills ===")
        return "\n".join(blocks)

    def render_summary(self) -> str:
        """One-line-per-skill summary for the Reviewer (so SHODAN knows what exists)."""
        if not self.skills:
            return "(no skills yet)"
        return "\n".join(
            f"[{s.id}] ({s.status}, conf {s.confidence:.1f}) {s.name}"
            for s in sorted(self.skills.values(), key=lambda s: (s.status != "verified", -s.confidence))
        )

    def apply_ops(self, ops: Dict[str, Any], iteration: int = 0) -> List[str]:
        """Apply SHODAN's skill operations. Returns human-readable log lines.

        ops = {
          "add":     [{"name":..,"when_to_use":..,"procedure":..,"pitfalls":..,"verification":..,"tags":[..]}],
          "improve": [{"id":.., <fields to change>}],
          "verify":  ["skill-id", ...],
          "remove":  ["skill-id", ...],
        }
        """
        log: List[str] = []
        if not isinstance(ops, dict):
            return log
        _ADD_FIELDS = {"name", "when_to_use", "procedure", "pitfalls", "verification",
                       "tags", "status", "confidence", "source_env"}
        for sk in ops.get("add", []) or []:
            if isinstance(sk, dict) and sk.get("name"):
                # Only pass safe fields - NOT id/created_iter/last_validated_iter (created_iter
                # is set explicitly below; passing it again would be a duplicate-kwarg TypeError).
                sid = self.add(created_iter=iteration, **{k: v for k, v in sk.items() if k in _ADD_FIELDS})
                log.append(f"SKILL +add [{sid}] {sk.get('name')}")
        for ch in ops.get("improve", []) or []:
            if isinstance(ch, dict) and ch.get("id"):
                if self.improve(ch["id"], **{k: v for k, v in ch.items() if k != "id"}):
                    log.append(f"SKILL ~improve [{ch['id']}]")
        for sid in ops.get("verify", []) or []:
            if self.verify(str(sid), iteration):
                log.append(f"SKILL OK verify [{sid}]")
        for sid in ops.get("remove", []) or []:
            if self.deprecate(str(sid)):
                log.append(f"SKILL -deprecate [{sid}]")
        if log:
            self.save()
        return log
