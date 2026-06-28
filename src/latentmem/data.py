"""Controlled user-memory data for the latent-extraction pilot.

The pilot asks: can a small *learned write head* compress a user-memory
document into k continuous vectors M such that a frozen LM, conditioned on
M alone, answers as well as if it had the full document in context?

We need a task where (a) the answer is governed by specific facts in the
document, (b) the gold label is unambiguous (no LLM judge), and (c) the
document length and the number of governing facts are both controllable so
we can measure where latent compression starts to break. The synthetic
persona / gated-decision design from the editable-kv memory experiments fits
exactly; this is a self-contained re-implementation so the pilot has no
cross-repo dependency.

Each example is:
  doc    : a USER-MEMORY markdown listing M account settings (enabled/disabled)
  probe  : a yes/no question whose answer is the AND of `n_relevant` settings
           (gated-decision probe) OR a single-setting value lookup (recall probe)
  answer : "yes" / "no"  (single leading token after the answer cue)

Pure-python, deterministic given a seed. No torch dependency.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

# (category, attribute, human-readable note) — the setting catalogue.
CATALOG: List[Tuple[str, str, str]] = [
    ("notifications", "marketing_emails", "promotional email campaigns"),
    ("notifications", "push_alerts", "mobile push notifications"),
    ("notifications", "sms_updates", "text-message status updates"),
    ("privacy", "data_sharing", "share usage data with partners"),
    ("privacy", "personalized_ads", "ad personalization from activity"),
    ("privacy", "location_history", "retain precise location history"),
    ("privacy", "third_party_export", "export records to third parties"),
    ("security", "two_factor", "two-factor authentication requirement"),
    ("security", "new_device_login", "auto-approve logins from new devices"),
    ("security", "biometric_unlock", "biometric unlock on this account"),
    ("billing", "auto_renew", "automatic subscription renewal"),
    ("billing", "overdraft_purchases", "allow purchases that overdraw balance"),
    ("billing", "saved_cards", "store card details for one-click pay"),
    ("content", "mature_content", "show age-restricted content"),
    ("content", "external_links", "open links to external sites"),
    ("content", "beta_features", "enroll in experimental beta features"),
    ("sharing", "public_profile", "make profile publicly visible"),
    ("sharing", "activity_status", "broadcast online/active status"),
    ("sharing", "contact_sync", "sync device contacts to the service"),
    ("automation", "auto_reply", "send automated replies when away"),
    ("automation", "smart_suggestions", "AI smart-suggestion features"),
    ("automation", "background_sync", "sync data in the background"),
    ("comms", "newsletter", "weekly product newsletter"),
    ("comms", "survey_invites", "invitations to user surveys"),
    ("accessibility", "high_contrast", "high-contrast display mode"),
    ("accessibility", "screen_reader", "screen-reader optimizations"),
    ("data", "cloud_backup", "encrypted cloud backups"),
    ("data", "cross_device", "cross-device session handoff"),
    ("data", "analytics_optin", "anonymous analytics opt-in"),
    ("integrations", "calendar_access", "calendar integration access"),
    ("integrations", "email_access", "email inbox integration access"),
    ("integrations", "drive_access", "cloud-drive file access"),
]

ACTIONS = [
    "send a promotional offer to the user",
    "share the user's record with a partner service",
    "auto-approve a login from an unrecognized device",
    "charge the user's saved card for renewal",
    "publish an update to the user's public profile",
    "sync the user's contacts to the cloud",
    "enroll the user in a new beta feature",
    "export the user's data to a third-party tool",
]


@dataclass
class Example:
    doc: str            # the user-memory document (context to compress)
    probe: str          # the question + answer cue (ends right before the answer)
    answer: str         # " yes" / " no"  (leading space => single clean token)
    kind: str           # "gated" | "recall"
    n_settings: int     # document length control
    n_relevant: int     # integration-depth control (gated probes)


def _render_doc(settings: List[dict]) -> str:
    lines = ["# USER MEMORY (account settings & preferences)", ""]
    for s in settings:
        state = "enabled" if s["enabled"] else "disabled"
        lines.append(f"- [{s['category']}] {s['attr']} ({s['note']}): {state}")
    return "\n".join(lines)


def make_example(rng: random.Random, n_settings: int = 16, n_relevant: int = 3,
                 kind: str = "gated") -> Example:
    """Generate one (doc, probe, answer) example.

    gated  : answer = "yes" iff ALL n_relevant governing settings are enabled.
             Label is balanced by construction (we force a coin-flip gold).
    recall : answer = the enabled/disabled state of one queried setting,
             remapped to yes/no ("yes" == enabled). A harder, single-fact probe.
    """
    n_settings = min(n_settings, len(CATALOG))
    chosen = rng.sample(CATALOG, n_settings)
    settings = [
        {"idx": i, "category": c, "attr": a, "note": n, "enabled": rng.random() < 0.5}
        for i, (c, a, n) in enumerate(chosen)
    ]

    if kind == "recall":
        q = rng.randrange(n_settings)
        tgt = settings[q]
        probe = (
            f"Question: Is the setting '{tgt['attr']}' ({tgt['note']}) currently "
            f"enabled for this user? Answer yes or no.\nAnswer:"
        )
        answer = " yes" if tgt["enabled"] else " no"
        return Example(_render_doc(settings), probe, answer, "recall", n_settings, 1)

    # gated decision
    n_relevant = max(1, min(n_relevant, n_settings))
    relevant = rng.sample(range(n_settings), n_relevant)
    # Force a balanced gold label: 50% of the time make all-enabled (yes),
    # otherwise disable exactly one governing setting (no).
    gold_yes = rng.random() < 0.5
    for r in relevant:
        settings[r]["enabled"] = True
    if not gold_yes:
        settings[rng.choice(relevant)]["enabled"] = False
    action = rng.choice(ACTIONS)
    rel_notes = "; ".join(settings[r]["note"] for r in relevant)
    probe = (
        f"Question: The system may {action} only if ALL of these are enabled: "
        f"{rel_notes}. Based on the user memory, may it proceed? Answer yes or no.\n"
        f"Answer:"
    )
    answer = " yes" if gold_yes else " no"
    return Example(_render_doc(settings), probe, answer, "gated", n_settings, n_relevant)


def make_multiprobe(rng: random.Random, n_settings: int = 16, n_probes: int = 8):
    """One doc plus n_probes single-fact recall probes over distinct settings.

    Returns (doc_text, [(probe, answer)]). This is the clean sufficiency signal:
    forcing M to answer many facts about the same document per step gives every
    setting dense gradient, instead of ~1/n_settings with one probe per doc.
    Labels are ~balanced because states are 50/50.
    """
    n_settings = min(n_settings, len(CATALOG))
    chosen = rng.sample(CATALOG, n_settings)
    settings = [{"idx": i, "category": c, "attr": a, "note": n, "enabled": rng.random() < 0.5}
                for i, (c, a, n) in enumerate(chosen)]
    doc = _render_doc(settings)
    qidx = rng.sample(range(n_settings), min(n_probes, n_settings))
    probes = []
    for q in qidx:
        t = settings[q]
        probe = (f"Question: Is the setting '{t['attr']}' ({t['note']}) currently "
                 f"enabled for this user? Answer yes or no.\nAnswer:")
        probes.append((probe, " yes" if t["enabled"] else " no"))
    return doc, probes


def make_multiprobe_dataset(n_docs: int, seed: int = 0, n_settings: int = 16,
                            n_probes: int = 8):
    rng = random.Random(seed)
    return [make_multiprobe(rng, n_settings, n_probes) for _ in range(n_docs)]


def make_dataset(n: int, seed: int = 0, n_settings: int = 16, n_relevant: int = 3,
                 recall_frac: float = 0.0) -> List[Example]:
    """A list of n examples. `recall_frac` mixes in single-fact recall probes."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = "recall" if rng.random() < recall_frac else "gated"
        out.append(make_example(rng, n_settings, n_relevant, kind))
    return out


if __name__ == "__main__":
    # Smoke test: print one of each kind and the label balance.
    ds = make_dataset(2000, seed=1, n_settings=16, n_relevant=3, recall_frac=0.5)
    yes = sum(1 for e in ds if e.answer.strip() == "yes")
    print(f"{len(ds)} examples, yes-rate={yes/len(ds):.3f} "
          f"(gated={sum(e.kind=='gated' for e in ds)}, recall={sum(e.kind=='recall' for e in ds)})")
    for kind in ("gated", "recall"):
        e = next(x for x in ds if x.kind == kind)
        print(f"\n===== {kind} =====\n{e.doc}\n\n{e.probe}{e.answer}")
