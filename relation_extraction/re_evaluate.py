"""Evaluate toponym relation classification against gold standard.

Compares the RE output's spatial categories (derived from event roles)
against the gold standard CSV with Soni et al. classifications.

Usage:
    python relation_extraction/re_evaluate.py \\
        --model deepseek-v4-flash --temperature 0.0 --inference-type cloud \\
        --source-text 1816_third_letter --duration 120.5
"""

import argparse
import csv
import json
import os
import re as _re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from rdflib import Graph, URIRef
from rdflib.namespace import RDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cli_flag_passed(flag, argv=None):
    """True if ``flag`` was given on the CLI.

    Handles both the space form (``--flag value``) and the equals form
    (``--flag=value``); the bare ``"--flag" not in sys.argv`` check misses the
    latter and lets auto-detection silently overwrite an explicit value.
    """
    if argv is None:
        argv = sys.argv
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def load_gold_standard(csv_path):
    """Load gold standard CSV, return list of E53/E18 rows with visited_type."""
    entities = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            vt = row.get("visited_type", "").strip()
            if label in ("E53_Place", "E18_Physical_Thing") and vt:
                # Normalise "NO REL" → "NO_REL" for consistency
                vt = vt.replace(" ", "_")
                entities.append({
                    "text": row.get("text", "").strip(),
                    "label": label,
                    "start": int(row.get("start_char", 0)),
                    "end": int(row.get("end_char", 0)),
                    "visited_type": vt,
                })
    return entities


def load_mention_map(path):
    """Load mention map JSON → dict mention_id → {text, label, start, end}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for mid, info in data.items():
        result[mid] = {
            "text": info.get("text", ""),
            "label": info.get("label", ""),
            "start": info.get("start", 0),
            "end": info.get("end", 0),
        }
    return result


def _normalize_toponym_relations(data):
    """Coerce a list-shaped ``toponym_relations`` (an LLM variant) to a dict.

    The prompt asks for an object ``{"e3": "FROM", ...}`` but models sometimes
    return an array of objects ``[{"mention_id": "e3", "category": "FROM"}]``.
    Downstream code calls ``.items()`` on it, which would raise
    ``AttributeError`` on a list. Normalise once at the load boundary.
    """
    tr = data.get("toponym_relations")
    if isinstance(tr, list):
        d = {}
        for item in tr:
            if isinstance(item, dict):
                mid = item.get("mention_id")
                cat = item.get("category")
                if mid and cat:
                    d[mid] = cat
        data["toponym_relations"] = d
    elif tr is None:
        data["toponym_relations"] = {}
    return data


def load_events(path):
    """Load RE events JSON, normalising ``toponym_relations`` to a dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_toponym_relations(data)


def derive_binary_from_rdf(rdf_path, mention_map, letter_id="BRF0003"):
    """Derive visited/non-visited from the generated RDF/TTL file.

    A LOC entity is 'visited' if it appears as the object of at least one
    P7_took_place_at, P26_moved_to, or P27_moved_from triple in the RDF.
    Otherwise it's 'non-visited'.

    Returns (mention_visited, loc_visited, loc_map):
      mention_visited: mention_id -> True/False (per-citation, dedup-consistent).
      loc_visited:     LOC uri -> True/False (per-place, authoritative — a LOC is
                       visited iff it is the object of a visit-property triple).
      loc_map:         mention_id -> LOC uri (the RDF's own dedup partition; all
                       mentions sharing a KB id collapse onto the same LOC).
    Only mention_ids that map to E53_Place/E18_Physical_Thing are included.
    """
    g = Graph()
    g.parse(rdf_path, format="turtle")

    # Namespaces
    CRM = URIRef("http://www.cidoc-crm.org/cidoc-crm/")
    ATO = URIRef("http://academictourism.com/entity/")

    # Properties that indicate a location was visited
    VISIT_PROPS = {
        URIRef("http://www.cidoc-crm.org/cidoc-crm/P7_took_place_at"),
        URIRef("http://www.cidoc-crm.org/cidoc-crm/P26_moved_to"),
        URIRef("http://www.cidoc-crm.org/cidoc-crm/P27_moved_from"),
        URIRef("http://academictourism.com/academictourism#happened_in_proximity_of"),
    }

    # 1. Find all LOC entities (ato:LOC.* with type E53_Place or E18_Physical_Thing)
    loc_entities = set()
    for s in g.subjects(RDF.type, CRM + "E53_Place"):
        if str(s).startswith(str(ATO) + "LOC."):
            loc_entities.add(s)
    for s in g.subjects(RDF.type, CRM + "E18_Physical_Thing"):
        if str(s).startswith(str(ATO) + "LOC."):
            loc_entities.add(s)

    # 2. Determine which LOC entities are visited
    visited_locs = set()
    for loc in loc_entities:
        # A LOC is visited if it is the object of any visit property triple.
        for s, p, o in g.triples((None, None, loc)):
            if p in VISIT_PROPS:
                visited_locs.add(loc)
                break

    # 3. Map LOC entities → mention_ids via P67i_is_referred_to_by
    #    LOC.Paderborn has: P67i_is_referred_to_by ato:CT.BRF0003.e5, ato:CT.BRF0003.e11, ...
    #    The CT URI contains the mention_id (e.g. e5)
    mention_visited = {}   # mention_id -> visited (per-citation, dedup-consistent)
    loc_map = {}           # mention_id -> LOC uri (the RDF's own dedup partition)
    ct_prefix = str(ATO) + f"CT.{letter_id}."
    for loc in loc_entities:
        is_visited = loc in visited_locs
        for ct_entity in g.objects(loc, CRM + "P67i_is_referred_to_by"):
            ct_str = str(ct_entity)
            if ct_str.startswith(ct_prefix):
                mid = ct_str[len(ct_prefix):]
                # Only include if it's an E53/E18 in the mention map
                if mid in mention_map and mention_map[mid]["label"] in ("E53_Place", "E18_Physical_Thing"):
                    mention_visited[mid] = is_visited
                    loc_map[mid] = str(loc)

    # Per-LOC visited verdict (authoritative): a LOC is visited iff it is the
    # object of a visit-property triple. This is the granularity the RDF can
    # actually express — one deduped LOC per place — so the binary RDF metric
    # is scored per-place against this, not per-citation.
    loc_visited = {str(loc): (loc in visited_locs) for loc in loc_entities}

    return mention_visited, loc_visited, loc_map


def derive_predictions(events_data, mention_map):
    """Derive predicted Soni categories from the LLM output.

    Primary source is ``toponym_relations`` (the LLM's explicit per-entity
    classification). Event roles are used as a fallback for entities not in
    ``toponym_relations``; ``mentioned_only_places`` confirms NO_REL.
    Remaining E53/E18 default to NO_REL (an entity the LLM did not classify
    is non-visited, not "in proximity").
    """
    predictions = {}  # mention_id → category

    # Primary: the LLM's explicit per-entity classification
    for mid, cat in events_data.get("toponym_relations", {}).items():
        predictions[mid] = cat

    # Fallback: event roles for entities not covered by toponym_relations
    for ev in events_data.get("events", []):
        if ev.get("moved_from") and ev["moved_from"] not in predictions:
            predictions[ev["moved_from"]] = "FROM"
        if ev.get("moved_to") and ev["moved_to"] not in predictions:
            predictions[ev["moved_to"]] = "TO"
        for vp in ev.get("via_points", []):
            if vp not in predictions:
                predictions[vp] = "THRU"
        if ev.get("took_place_at") and ev["took_place_at"] not in predictions:
            predictions[ev["took_place_at"]] = "IN"
        if ev.get("on_or_within") and ev["on_or_within"] not in predictions:
            predictions[ev["on_or_within"]] = "IN"
        for np in ev.get("near_places", []):
            if np not in predictions:
                predictions[np] = "NEAR"

    # NO_REL from mentioned_only_places (only if not already classified)
    for mp in events_data.get("mentioned_only_places", []):
        mid = mp.get("mention_id") if isinstance(mp, dict) else mp
        if mid and mid not in predictions:
            predictions[mid] = "NO_REL"

    # Default: remaining E53/E18 → NO_REL (toponym_relations covers ~all;
    # an unclassified entity is non-visited, not "in proximity")
    for mid, info in mention_map.items():
        if info["label"] in ("E53_Place", "E18_Physical_Thing"):
            if mid not in predictions:
                predictions[mid] = "NO_REL"

    return predictions


def match_gold_to_mention(gold_entities, mention_map):
    """Match gold standard entities to mention_map entries by overlapping spans.

    Returns list of (gold_entity, mention_id) tuples.
    Uses relaxed overlap: any overlap between gold and mention span.
    If multiple matches, picks the one with the highest Jaccard overlap.
    """
    matches = []
    for g in gold_entities:
        best_mid = None
        best_score = 0
        for mid, info in mention_map.items():
            if info["label"] != g["label"]:
                continue
            # Check span overlap
            overlap_start = max(g["start"], info["start"])
            overlap_end = min(g["end"], info["end"])
            if overlap_start < overlap_end:
                # Jaccard-like: intersection / union
                intersection = overlap_end - overlap_start
                union = max(g["end"], info["end"]) - min(g["start"], info["start"])
                score = intersection / union if union > 0 else 0
                if score > best_score:
                    best_score = score
                    best_mid = mid
        if best_mid:
            matches.append((g, best_mid))
    return matches


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_fine_grained(gold_entities, predictions, mention_map):
    """Per-category precision, recall, F1 for THRU/TO/FROM/IN/NEAR/NO_REL."""
    matches = match_gold_to_mention(gold_entities, mention_map)

    # Build gold and predicted label lists
    gold_labels = []
    pred_labels = []
    for g, mid in matches:
        gold_labels.append(g["visited_type"])
        pred_labels.append(predictions.get(mid, "NEAR"))

    categories = ["THRU", "TO", "FROM", "IN", "NEAR", "NO_REL"]
    results = {}
    for cat in categories:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cat and p == cat)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != cat and p == cat)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cat and p != cat)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
        }

    # Macro average
    macro_p = sum(r["precision"] for r in results.values()) / len(categories)
    macro_r = sum(r["recall"] for r in results.values()) / len(categories)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0

    # Micro average (accuracy-equivalent)
    total_tp = sum(r["tp"] for r in results.values())
    total = len(gold_labels)
    micro_f1 = total_tp / total if total > 0 else 0

    return results, {
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(macro_f1, 3),
        "micro_f1": round(micro_f1, 3),
        "total": total,
        "matched": len(matches),
    }


def evaluate_binary(gold_entities, predictions, mention_map):
    """Binary visited (THRU/TO/FROM/IN/NEAR) vs non-visited (NO_REL).

    Reports precision, recall, F1 for the 'visited' class.
    """
    matches = match_gold_to_mention(gold_entities, mention_map)

    visited_cats = {"THRU", "TO", "FROM", "IN", "NEAR"}

    tp = 0  # gold=visited, pred=visited
    fp = 0  # gold=non-visited, pred=visited
    fn = 0  # gold=visited, pred=non-visited
    tn = 0  # gold=non-visited, pred=non-visited

    for g, mid in matches:
        gold_visited = g["visited_type"] in visited_cats
        pred_visited = predictions.get(mid, "NEAR") in visited_cats
        if gold_visited and pred_visited:
            tp += 1
        elif not gold_visited and pred_visited:
            fp += 1
        elif gold_visited and not pred_visited:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "total": len(matches),
    }


def evaluate_binary_rdf(gold_entities, mention_map, rdf_visited):
    """Binary visited vs non-visited using RDF-derived predictions.

    *rdf_visited* is a dict: mention_id → True (visited) / False (non-visited).
    Only mention_ids present in rdf_visited are evaluated (those that made it
    into the RDF output at all).
    """
    matches = match_gold_to_mention(gold_entities, mention_map)

    # Filter to matches where the mention_id is in the RDF output
    rdf_matches = [(g, mid) for g, mid in matches if mid in rdf_visited]

    tp = fp = fn = tn = 0
    for g, mid in rdf_matches:
        gold_visited = g["visited_type"] in {"THRU", "TO", "FROM", "IN", "NEAR"}
        pred_visited = rdf_visited[mid]
        if gold_visited and pred_visited:
            tp += 1
        elif not gold_visited and pred_visited:
            fp += 1
        elif gold_visited and not pred_visited:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "total": len(rdf_matches),
        "dropped": len(matches) - len(rdf_matches),
    }


def evaluate_binary_rdf_per_place(gold_entities, mention_map, loc_visited, loc_map):
    """Per-place (LOC-level) binary visited vs non-visited from the RDF.

    The RDF deduplicates mentions of the same place into a single LOC entity,
    so scoring it per-citation double-counts a place that is mentioned twice.
    This collapses matched gold mentions onto the RDF's own LOC partition
    (``loc_map``) and scores one binary verdict per place:

      * gold-LOC visited   = any member mention's ``visited_type`` is a
                              visited category (THRU/TO/FROM/IN/NEAR).
      * RDF-LOC visited     = ``loc_visited[LOC]`` — the LOC is the object of
                              a visit-property triple in the RDF.

    ``total`` is the number of gold-anchored LOCs in the universe (one per
    distinct place that has ≥1 matched gold mention). ``dropped`` counts
    matched gold mentions whose mention_id has no LOC in the RDF (no CT/LOC
    was generated for them) and are therefore excluded from the per-place
    universe.
    """
    visited_cats = {"THRU", "TO", "FROM", "IN", "NEAR"}
    matches = match_gold_to_mention(gold_entities, mention_map)

    loc_gold_types = {}   # LOC uri -> list of member gold visited_types
    dropped = 0
    for g, mid in matches:
        loc = loc_map.get(mid)
        if loc is None:
            dropped += 1
            continue
        loc_gold_types.setdefault(loc, []).append(g["visited_type"])

    tp = fp = fn = tn = 0
    for loc, types in loc_gold_types.items():
        gold_visited = any(t in visited_cats for t in types)
        pred_visited = loc_visited.get(loc, False)
        if gold_visited and pred_visited:
            tp += 1
        elif not gold_visited and pred_visited:
            fp += 1
        elif gold_visited and not pred_visited:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = (tp + tn) / len(loc_gold_types) if loc_gold_types else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "total": len(loc_gold_types),
        "dropped": dropped,
    }


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def confusion_matrix(gold_entities, predictions, mention_map):
    """Print a simple confusion matrix."""
    matches = match_gold_to_mention(gold_entities, mention_map)
    categories = ["THRU", "TO", "FROM", "IN", "NEAR", "NO_REL"]
    cm = {g: {p: 0 for p in categories} for g in categories}
    for g, mid in matches:
        gold = g["visited_type"]
        pred = predictions.get(mid, "NEAR")
        if gold in cm and pred in cm[gold]:
            cm[gold][pred] += 1

    print(f"\n{'':>14}", " ".join(f"{c:>6}" for c in categories))
    print(f"{'':>14}", "-" * (7 * len(categories)))
    for gold in categories:
        row = " ".join(f"{cm[gold][p]:>6}" for p in categories)
        print(f"{gold:>12}  | {row}")
    print()


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

RDF_DIR = Path(__file__).resolve().parent.parent / "output" / "rdf"
RE_DIR = Path(__file__).resolve().parent.parent / "output" / "re"


def select_ttl_file():
    """Scan output/rdf/ for *_events.ttl files and let the user pick one."""
    ttl_files = sorted(RDF_DIR.glob("*_events.ttl"))
    if not ttl_files:
        print(f"No *_events.ttl files found in {RDF_DIR}")
        print("Run relation_extraction/rel_extraction.py first.")
        raise SystemExit(1)

    if len(ttl_files) == 1:
        print(f"Auto-selected: {ttl_files[0].name}")
        return str(ttl_files[0])

    print("Available events TTL files:")
    for i, f in enumerate(ttl_files, 1):
        print(f"  [{i}] {f.name}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ttl_files):
            return str(ttl_files[idx])
    except ValueError:
        pass
    print(f"Invalid selection: {choice}")
    raise SystemExit(1)


def find_matching_files(ttl_path):
    """Given a TTL path, find the corresponding events JSON and mention map.

    E.g.  output/rdf/1816...__deepseek-v4-flash_t0.7_events.ttl
      →   output/re/1816...__deepseek-v4-flash_t0.7_events.json
      →   output/re/1816...__deepseek-v4-flash_t0.7_mention_map.json
    """
    stem = Path(ttl_path).stem  # e.g. 1816...__deepseek-v4-flash_t0.7_events
    if stem.endswith("_events"):
        base = stem[:-7]  # strip _events suffix
    else:
        base = stem

    events_json = RE_DIR / f"{base}_events.json"
    mention_map_json = RE_DIR / f"{base}_mention_map.json"

    if not events_json.exists():
        print(f"ERROR: expected events JSON not found: {events_json}")
        raise SystemExit(1)
    if not mention_map_json.exists():
        print(f"ERROR: expected mention map not found: {mention_map_json}")
        raise SystemExit(1)

    return str(events_json), str(mention_map_json)


# ---------------------------------------------------------------------------
# CSV + metadata output
# ---------------------------------------------------------------------------

SCORES_DIR = Path(__file__).resolve().parent / "re-evaluation"


def write_scores_csv(meta, fine_results, fine_summary, bin_results, rdf_bin_results):
    """Append one row to scores.csv, creating the file with header if needed.

    The ``binary_rdf_*`` and ``rdf_entities``/``rdf_dropped`` columns come from
    the **per-place (LOC-level)** RDF binary evaluation: the RDF deduplicates
    mentions of the same place into one LOC, so the metric is scored per place,
    not per citation. ``rdf_entities`` is the number of gold-anchored LOCs in
    the universe and ``rdf_dropped`` is the count of matched gold mentions with
    no LOC in the RDF.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORES_DIR / "scores.csv"

    row = {
        "source_text": meta["source_text"],
        "model": meta["model"],
        "temperature": meta["temperature"],
        "think_mode": meta.get("think_mode", ""),
        "inference_type": meta["inference_type"],
        "datetime": meta["datetime"],
        "duration_seconds": meta["duration_seconds"],
        # Fine-grained
        "fine_macro_f1": fine_summary["macro_f1"],
        "fine_micro_f1": fine_summary["micro_f1"],
        "thru_f1": fine_results["THRU"]["f1"],
        "to_f1": fine_results["TO"]["f1"],
        "from_f1": fine_results["FROM"]["f1"],
        "in_f1": fine_results["IN"]["f1"],
        "near_f1": fine_results["NEAR"]["f1"],
        "no_rel_f1": fine_results["NO_REL"]["f1"],
        # Binary JSON (per-citation, from the LLM toponym_relations)
        "binary_json_p": bin_results["precision"],
        "binary_json_r": bin_results["recall"],
        "binary_json_f1": bin_results["f1"],
        "binary_json_accuracy": bin_results["accuracy"],
        # Binary RDF (per-place / LOC-level — the RDF's own dedup granularity)
        "binary_rdf_p": rdf_bin_results["precision"],
        "binary_rdf_r": rdf_bin_results["recall"],
        "binary_rdf_f1": rdf_bin_results["f1"],
        "binary_rdf_accuracy": rdf_bin_results["accuracy"],
        "rdf_entities": rdf_bin_results["total"],
        "rdf_dropped": rdf_bin_results.get("dropped", 0),
    }

    fieldnames = list(row.keys())
    exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nScores appended to: {path}")


def write_metadata_json(meta):
    """Write a small JSON metadata file alongside scores.csv for provenance."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = _re.sub(r"[^a-zA-Z0-9_.-]", "_", meta["model"])
    ts = meta["datetime"].replace(" ", "_").replace(":", "-")
    filename = f"re_eval_{safe_model}_{ts}.json"
    path = SCORES_DIR / filename
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to: {path}")


def extract_model_from_filename(ttl_path):
    """Extract RE model name, temperature, and think mode from the TTL filename.

    Filename format (both old and new):
      {source}__{model}_t{temp}_events.ttl
      {source}__{model}_t{temp}_think{Mode}_events.ttl
      {source}__{el_model}_t{el_temp}__{re_model}_t{re_temp}_events.ttl
      {source}__{el_model}_t{el_temp}__{re_model}_t{re_temp}_think{Mode}_events.ttl

    ``_thinkDefault`` means the ``--think`` flag was not passed (model default).
    ``_thinkTrue`` / ``_thinkFalse`` / ``_thinkLow`` / ``_thinkMedium`` / ``_thinkHigh``
    mean the flag was explicitly set.

    Returns (model_name, temperature, think_mode) or (None, None, None).
    think_mode is returned lowercase (e.g. "true", "false", "low", "default")
    to match argparse choices.
    """
    stem = Path(ttl_path).stem
    if stem.endswith("_events"):
        base = stem[:-7]
    else:
        base = stem

    parts = base.split("__")
    # Take the last __{model}_t{temp}[_think{Mode}] segment
    if len(parts) >= 2:
        last = parts[-1]
        # Try with think mode first
        m = _re.match(r'^(.+)_t(\d+\.?\d*)_think(\w+)$', last)
        if m:
            return m.group(1), float(m.group(2)), m.group(3).lower()
        # Fall back to without think mode (legacy filenames)
        m = _re.match(r'^(.+)_t(\d+\.?\d*)$', last)
        if m:
            return m.group(1), float(m.group(2)), None
    return None, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RE toponym relations")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model name used for RE")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature setting")
    parser.add_argument("--inference-type", default="cloud",
                        choices=["cloud", "local"],
                        help="Inference environment")
    parser.add_argument("--source-text", default="1816_third_letter",
                        help="Source text identifier")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Duration in seconds")
    parser.add_argument("--think", default=None,
                        choices=["true", "false", "low", "medium", "high", "default"],
                        help="Thinking mode (overrides auto-detect from metadata). "
                             "'default' means the --think flag was not passed at "
                             "extraction time (model default).")
    parser.add_argument("--gold", default=None,
                        help="Path to gold standard CSV (default: alongside this script)")
    parser.add_argument("--events", default=None,
                        help="Path to RE events JSON (omit for interactive picker)")
    parser.add_argument("--mention-map", default=None,
                        help="Path to mention map JSON (omit for interactive picker)")
    parser.add_argument("--rdf", default=None,
                        help="Path to RDF/TTL output (omit for interactive picker)")
    args = parser.parse_args()

    # Resolve gold path relative to script directory
    if args.gold is None:
        args.gold = str(Path(__file__).resolve().parent / "1816_re_gs.csv")

    # Interactive file selection if no explicit paths given
    if args.rdf is None:
        args.rdf = select_ttl_file()
    if args.events is None or args.mention_map is None:
        ev, mm = find_matching_files(args.rdf)
        if args.events is None:
            args.events = ev
        if args.mention_map is None:
            args.mention_map = mm

    # Auto-detect model and temperature from filename if not explicitly set
    if (not _cli_flag_passed("--model") or not _cli_flag_passed("--temperature")
            or not _cli_flag_passed("--think")):
        detected_model, detected_temp, detected_think = extract_model_from_filename(args.rdf)
        if detected_model and not _cli_flag_passed("--model"):
            args.model = detected_model
        if detected_temp is not None and not _cli_flag_passed("--temperature"):
            args.temperature = detected_temp
        if detected_think is not None and not _cli_flag_passed("--think"):
            args.think = detected_think

    # Auto-detect source text from the TTL filename if not explicitly set
    if not _cli_flag_passed("--source-text"):
        _base = Path(args.rdf).stem
        if _base.endswith("_events"):
            _base = _base[:-7]
        _src = _base.split("__")[0]
        if _src:
            args.source_text = _src

    # Auto-detect duration from metadata JSON if not explicitly set
    if not _cli_flag_passed("--duration") and args.events:
        meta_path = Path(args.events).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                args.duration = meta.get("duration_seconds", 0.0)
                print(f"   Duration auto-detected from metadata: {args.duration}s")
            except (json.JSONDecodeError, OSError):
                pass

    print("=" * 60)
    print("Toponym Relation Evaluation")
    print("=" * 60)

    # Load data
    gold = load_gold_standard(args.gold)
    mention_map = load_mention_map(args.mention_map)
    events_data = load_events(args.events)

    print(f"\nGold standard E53/E18 with visited_type: {len(gold)}")
    print(f"Mention map entries: {len(mention_map)}")

    # Derive predictions from JSON events
    predictions = derive_predictions(events_data, mention_map)
    n_pred = sum(1 for mid in predictions
                 if mention_map.get(mid, {}).get("label") in ("E53_Place", "E18_Physical_Thing"))
    print(f"Predicted E53/E18 with categories: {n_pred}")

    # Count gold categories
    gold_counts = Counter(g["visited_type"] for g in gold)
    print(f"\nGold distribution: {dict(gold_counts)}")

    # Count predicted categories (for matched E53/E18 only)
    pred_counts = Counter()
    for mid, cat in predictions.items():
        info = mention_map.get(mid, {})
        if info.get("label") in ("E53_Place", "E18_Physical_Thing"):
            pred_counts[cat] += 1
    print(f"Predicted distribution: {dict(pred_counts)}")

    # --- Fine-grained evaluation ---
    print("\n" + "-" * 60)
    print("1. Fine-grained classification (6 categories)")
    print("-" * 60)
    fine_results, fine_summary = evaluate_fine_grained(gold, predictions, mention_map)
    print(f"\n{'Category':>12}  {'TP':>4} {'FP':>4} {'FN':>4}  "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"{'':>12}  {'-'*3} {'-'*3} {'-'*3}  {'-'*5} {'-'*5} {'-'*5}")
    for cat in ["THRU", "TO", "FROM", "IN", "NEAR", "NO_REL"]:
        r = fine_results[cat]
        print(f"{cat:>12}  {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
    print(f"\nMacro avg:  P={fine_summary['macro_precision']:.3f}  "
          f"R={fine_summary['macro_recall']:.3f}  F1={fine_summary['macro_f1']:.3f}")
    print(f"Micro avg (accuracy):  F1={fine_summary['micro_f1']:.3f}")
    print(f"Total matched entities: {fine_summary['matched']}/{fine_summary['total']}")

    # Confusion matrix
    print("\nConfusion matrix (rows=gold, cols=predicted):")
    confusion_matrix(gold, predictions, mention_map)

    # --- Binary evaluation (from JSON events) ---
    print("-" * 60)
    print("2a. Binary classification — from JSON events")
    print("-" * 60)
    bin_results = evaluate_binary(gold, predictions, mention_map)
    print(f"\n{'Metric':>15}  {'Value':>6}")
    print(f"{'':>15}  {'-'*6}")
    print(f"{'TP':>15}  {bin_results['tp']:>6}")
    print(f"{'FP':>15}  {bin_results['fp']:>6}")
    print(f"{'FN':>15}  {bin_results['fn']:>6}")
    print(f"{'TN':>15}  {bin_results['tn']:>6}")
    print(f"{'Precision':>15}  {bin_results['precision']:>6.3f}")
    print(f"{'Recall':>15}  {bin_results['recall']:>6.3f}")
    print(f"{'F1':>15}  {bin_results['f1']:>6.3f}")
    print(f"{'Accuracy':>15}  {bin_results['accuracy']:>6.3f}")
    print(f"{'Total':>15}  {bin_results['total']:>6}")

    # --- Binary evaluation (from RDF, per-place / LOC-level) ---
    print("\n" + "-" * 60)
    print("2b. Binary classification — from RDF/TTL (per-place / LOC-level)")
    print("-" * 60)
    rdf_visited, loc_visited, loc_map = derive_binary_from_rdf(args.rdf, mention_map)
    print(f"\nRDF LOC entities (places): {len(loc_visited)}  "
          f"(visited: {sum(1 for v in loc_visited.values() if v)}, "
          f"non-visited: {sum(1 for v in loc_visited.values() if not v)})")
    print(f"RDF citations mapped to a LOC: {len(loc_map)}")
    # Per-place (LOC-level): score the RDF at its natural granularity (one
    # deduped LOC per place) instead of double-counting repeated mentions.
    rdf_bin_results = evaluate_binary_rdf_per_place(gold, mention_map, loc_visited, loc_map)
    print(f"\n{'Metric':>15}  {'Value':>6}")
    print(f"{'':>15}  {'-'*6}")
    print(f"{'TP':>15}  {rdf_bin_results['tp']:>6}")
    print(f"{'FP':>15}  {rdf_bin_results['fp']:>6}")
    print(f"{'FN':>15}  {rdf_bin_results['fn']:>6}")
    print(f"{'TN':>15}  {rdf_bin_results['tn']:>6}")
    print(f"{'Precision':>15}  {rdf_bin_results['precision']:>6.3f}")
    print(f"{'Recall':>15}  {rdf_bin_results['recall']:>6.3f}")
    print(f"{'F1':>15}  {rdf_bin_results['f1']:>6.3f}")
    print(f"{'Accuracy':>15}  {rdf_bin_results['accuracy']:>6.3f}")
    print(f"{'LOCs':>15}  {rdf_bin_results['total']:>6}")
    if rdf_bin_results.get("dropped", 0):
        print(f"{'Dropped':>15}  {rdf_bin_results['dropped']:>6}  "
              f"(matched gold mentions with no LOC in the RDF)")

    # --- Write scores.csv + metadata ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Try to load think_mode from metadata JSON (unless explicitly set via --think)
    think_mode = args.think if args.think is not None else ""
    if not think_mode and args.events:
        meta_path = Path(args.events).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    re_meta = json.load(f)
                think_mode = re_meta.get("think_mode") or ""
            except (json.JSONDecodeError, OSError):
                pass
    # Normalise the "not passed" state to "default" so scores.csv is consistent
    # regardless of whether the value came from filename auto-detect, the meta
    # JSON, or nowhere at all.
    if not think_mode:
        think_mode = "default"

    meta = {
        "source_text": args.source_text,
        "model": args.model,
        "temperature": args.temperature,
        "think_mode": think_mode,
        "inference_type": args.inference_type,
        "datetime": now,
        "duration_seconds": args.duration,
    }
    write_scores_csv(meta, fine_results, fine_summary,
                     bin_results, rdf_bin_results)
    write_metadata_json(meta)

    print("\n" + "=" * 60)
    print("Done.")
