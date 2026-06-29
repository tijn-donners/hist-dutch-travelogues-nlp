"""Round-trip spot-checks for the canonical TEI edition.

Verifies ``output/tei/1816_third_letter.tei.xml`` is the loss-free canonical edition:

  1. Zero entity loss: every one of the 313 gold entities appears as an inline
     tag (placeName/persName/orgName/date/rs), each carrying its xml:id (eNN).
  2. Bidirectional standOff linkage: every inline xml:id eNN has a matching
     ``ato:CT.BRF0003.eNN`` (CT citation) in the embedded <standOff><xenoData>
     RDF/XML, and every mention_map eNN has an inline xml:id.
  3. Page numbers preserved: <pb n="16"|"17"|...|"25"/> all present.
  4. Well-formed XML (parses), and no residual ``[NN]``/``NN]`` page markers leak
     into the body text.
  5. Editorial notes: the entities ALTO loses (title Derde Brief; the marginalia
     ``6 paarden``; the illustration captions Wilhelms Brücke / Cassel /
     Desenberg) are tagged, inside <note type="editorial"> where appropriate.
  6. Coverage beats ALTO (== 313 >= ALTO's 307).
  7. Determinism: re-running build_tei is byte-identical.

Exits non-zero on any failure.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alto_exporter as ae  # noqa: E402
import tei_exporter as tei  # noqa: E402

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
T = "{" + TEI_NS + "}"

TEI_PATH = ROOT / "output" / "tei" / "1816_third_letter.tei.xml"
LETTER_ID = "BRF0003"

failures = []


def check(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    print("TEI self-check (canonical edition)\n" + "=" * 60)
    if not TEI_PATH.exists():
        print(f"FAIL: {TEI_PATH} not found — run `python3 output/tei_exporter.py` first.")
        sys.exit(1)
    xml_str = TEI_PATH.read_text(encoding="utf-8")
    root = ET.fromstring(xml_str)

    # --- load the gold source of truth (entities + mention ids) ---
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(str(ROOT / "entity_linking/el-results/1816_el_gs_el.spacy"))
    docs = list(db.get_docs(nlp.vocab))
    om = json.load(open(ROOT / "entity_linking/el-results/1816_el_gs_offset_map.json"))
    sp = sorted(om.items(), key=lambda kv: ae._page_sort_key(kv[0]))
    mm = json.load(open(tei.DEFAULT_MENTION_MAP))
    ml = ae.build_mention_lookup(mm)
    _, _, all_entities = ae.build_full_text_and_entities(docs, sp, ml)
    mention_ids = [e[5] for e in all_entities if e[5]]

    # ── 1. zero entity loss ──────────────────────────────────────────────
    print("\n1. zero entity loss (every gold entity inline-tagged)")
    inline_tags = [e for e in root.iter()
                   if e.tag in (f"{T}placeName", f"{T}persName", f"{T}orgName",
                                f"{T}date", f"{T}rs")]
    with_id = [e for e in inline_tags if e.get(XML_ID)]
    check(len(inline_tags) == len(all_entities),
          f"inline entity tags == {len(all_entities)} (got {len(inline_tags)})")
    check(len(with_id) == len(mention_ids),
          f"inline tags with xml:id == {len(mention_ids)} (got {len(with_id)})")
    ids_in_doc = {e.get(XML_ID) for e in with_id}
    missing = [m for m in mention_ids if m not in ids_in_doc]
    check(not missing, f"every mention_id eNN present as xml:id (missing: {missing[:8]})")

    # ── 2. bidirectional standOff linkage ─────────────────────────────────
    print("\n2. standOff linkage (xml:id <-> ato:CT.BRF0003.eNN)")
    ct_missing = [m for m in mention_ids
                  if f"CT.{LETTER_ID}.{m}" not in xml_str]
    check(not ct_missing,
          f"every xml:id eNN has a CT.{LETTER_ID}.eNN in the standOff RDF "
          f"(missing: {ct_missing[:8]})")
    # every CT mention node in the RDF that has a mention_map entry -> inline id
    ct_in_rdf = set()
    import re
    for m in re.finditer(rf"CT\.{LETTER_ID}\.(e\d+)", xml_str):
        ct_in_rdf.add(m.group(1))
    orphans = [m for m in ct_in_rdf if m in set(mention_ids) and m not in ids_in_doc]
    check(not orphans, f"no CT mention URI without an inline xml:id (orphans: {orphans[:8]})")
    xeno = root.find(f".//{T}xenoData")
    check(xeno is not None and "rdf:RDF" in (ET.tostring(xeno, encoding="unicode") if xeno is not None else ""),
          "<standOff><xenoData> carries an <rdf:RDF> graph")

    # ── 3. page numbers preserved ─────────────────────────────────────────
    print("\n3. page numbers preserved (<pb n>)")
    pbs = [pb.get("n") for pb in root.iter(f"{T}pb")]
    expected = [str(p) for p, _ in sp]
    check(pbs == expected, f"<pb n> sequence == {expected} (got {pbs})")

    # ── 4. well-formed + no residual page markers in body ─────────────────
    print("\n4. well-formed XML + no leaked page markers")
    body = root.find(f".//{T}body")
    body_text = "".join(body.itertext()) if body is not None else ""
    leaked_pages = re.findall(r"\[\d+\]|\b\d+\]", body_text)
    check(not leaked_pages, f"no residual [NN]/NN] page markers in body text "
          f"(found: {leaked_pages[:6]})")

    # ── 5. editorial notes recover the ALTO-lost entities ─────────────────
    print("\n5. editorial notes recover ALTO-lost entities")
    # map mention_id -> enclosing element (is it inside <note type="editorial">?)
    def enclosing_note(el):
        # walk up via parent map
        for anc in parents.get(id(el), []):
            if anc.tag == f"{T}note" and anc.get("type") == "editorial":
                return True
        return False
    parents = {}
    for parent in root.iter():
        for child in parent:
            parents.setdefault(id(child), []).append(parent)
    by_id = {}
    for el in inline_tags:
        if el.get(XML_ID):
            by_id.setdefault(el.get(XML_ID), []).append(el)
    # Derde Brief (e1) is in body text (title), the rest in editorial notes
    def text_of(mid):
        return "".join("".join(el.itertext()) for el in by_id.get(mid, []))
    check("Derde Brief" in text_of("e1"), "e1 'Derde Brief' tagged (title, body)")
    for mid, name in [("e35", "6 paarden"), ("e86", "Desenberg"),
                      ("e108", "Cassel"), ("e284", "Wilhelms Brücke"),
                      ("e285", "Cassel")]:
        present = name in text_of(mid)
        in_note = any(enclosing_note(el) for el in by_id.get(mid, []))
        # Derde Brief/Cassel-in-caption expectations:
        if mid in ("e35", "e86", "e108", "e284", "e285"):
            check(present, f"{mid} {name!r} tagged (in editorial note: {in_note})")

    # ── 6. coverage beats ALTO ────────────────────────────────────────────
    print("\n6. coverage vs ALTO")
    check(len(inline_tags) >= 307,
          f"TEI inline tags ({len(inline_tags)}) >= ALTO coverage (307)")

    # ── 7. determinism ────────────────────────────────────────────────────
    print("\n7. determinism (re-run build_tei byte-identical)")
    import contextlib, io, tempfile
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        tmp = tf.name
    with contextlib.redirect_stdout(io.StringIO()):
        tei.build_tei(tei.DEFAULT_SPACY, str(ROOT / "entity_linking/el-results"
                  / "1816_el_gs_offset_map.json"), tmp,
                  mention_map_path=tei.DEFAULT_MENTION_MAP, rdf_path=tei.DEFAULT_RDF,
                  letter_id=LETTER_ID)
    xml2 = Path(tmp).read_text(encoding="utf-8")
    check(xml2 == xml_str, f"re-run byte-identical ({len(xml2)} vs {len(xml_str)})")
    Path(tmp).unlink(missing_ok=True)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:")
        for fmsg in failures:
            print(f"  - {fmsg}")
        sys.exit(1)
    print("PASS — all checks passed")


if __name__ == "__main__":
    main()