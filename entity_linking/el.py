"""Entity Linking pipeline for 19th-century Dutch travelogue NER results.

Three-stage LELA-inspired architecture:
  1. Candidate Generation (Wikidata + GeoNames APIs)
  2. Candidate Reranking (cross-encoder or heuristic)
  3. LLM Selection (Ollama with domain-aware Dutch prompt)

Loads .spacy files from NER, runs EL, and saves results with kb_ids.
"""

import os
from pathlib import Path

import spacy
from dotenv import load_dotenv
from spacy.tokens import DocBin

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

from el_candidates import generate_candidates, clear_cache
from el_reranker import rerank_candidates
from el_selector import select_candidate

# ── Configuration ──────────────────────────────────────────────────────────

SPACY_FILE = str(ROOT_DIR / "ner" / "ner-results" / "1816_all_pages_deepseek-v4-pro.spacy")
MODEL_NAME = "gemma4:31b-cloud"
GEONAMES_USERNAME = os.environ.get("GEONAMES_USERNAME")
TOP_K_RERANK = 3

if os.environ.get("OLLAMA_API_KEY"):
    OLLAMA_URL = "https://ollama.com"
    OLLAMA_HEADERS = {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
else:
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_HEADERS = None
def extract_context(doc_text: str, start_char: int, end_char: int,
                    window: int = 500) -> str:
    """Extract surrounding text window around a character-offset entity span.

    Args:
        doc_text: Full page text.
        start_char: Start character offset of the entity.
        end_char: End character offset of the entity.
        window: Number of characters to include on each side (default 500).

    Returns:
        Context string with newlines replaced by spaces.
    """
    ctx_start = max(0, start_char - window)
    ctx_end = min(len(doc_text), end_char + window)
    context = doc_text[ctx_start:ctx_end].replace('\n', ' ')
    return context


def link_entities(
    spacy_file: str,
    model_name: str = "gemma4:31b-cloud",
    geonames_username: str | None = None,
    top_k_rerank: int = 3,
    ollama_url: str = "http://localhost:11434",
    ollama_headers: dict | None = None,
) -> list:
    """Run the full EL pipeline on a .spacy file.

    Args:
        spacy_file: Path to the .spacy DocBin file with NER annotations.
        model_name: Ollama model name for Stage 2 reranking and Stage 3 selection.
        geonames_username: Optional GeoNames API username for Stage 1.
        top_k_rerank: Number of candidates to retain after reranking.
        ollama_url: Ollama server base URL.
        ollama_headers: Optional auth headers for cloud API (Bearer token).

    Returns:
        List of updated spaCy Docs with kb_ids set on entities.
    """
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_file)
    docs = list(db.get_docs(nlp.vocab))

    total_ents = 0
    linked = 0

    for doc in docs:
        for ent in doc.ents:
            total_ents += 1
            entity_text = ent.text
            context = extract_context(doc.text, ent.start_char, ent.end_char)

            # Stage 1: Candidate Generation
            candidates = generate_candidates(
                entity_text,
                ent.label_,
                context=context,
                geonames_username=geonames_username,
                ollama_url=ollama_url,
                model_name=model_name,
                ollama_headers=ollama_headers,
            )

            if not candidates:
                continue

            # Stage 2: Candidate Reranking
            reranked = rerank_candidates(
                entity_text,
                context,
                candidates,
                top_k=top_k_rerank,
                ollama_url=ollama_url,
                model_name=model_name,
                ollama_headers=ollama_headers,
            )

            # Stage 3: LLM Selection
            selected_id = select_candidate(
                entity_text,
                context,
                reranked,
                model_name=model_name,
                ollama_url=ollama_url,
                ollama_headers=ollama_headers,
            )

            if selected_id != "NIL":
                ent.kb_id_ = selected_id
                for token in ent:
                    token.ent_kb_id_ = selected_id
                linked += 1
                selected_label = next(
                    (c["label"] for c in reranked if c["id"] == selected_id),
                    "",
                )
                print(f"  -> {selected_id} ({selected_label})")
            else:
                print(f"  -> NIL (no match)")

    print(f"\nLinked {linked}/{total_ents} entities ({linked/total_ents*100:.1f}%)"
          if total_ents else "\nNo entities found.")
    return docs


# ── Run ────────────────────────────────────────────────────────────────────

def main():
    """Run the full Entity Linking pipeline and save results.

    Prints configuration, clears caches, runs the three-stage pipeline on the
    configured .spacy file, and writes the enriched DocBin to disk.
    """
    print(f"Loading: {SPACY_FILE}")
    print(f"LLM: {MODEL_NAME}")
    print(f"Ollama: {OLLAMA_URL} {'(cloud)' if OLLAMA_HEADERS else '(local)'}")
    print(f"GeoNames: {'enabled' if GEONAMES_USERNAME else 'disabled'}")
    print(f"Top-k rerank: {TOP_K_RERANK}")
    print()

    clear_cache()

    docs = link_entities(
        spacy_file=SPACY_FILE,
        model_name=MODEL_NAME,
        geonames_username=GEONAMES_USERNAME,
        top_k_rerank=TOP_K_RERANK,
        ollama_url=OLLAMA_URL,
        ollama_headers=OLLAMA_HEADERS,
    )

    output_path = SPACY_FILE.replace(".spacy", "_el.spacy")
    merged_docbin = DocBin(docs=docs)
    merged_docbin.to_disk(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
