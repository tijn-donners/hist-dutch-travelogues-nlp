import json
from pathlib import Path
import spacy
from spacy_llm.util import assemble
from spacy.tokens import DocBin
from spacy.util import load_config
import logging
import regex as re

# make sure ollama is serving to http://localhost:11434
# in order to use ollama's cloud models you have to create an account
# login via the terminal with `ollama signin`
# local models can be run without logging in

SCRIPT_DIR = Path(__file__).resolve().parent

# show prompt and response for the LLM in terminal
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("spacy_llm").setLevel(logging.DEBUG)

# load the .txt
with open(SCRIPT_DIR.parent / "data/1816_third_letter.txt", 'r') as file:
    letter = file.read()

# split on [number] markers but keep them in the page text so character offsets
# align with Recogito annotations on the full original text
marker_matches = list(re.finditer(r'\[(\d+)\]', letter))

# dict: pagenumber -> (full_text_offset, page_text_including_marker)
pages = {}
for i, m in enumerate(marker_matches):
    page_num = int(m.group(1))
    page_start = m.start()
    next_start = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(letter)
    page_text = letter[page_start:next_start]
    if page_text.strip():
        pages[page_num] = (page_start, page_text)

# get the model name from the config file, important for naming the .spacy files later
config_path = str(SCRIPT_DIR / "ner_config.cfg")
config = load_config(config_path)
model_name = config["components"]["llm"]["model"]["name"]
print(f"Starting LLM NER in SpaCy framework with: {model_name}\n")

# assemble the Ollama LLM config as spacy nlp object (you can change the LLM used in the config.cfg)
nlp = assemble(config_path)
print('Pipeline loaded:', nlp.pipe_names)

# collect all docs for merging later
all_docs = []
offset_map = {}  # pagenumber -> full-text character offset

# create DocBin / Run NER per page
for pagenumber, (page_offset, page_text) in sorted(pages.items()):
    doc = nlp(page_text)
    offset_map[pagenumber] = page_offset
    print(f"SpaCy DocBin created for page {pagenumber} (full-text offset: {page_offset})")

    # print the entities that were found
    if doc.ents != []:
        for ent in doc.ents:
            print(ent.text, ent.label_)
    else:
            print("There were no entities recognised")

    all_docs.append(doc)

# merge all docs into one DocBin and save
merged_docbin = DocBin(docs=all_docs)
merged_docbin.to_disk(SCRIPT_DIR / "ner-results" / f"1816_all_pages_{model_name}.spacy")
print(f"Merged DocBin saved to: ner-results/1816_all_pages_{model_name}.spacy")

# save offset map for aligning with full-text annotations
offset_map_path = SCRIPT_DIR / "ner-results" / f"1816_offset_map_{model_name}.json"
with open(offset_map_path, 'w') as f:
    json.dump(offset_map, f)
print(f"Offset map saved to: {offset_map_path}")