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

# show prompt and response for the LLM in terminal
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("spacy_llm").setLevel(logging.DEBUG)

# load the .txt
with open("data/1816_third_letter.txt", 'r') as file:
    letter = file.read()

# split on [number] markers, keeping the number using a capture group 
# (right now it only works for 1816 letters where pages are split like this: [16] {page 16 text} [17]})
parts = re.split(r'\[(\d+)\]', letter)  

# make a dict with the pagenumbers (integers )and their texts
pages = {}
for i in range(1, len(parts), 2):
    page_num = int(parts[i])
    page_text = parts[i + 1].strip()
    if page_text:
        pages[page_num] = page_text

# get the model name from the config file, important for naming the .spacy files later
config = load_config("config.cfg")
model_name = config["components"]["llm"]["model"]["name"]
print(f"Starting LLM NER in SpaCy framework with: {model_name}\n")  

# assemble the Ollama LLM config as spacy nlp object (you can change the LLM used in the config.cfg)
nlp = assemble("config.cfg")
print('Pipeline loaded:', nlp.pipe_names)

# collect all docs for merging later
all_docs = []

# create DocBin / Run NER per page
for pagenumber in pages.keys():
    doc = nlp(pages[pagenumber])
    print(f"SpaCy DocBin created for page {pagenumber}")

    # print the entities that were found
    if doc.ents != []:
        for ent in doc.ents:
            print(ent.text, ent.label_)
    else:
            print("There were no entities recognised, probably due to a formatting problem in the LLM output")

    # save DocBin and NER results
    DocBin(docs=[doc]).to_disk(f"ner-results/1816_p{pagenumber}_{model_name}.spacy")
    print(f"ner-results/1816_p{pagenumber}_{model_name}.spacy")

    all_docs.append(doc)

# merge all docs into one DocBin and save
merged_docbin = DocBin(docs=all_docs)
merged_docbin.to_disk(f"ner-results/1816_all_pages_{model_name}.spacy")
print(f"Merged DocBin saved to: ner-results/1816_all_pages_{model_name}.spacy")









