# %%
import spacy
from spacy import displacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

sentence_nlp = nlp(
    "'Don Quixote' is a tragedy written by William Shakespeare sometime between 1599 and 1601."
)
html = displacy.render(sentence_nlp, style="ent", page=True, jupyter=False)

with open("ner_hamlet.html", "w") as fin:
    fin.write(html)

# %%
