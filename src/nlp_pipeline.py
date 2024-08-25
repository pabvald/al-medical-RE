# Base Dependencies
# -----------------
import logging
from typing import List 

# Spacy Dependencies
# ------------------
from negspacy.negation import Negex
from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens import Doc, Span

# Constants
# ---------
from constants import N2C2_ENTITY_TYPES, DDI_ENTITY_TYPES


# Spacy's pipeline
NLP: Language = None


# Auxiliar functions
# ------------------
def get_pipeline() -> Language:
    """Gets Spacy's pipeline, loading it if necessary.

    Returns:
        Language: Spacy's pipeline singleton
    """
    global NLP

    # load only once
    if NLP is None:
        logging.warning("Loading Spacy's pipeline...")

        # load Scispacy's pipeline
        NLP = spacy_load("en_core_sci_sm", exclude=["ner"])

        # add negation detection component
        ent_types = [t.upper() for t in N2C2_ENTITY_TYPES] + [
            t.upper() for t in DDI_ENTITY_TYPES
        ]
        NLP.add_pipe("negex", config={"ent_types": ent_types})

        logging.warning("Spacy loaded!")
    return NLP


def set_spacy_entities(
    relation: Doc, 
    left_tokens: Doc,
    entity1_tokens: Doc,
    entity1_type: str,
    middle_tokens: Doc,
    entity2_tokens: Doc,
    entity2_type: str, 
    right_tokens: Doc,
) -> List[Span]:
    """_summary_

    Args:
        relation (Doc): _description_
        left_tokens (Doc): _description_
        entity1_tokens (Doc): _description_
        entity1_type (str): _description_
        middle_tokens (Doc): _description_
        entity2_tokens (Doc): _description_
        entity2_type (str): _description_
        right_tokens (Doc): _description_

    Returns:
        List[Span]: _description_
    """

    begin_e1 = len(left_tokens)
    end_e1 = begin_e1 + len(entity1_tokens)

    begin_e2 = end_e1 + len(middle_tokens)
    end_e2 = begin_e2 + len(entity2_tokens)

    e1 = Span(relation, begin_e1, end_e1, label=entity1_type)
    e2 = Span(relation, begin_e2, end_e2, label=entity2_type)
    
    relation.ents = [e1, e2]
