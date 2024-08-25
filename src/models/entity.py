# Base Dependencies
# -----------------
from typing import List

# Local Dependencies 
# ------------------
from nlp_pipeline import get_pipeline

# 3rd-Party Dependencies
# ----------------------
from spacy.language import Language
from spacy.tokens import Doc


class Entity:
    """
    Entity

    Representation of a biomedical entity.
    """

    # Spacy's pipeline
    NLP: Language = None

    def __init__(
        self, id: str, text: str, type: str, doc_id: str, start: int, end: int
    ):
        """
        Args:
            id (str): identifier
            text (str): text
            type (str): entity type
            doc_id (str): the identifier of the document the entity belongs to
            start (int): start character in the sentence
            end (int): end character in the sentence
        """
        self.id = id
        self.type = type.strip()
        self.text = text.strip()
        self.doc_id = doc_id
        self.start = start
        self.end = end
        self._tokens = None

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return "Entity(id: {}, type: {}, text: {}, start: {}, end: {})".format(
            self.id, self.type, self.text, self.start, self.end
        )

    @property
    def uid(self) -> str:
        """Unique idenfitifer"""
        return "{}-{}".format(self.doc_id, self.id)

    @property
    def tokens(self) -> Doc:
        """Obtains the tokenized text of the entity's text

        Returns:
            Doc: processed text through Spacy's pipeline
        """
        if self._tokens is None:
            self._tokens = Entity.tokenize(self.text)
        return self._tokens

    # Class Methods
    # -------------
    @classmethod
    def set_nlp(cls, nlp: Language):
        """Sets the Entity Class' Spacy's pipeline

        Args:
            nlp (Language): pipeline
        """
        cls.NLP = nlp

    @classmethod
    def tokenize(cls, text: str, disable: List[str] = ["parser", "negex"]) -> Doc:
        """Tokenizes a text fragment with the configured Spacy's pipeline

        Args:
            text (str): text fragment
            disable (List[str], optional): pipes of the Spacy's pipeline to be disabled. Defaults to ["parser"].

        Returns:
            Doc: tokenized text
        """
        if cls.NLP is None:
            cls.NLP = get_pipeline()

        with cls.NLP.select_pipes(disable=disable):
            doc = cls.NLP(text)
        return doc

    @classmethod
    def from_n2c2_annotation(cls, doc_id: str, annotation: str) -> "Entity":
        """Creates an Entity instance from an n2c2 annotation line

        Args:
            doc_id (str): the identifier of the document the entity belongs to
            annotation (str): the entity description in the n2c2 corpus' format

        Returns:
            Entity: the annotated entity

        """
        id, definition, text = annotation.strip().split("\t")
        definition = definition.split()  # definition: entity type and location in text
        type = definition[0]
        start = int(definition[1])
        end = int(definition[-1])

        return cls(id, text, type, doc_id, start, end)

    @classmethod
    def from_ddi_annotation(cls, doc_id: str, annotation: dict) -> "Entity":
        """Creates an Entity instance from an ddi xml annotation

        Args:
            doc_id (str): the identifier of the document the entity belongs to
            annotation (dict): the entity description in the DDi Extraction Corpus' format

        Returns:
            Entity: the annotated entity
        """
        id = annotation["id"]
        type = annotation["type"].upper()
        text = annotation["text"]
        char_offset = annotation["charOffset"].split("-")
        start = int(char_offset[0])
        end = int(char_offset[-1]) + 1

        return cls(id, text, type, doc_id, start, end)

    # Instance methods
    # ----------------
    def todict(self) -> dict:
        """Dict representation of an entity

        Returns:
            dict: representation of the Entity
        """
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
        }
