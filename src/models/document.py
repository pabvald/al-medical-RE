# Base Dependencies
# --------------------
import json
from typing import List, Optional

# Package Dependencies
# --------------------
from .sentence import Sentence

# Spacy Dependencies
# -------------------
from spacy.lang.en import English

# from PyRuSH import PyRuSHSentencizer


# Document Class
# --------------
class Document:
    """
    Document representation of a n2c2 record in sentences
    """

    SENTEZICER = None

    def __init__(self, doc_id: str, sentences: List[Sentence]):
        """
        Args:
            doc_id (str): document's id
            sentences (List[Sentence]): list of sentences that form the document
        """
        self.doc_id = doc_id
        self.sentences = sentences

        if self.SENTEZICER is None:
            self.__init_sentecizer()

    @classmethod
    def __init_sentecizer(cls):
        cls.SENTEZICER = English()
        cls.SENTEZICER.add_pipe("medspacy_pyrush")

    def find_sentence(self, x: int) -> int:
        """
        Obtains the index of the sentence that contains the xth character

        Args:
            x (int): start index

        Return:
            index of the sentence containing the xth character, -1 if not found

        Source: https://www.geeksforgeeks.org/python-program-for-binary-search/
        """
        low = 0
        high = len(self.sentences) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # If x is greater, ignore left half
            if self.sentences[mid].start < x and self.sentences[mid].end < x:
                low = mid + 1

            # If x is smaller, ignore right half
            elif self.sentences[mid].start > x and self.sentences[mid].end > x:
                high = mid - 1

            # means x is present at mid
            else:
                return mid

        # If we reach here, then the element was not present
        return -1

    def to_txt(
        self, start_sentence: Optional[int] = None, end_sentence: Optional[int] = None
    ) -> str:
        """
        Textual representation of a fragment of the document

        Args:
            start_sentence (int, optional): first sentence to be considered
            end_sentence (int, optional): last sentenced to be considered

        Returns:
            text (fragment) of the document
        """
        init = 0
        end = len(self.sentences)

        if start_sentence is not None:
            init = start_sentence

        if end_sentence is not None:
            end = end_sentence

        return "".join([s.text for s in self.sentences[init:end]])

    def toJSON(self):
        """JSON representation of the document"""
        sentences = []
        for s in self.sentences:
            sentences.append(s.todict())
        return json.dumps({"doc_id": self.doc_id, "sentences": sentences})

    @classmethod
    def from_txt_n2c2(cls, doc_id: str, text: str):
        """Creates a Document from a .txt file"""
        sentences = []
        doc = SENTEZICER(text)

        starts = [s[0].idx for s in doc.sents]

        for index, start in enumerate(starts[:-1]):
            end = starts[index + 1]
            sentences.append(
                Sentence(
                    id=index,
                    text=text[start:end],
                    start=start,
                    end=end,
                )
            )

        # last sentence
        start = starts[-1]
        end = len(text)
        sentences.append(
            Sentence(
                id=index,
                text=text[start:end],
                start=start,
                end=end,
            )
        )

        return cls(doc_id, sentences)

    @classmethod
    def from_json(cls, json_content: str):
        """Creates a Document from a JSON representation"""
        json_dict = json.loads(json_content)

        doc_id = json_dict["doc_id"]
        sentences = [Sentence(**s) for s in json_dict["sentences"]]

        return cls(doc_id, sentences)
