# Base Dependencies
# ------------------
import json
from typing import Dict


class Sentence:
    """Sentence of a document"""

    def __init__(self, id: int, text: str, start: int, end: int):
        self.id = id
        self.text = text
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return "Sentence(id={}, start={}, end={})".format(self.id, self.start, self.end)

    def todict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }

    def toJSON(self) -> str:
        """JSON representation of the Sentence"""
        return json.dumps(self.todict())
