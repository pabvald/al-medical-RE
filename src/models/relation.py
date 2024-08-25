# Base Dependencies
# ------------------
import logging
import re

from abc import ABC
from typing import Optional, Tuple, List, Set, Optional
from xml.etree import ElementTree

# Package Dependencies
# ------------------
from .entity import Entity
from .document import Document

# Local Dependencies
# ------------------
from nlp_pipeline import get_pipeline
from utils import clean_text_ddi, clean_text_n2c2

# 3-rd Party Dependencies
# -----------------------
from spacy.tokens import Doc
from spacy.language import Language

# Constants
# ---------
from constants import DDI_ALL_TYPES


logger = logging.getLogger(__name__)


class Relation(ABC):
    """
    Relation
    """

    NLP: Language = None

    def __init__(
        self,
        doc_id: str,
        type: str,
        entity1: Entity,
        entity2: Entity,
        left_context: str,
        middle_context: str,
        right_context: str,
        middle_entities: List[Entity],
        label: Optional[int] = None,
    ):
        """
        Args:
            doc_id (str): the identifier of the document the relation belongs to.
            type (str): type of relation (e.g. for the DDI Extraction corpus: mechanism, interaction, etc.)
            entity1 (Entity): first target entity
            entity2 (Entity): second target entity
            left_context (str): the text in the relation's sentence before both target entities
            middle_context (str): the text in the relation's sentence in between the target entities
            right_context (str): the text in the relation's sentence after the target entities
            middle_entities (List[Entity]): other entities present in between the target entitites
            label (Optional[int], optional): classification label. In the case of the n2c2 corpus, this label will be 0 (negative)
            or 1 (positive). In the case of the DDI corpus, this label will be 0 (NO-REL), 1, 2, 3 or 4 depending on the relation
            type. Defaults to None.
        """
        self.doc_id = doc_id
        self.type = type
        self.entity1 = entity1
        self.entity2 = entity2
        self.left_context = left_context
        self.middle_context = middle_context
        self.right_context = right_context
        self.middle_entities = middle_entities
        self.label = label

        if self.entity1.start > self.entity2.start:
            entity = self.entity1
            self.entity1 = self.entity2
            self.entity2 = entity

    def __str__(self) -> str:
        return "Relation(id={}, type={}, e1={}, e2={})".format(
            self.id, self.type, self.entity1.id, self.entity2.id
        )

    def __len__(self) -> str:
        return len(self.text)

    # Properties
    # ----------
    @property
    def id(self) -> str:
        """Unique idenfitifer"""
        e1, e2 = self._ordered_entities
        return "{}-{}-{}".format(self.doc_id, e1.id, e2.id)

    @property
    def text(self) -> str:
        """Textual representation of the relation"""
        s = "{left_context} {e1} {middle_context} {e2} {right_context}".format(
            left_context=self.left_context,
            e1=self.entity1.text,
            middle_context=self.middle_context,
            e2=self.entity2.text,
            right_context=self.right_context,
        )
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    @property
    def tokens(self) -> Doc:
        """Obtains the tokenized text of the relation's text

        Returns:
            Doc: processed text through Spacy's pipeline
        """

        logger.warning(
            "Use RelationCollection.relations_tokens if you want to tokenize several relations"
        )
        return self.tokenize(self.text)

    @property
    def left_tokens(self) -> Doc:
        """Obtains the tokenized text of the relation's left context"""
        return self.tokenize(self.left_context.strip())

    @property
    def middle_tokens(self) -> Doc:
        """Obtains the tokenized text of the relation's middle context"""
        return self.tokenize(self.middle_context.strip())

    @property
    def right_tokens(self) -> Doc:
        """Obtains the tokenized text of the relation's right context"""
        return self.tokenize(self.right_context.strip())

    @property
    def _ordered_entities(self) -> Tuple[Entity, Entity]:
        """Provides the order in which tne target entities appear in the text"""
        if self.entity1.start > self.entity2.start:
            entity2 = self.entity1
            entity1 = self.entity2
        else:
            entity1 = self.entity1
            entity2 = self.entity2

        return entity1, entity2

    # Class methods
    # --------------
    @classmethod
    def set_nlp(cls, nlp):
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

    # Instance methods
    # ---------------
    def todict(self) -> dict:
        """Dictionary representation of a Relation"""
        return {
            "doc_id": self.doc_id,
            "type": self.type,
            "entity1": self.entity1.todict(),
            "entity2": self.entity2.todict(),
            "left_context": self.left_context,
            "middle_context": self.middle_context,
            "right_context": self.right_context,
            "middle_entities": [ent.todict() for ent in self.middle_entities],
            "label": self.label,
        }

    # Static methods
    # --------------
    @staticmethod
    def get_middle_entities(
        entities: List[Entity], entity1: Entity, entity2: Entity
    ) -> Tuple[bool, List[Entity]]:
        """Obtains the entities that are in between

        Args:
            entities (List[Entity]): list of entities
            entity1 (Entity): first target entity
            entity2 (Entity): second target entity
        Returns:
            bool, List[Entity]: whether the target entities overlap, the entities
            in between the target entities
        """
        overlap = False
        start: int = min(entity1.end, entity2.end)
        end: int = max(entity1.start, entity2.start)

        if not start <= end:
            overlap = True

        f = lambda ent: ent.start >= start and ent.end <= end
        middle_entities: List[Entity] = list(filter(f, entities))

        return overlap, middle_entities


class RelationN2C2(Relation):
    """
    RelationN2C2

    Relation subclass specific for the 2018 n2c2 challenge
    """

    @property
    def _ordered_entities(self) -> Tuple[Entity, Entity]:
        """Provides n2c2 order of the entities, where the attribute goes first and
        the drug in second place"""

        if self.entity1.type == "Drug":
            entity2 = self.entity1
            entity1 = self.entity2
        else:
            entity1 = self.entity1
            entity2 = self.entity2

        return entity1, entity2

    @staticmethod
    def generate_relations_n2c2(
        doc: Document,
        entities: List[Entity],
        gt_relations: Set[str],
        is_test: bool = False,
    ) -> List["Relation"]:
        """
        Generates the relations of a n2c2 document. Method: generate all
        valid combinations of entity pairs within the same sentence.

        Args:
            doc: n2c2 clinical document
            entities: list of all the entities contained in the document
            gt_relations: ground_truth relations
            is_test: the document belongs to the test split
        Returns:
            List[Relation]: all possible relation in the n2c2 corpus' document
        """
        if is_test:
            return RelationN2C2._generate_relations_n2c2_test(
                doc, entities, gt_relations
            )
        else:
            return RelationN2C2._generate_relations_n2c2_train(
                doc, entities, gt_relations
            )

    @staticmethod
    def _generate_relations_n2c2_test(
        doc: Document, entities: List[Entity], gt_relations: Set[str]
    ) -> List["Relation"]:
        """
        Generates the relations of a n2c2 document of the test split.

        Args:
            doc: n2c2 clinical document
            entities: list of all the entities contained in the document
            gt_relations: ground_truth relations
        
        Returns:
            List[Relation]: all relations for the test set present in a particular document
        """
        relations: List["Relation"] = RelationN2C2._generate_relations_n2c2_train(
            doc, entities, gt_relations
        )
        ids = []
        for r in relations:
            e1, e2 = r._ordered_entities
            ids.append("{}-{}".format(e1.id, e2.id))

        for gt_relation in gt_relations:
            if gt_relation in ids:
                continue

            entity1 = None
            entity2 = None
            entity1_id, entity2_id = gt_relation.split("-")

            # get entities
            for ent in entities:
                if ent.id == entity1_id:
                    entity1 = ent
                elif ent.id == entity2_id:
                    entity2 = ent

                if entity1 is not None and entity2 is not None:
                    break

            # consider only the entities within the same sentence
            overlap, middle_entities = Relation.get_middle_entities(
                entities, entity1, entity2
            )
            if not overlap:
                (
                    left_context,
                    middle_context,
                    right_context,
                ) = RelationN2C2.get_context(doc, entity1, entity2)

                attr_type = entity1.type if entity1.type != "Drug" else entity2.type

                relations.append(
                    # create relation
                    RelationN2C2(
                        doc_id=doc.doc_id,
                        type="{}-Drug".format(attr_type),
                        entity1=entity1,
                        entity2=entity2,
                        left_context=left_context,
                        middle_context=middle_context,
                        right_context=right_context,
                        middle_entities=middle_entities,
                        label=int(1),
                    )
                )
            # end if
        # end for

        return relations

    @staticmethod
    def _generate_relations_n2c2_train(
        doc: Document, entities: List[Entity], gt_relations: Set[str]
    ) -> List["Relation"]:
        """
        Generates the relations of a n2c2 document of the train split.

        Args:
            doc: n2c2 clinical document
            entities: list of all the entities contained in the document
            gt_relations: ground_truth relations
        
        Returns:
            List[Relation]: all relations for the train set present in a particular document
        """
        
        ent_index = 0
        relations: List["Relation"] = []

        for sent_index, sentence in enumerate(doc.sentences):
            drugs = []
            attrs = []

            # select the entities contained in the current sentence
            i = ent_index
            while i < len(entities):
                if entities[i].start >= sentence.end:
                    break
                if entities[i].type == "Drug":
                    drugs.append(entities[i])
                else:
                    attrs.append(entities[i])

                i += 1
            # end while

            ent_index = i

            # generate all valid combinations of relations
            for drug in drugs:
                for attr in attrs:
                    rel_id = "{}-{}".format(attr.id, drug.id)
                    label = int(rel_id in gt_relations)

                    # consider only the entities within the same sentence
                    overlap, middle_entities = Relation.get_middle_entities(
                        drugs + attrs, drug, attr
                    )
                    if not overlap:
                        (
                            left_context,
                            middle_context,
                            right_context,
                        ) = RelationN2C2.get_context(
                            doc, drug, attr, sent_index, sent_index
                        )

                        relations.append(
                            # create relation
                            RelationN2C2(
                                doc_id=doc.doc_id,
                                type="{}-Drug".format(attr.type),
                                entity1=drug,
                                entity2=attr,
                                left_context=left_context,
                                middle_context=middle_context,
                                right_context=right_context,
                                middle_entities=middle_entities,
                                label=label,
                            )
                        )
                    # end if
            # end for

            # ignore the rest of the sentences if there are no more entities
            if ent_index >= len(entities):
                break

        return relations

    @staticmethod
    def get_context(
        document: Document,
        entity1: Entity,
        entity2: Entity,
        s1: Optional[int] = None,
        s2: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """Obtains the left, middle and right context of a pair of entities

        Args:
            document (Document): clinical document
            entity1 (Entity): first target entity
            entity2 (Entity): second target entity
            s1 (Optional[int], optional): sentence containing the first target entity. Defaults to None.
            s2 (Optional[int], optional): sentence containing the second target entiy. Defaults to None.

        Returns:
            Tuple[str, str, str]: left, middle and left context of the target entities
        """
        if entity1.start > entity2.start:
            entity = entity1
            entity1 = entity2
            entity2 = entity

        if s1 is None:
            s1 = document.find_sentence(entity1.start)

        if s2 is None:
            s2 = document.find_sentence(entity2.end)

        text = document.to_txt(s1, s2 + 1)
        sentence1 = document.sentences[s1]
        sentence2 = document.sentences[s2]

        offset = sentence1.start

        # left context
        if entity1.start > sentence1.start:
            left_context = text[0 : (entity1.start - offset)]
            left_context = clean_text_n2c2(left_context)
        else:
            left_context = ""

        # middle context
        middle_context = text[(entity1.end - offset) : (entity2.start - offset)]
        middle_context = clean_text_n2c2(middle_context)

        # right context
        if entity2.end < sentence2.end:
            right_context = text[(entity2.end - offset) :]
            right_context = clean_text_n2c2(right_context)
        else:
            right_context = ""

        return left_context, middle_context, right_context


class RelationDDI(Relation):
    """
    RelationDDI

    Relation subclass specific for the 2013 DDI Extraction challenge
    """

    @staticmethod
    def generate_relations_ddi(xml_tree: ElementTree) -> List["Relation"]:
        """Generates all relations present in the DDI document

        Args: 
            xml_tree (ElementTree): a tree containing the structured content of an XML file
        Returns:
            List[Relation]: relations present in the DDI corpus' document
        """

        relations: List["Relation"] = []
        document = xml_tree.getroot()
        doc_id: str = document.attrib["id"]

        for sentence in document:
            entities: List[Entity] = []
            for child in sentence:
                if child.tag == "entity":
                    annotation = child.attrib
                    entities.append(Entity.from_ddi_annotation(doc_id, annotation))

            for child in sentence:
                if child.tag == "pair":
                    annotation = child.attrib

                    if annotation["ddi"] == "true":
                        label: int = DDI_ALL_TYPES.index(annotation["type"].upper())
                    else:
                        label: int = 0

                    # rel_id = annotation["id"]
                    entity1: Entity = entities[int(annotation["e1"].split(".")[-1][1:])]
                    entity2: Entity = entities[int(annotation["e2"].split(".")[-1][1:])]
                    overlap, middle_entities = Relation.get_middle_entities(
                        entities, entity1, entity2
                    )
                    text = sentence.attrib["text"]
                    type: str = DDI_ALL_TYPES[label]
                    left_context: str = clean_text_ddi(text[: entity1.start])
                    middle_context: str = clean_text_ddi(
                        text[entity1.end : entity2.start]
                    )
                    right_context: str = clean_text_ddi(text[entity2.end :])

                    if not overlap:
                        relations.append(
                            RelationDDI(
                                doc_id=doc_id,
                                type=type,
                                entity1=entity1,
                                entity2=entity2,
                                left_context=left_context,
                                middle_context=middle_context,
                                right_context=right_context,
                                middle_entities=middle_entities,
                                label=label,
                            )
                        )

        return relations
