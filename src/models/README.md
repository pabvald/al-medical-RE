# Data Models

This directory contains the data models used in the project. The data models are used for preprocessing and feature generation. 

## Sentence and Document
The `Document` class represents a n2c2 record. It is modelled as collection of `Sentence`s. These two data models are only used for preprocessing of the n2c2 corpus.

## Entity

The `Entity` class represents an biomedical entity. It contains the following attributes:

 - `id`: The entity id.
 - `text`: The entity text.
 - `type`: The entity type.
 - `doc_id`: The document id.
 - `start`: The start index of the entity in the document.
 - `end`: The end index of the entity in the document.

## Relation

The `Relation` class represents a relation between two `Entity` instances. Its creation requires the following attributes:

 - `doc_id`: The document id.
 - `type`: The relation type.
 - `entity1`: The first `Entity` of the relation.
 - `entity2`: The second `Entity` of the relation.
 - `left_context`: The text of the relation to the left of the first entity.
 - `middle_context`: The text of the relation between the two entities.
 - `right_context`: The text of the relation to the right of the second entity.
 - `middle_entities`: The entities in the middle context.
 - `label`: The relation label.


## RelatioCollection

The `RelationCollection` class represents a collection of `Relation` instances. It is used for feature generation. It is created from a list of `Relation` instances. 

The train and test splits of each corpora give a `RelationCollection`. 