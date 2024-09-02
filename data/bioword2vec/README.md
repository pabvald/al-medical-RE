# Bioword2vec embeddings

The Bioword2vec pre-trained word embeddings can be obtained from the GitHub repository [BioWordVec: Improving Biomedical Word Embeddings with Subowrd Information and MeSH](https://github.com/ncbi-nlp/BioWordVec).  Execute the following command to download the file: 

```bash
wget -O bio_embedding_extrinsic https://figshare.com/ndownloader/files/12551780
```

Once the binary file `bio_embedding_extrinsic` has been downloaded, it is necessary to transform it into .txt. In order to do so, execute the following Python code:

```python
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('path/to/bio_embedding_extrinsic', binary=True)
model.save_word2vec_format('path/to/bio_embedding_extrinsic.txt', binary=False)
```

The `bio_embedding_extrinsic.txt` file should be stored in this directory.
