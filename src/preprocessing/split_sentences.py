# coding: utf-8

# Base Dependencies
# -----------------
from tqdm import tqdm

# Local Dependencies
# ------------------
from models.document import Document
from utils import doc_id_n2c2, files_n2c2


def split_sentences(dataset: str):
    if dataset == "n2c2":
        split_sentences_n2c2()
    elif dataset == "ddi":
        print("Splitting ddi dataset in sentences not necessary. All done \n")
    else:
        raise ValueError("unsupported dataset '{}'".format(dataset))


def split_sentences_n2c2():
    """
    Splits the n2c2 records into sentences and cleans the text. The split documents
    are stored in the same folder as the original text files in json format.
    """

    print("\nSplitting n2c2 dataset in sentences: ")

    dataset = files_n2c2()

    for split, files in dataset.items():
        print("\n", split, ": ")

        for basefile in tqdm(files):
            # read text file
            with open(basefile + ".txt", "r", encoding="utf-8") as fin:
                text = fin.read()

            # create document with sentence split
            document = Document.from_txt_n2c2(doc_id_n2c2(basefile), text)

            # assert len(text) == len(document.to_txt())

            with open(basefile + ".json", "w+", encoding="utf-8") as fout:
                fout.write(document.toJSON())
