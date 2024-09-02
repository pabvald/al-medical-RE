# DDI Corpus 

The DDI Extraction corpus has been downloaded from its official GitHub repository [DDICorpus](https://github.com/isegura/DDICorpus/blob/master/DDICorpus-2013.zip). 

The file `./ddi_corpus.zip` contains the train and test splits of the DDI corpus with the necessary directory structure to apply preprocessing steps. If you decide to download the corpus yourself, make sure to rename the folders to match the following directory structure 

```
interactive-relation-extraction/
    data/
        ddi/
            train/
                DrugBank/
                MedLine/

            test/
                ner/
                    DrugBank/
                    MedLine/
                re/
                    DrugBank/
                    MedLine/
        ...
    ...
```
where the `DrugBank/` and `MedLine/` subfolders contain the `.xml` files.

## References 

[1] María Herrero-Zazo, Isabel Segura-Bedmar, Paloma Martínez, Thierry Declerck, The DDI corpus: An annotated corpus with pharmacological substances and drug–drug interactions, Journal of Biomedical Informatics, Volume 46, Issue 5, October 2013, Pages 914-920, http://dx.doi.org/10.1016/j.jbi.2013.07.011. 

[2] Isabel Segura-Bedmar, Paloma Martínez, María Herrero Zazo, (2014). Lessons learnt from the DDIExtraction-2013 shared task, Journal of Biomedical Informatics, Vol.51, pp:152-164.
