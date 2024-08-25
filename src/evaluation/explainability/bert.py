# Base Dependencies
# -----------------
from pathlib import Path
from os.path import join

# Local Dependencies
# ------------------
from ml_models.bert import ClinicalBERTTokenizer

# 3rd-Party Dependencies
# ----------------------
import torch
from datasets import load_from_disk

from transformers import BertForSequenceClassification, AutoConfig
from bertviz import head_view, model_view

# Constants
# ---------
from constants import CHECKPOINTS_CACHE_DIR, DDI_HF_TEST_PATH

N_CHANGES = 50


def run():
    """This scripts stores the attention maps of the Clinical BERT model for the first 50 changes in the DDI test set."""
    init_model_path = Path(
        join(CHECKPOINTS_CACHE_DIR, "al", "bert", "ddi", "model_5.ck")
    )
    end_model_path = Path(
        join(CHECKPOINTS_CACHE_DIR, "al", "bert", "ddi", "model_6.ck")
    )

    head_views_output_folder = Path(
        join("results", "ddi", "bert", "interpretability", "head_views")
    )
    model_views_output_folder = Path(
        join("results", "ddi", "bert", "interpretability", "model_views")
    )

    # load dataset and tokenize
    tokenizer = ClinicalBERTTokenizer()
    test_dataset = load_from_disk(Path(join(DDI_HF_TEST_PATH, "bert")))

    sentences = test_dataset["sentence"]
    labels = test_dataset["label"]

    # load BERT models
    init_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=init_model_path
    )
    init_config.output_attentions = True

    end_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=end_model_path
    )
    end_config.output_attentions = True

    init_model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=init_model_path, config=init_config
    )
    end_model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=end_model_path, config=end_config
    )

    changes = []

    for index, (sentence, label) in enumerate(zip(sentences, labels)):
        if label > 0:
            inputs = tokenizer.encode(sentence, return_tensors="pt")
            init_outputs = init_model(inputs)
            end_outputs = end_model(inputs)

            init_y_pred = torch.argmax(init_outputs["logits"])
            end_y_pred = torch.argmax(end_outputs["logits"])

            if end_y_pred == label and init_y_pred != label:

                tokens = tokenizer.convert_ids_to_tokens(inputs[0])
                init_head_view = head_view(
                    init_outputs["attentions"], tokens, html_action="return"
                )
                init_model_view = model_view(
                    init_outputs["attentions"], tokens, html_action="return"
                )
                end_head_view = head_view(
                    end_outputs["attentions"], tokens, html_action="return"
                )
                end_model_view = model_view(
                    end_outputs["attentions"], tokens, html_action="return"
                )

                # Save the HTMLs object to file
                file_path = Path(
                    join(head_views_output_folder, str(index) + "_init.html")
                )
                with open(file_path, "w") as f:
                    f.write(init_head_view.data)

                file_path = Path(
                    join(head_views_output_folder, str(index) + "_end.html")
                )
                with open(file_path, "w") as f:
                    f.write(end_head_view.data)

                file_path = Path(
                    join(model_views_output_folder, str(index) + "_init.html")
                )
                with open(file_path, "w") as f:
                    f.write(init_model_view.data)

                file_path = Path(
                    join(model_views_output_folder, str(index) + "_end.html")
                )
                with open(file_path, "w") as f:
                    f.write(end_model_view.data)

                changes.append(
                    f"Index: {str(index)} Initial prediction: {init_y_pred} Final Prediction: {end_y_pred}"
                )
                if len(changes) == N_CHANGES:
                    break

    # save list of changes to file
    with open(
        join("results", "ddi", "bert", "interpretability", "changes.txt"), "w"
    ) as f:
        for item in changes:
            f.write(f"{item}\n")
