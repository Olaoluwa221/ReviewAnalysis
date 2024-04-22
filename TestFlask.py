import os
import random
import gradio as gr
import pandas as pd
import requests

from pyabsa import (
    download_all_available_datasets,
    AspectTermExtraction as ATEPC,
    TaskCodeOption,
)
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset

download_all_available_datasets()

dataset_items = {dataset.name: dataset for dataset in ATEPC.ATEPCDatasetList()}


def get_example(dataset):
    task = TaskCodeOption.Aspect_Polarity_Classification
    dataset_file = detect_infer_dataset(dataset_items[dataset], task)

    for fname in dataset_file:
        lines = []
        if isinstance(fname, str):
            fname = [fname]

        for f in fname:
            print("loading: {}".format(f))
            fin = open(f, "r", encoding="utf-8")
            lines.extend(fin.readlines())
            fin.close()
        for i in range(len(lines)):
            lines[i] = (
                lines[i][: lines[i].find("$LABEL$")]
                .replace("[B-ASP]", "")
                .replace("[E-ASP]", "")
                .strip()
            )
        return sorted(set(lines), key=lines.index)


dataset_dict = {
    dataset.name: get_example(dataset.name) for dataset in ATEPC.ATEPCDatasetList()
}
aspect_extractor = ATEPC.AspectExtractor(checkpoint="multilingual")


def perform_inference(text, dataset):
    if not text:
        text = dataset_dict[dataset][random.randint(0, len(dataset_dict[dataset]) - 1)]
        print(text)

    result = aspect_extractor.predict(text, pred_sentiment=True)

    result = pd.DataFrame(
        {
            "aspect": result["aspect"],
            "sentiment": result["sentiment"],
            # 'probability': result[0]['probs'],
            "confidence": [round(x, 4) for x in result["confidence"]],
            # "position": result["position"],
        }
    )
    return result, "{}".format(text)


demo = gr.Blocks()

with demo:
    gr.Markdown(
        "# <p align='center'>Product Reviews Analysis</p>"
    )
    output_dfs = []
    with gr.Row():
        with gr.Column():
            input_sentence = gr.Textbox(
                placeholder="Leave this box blank and choose a dataset will give you a random example...",
                label="Example:",
            )
            gr.Markdown(
                "You can find the datasets at [github.com/yangheng95/ABSADatasets](https://github.com/yangheng95/ABSADatasets/tree/v1.2/datasets/text_classification)"
            )
            dataset_ids = gr.Radio(
                choices=[dataset.name for dataset in ATEPC.ATEPCDatasetList()[:-1]],
                value="Laptop14",
                label="Datasets",
            )
            inference_button = gr.Button("Let's go!")


        with gr.Column():
            output_text = gr.TextArea(label="Example:")
            output_df = gr.DataFrame(label="Prediction Results:")
            output_dfs.append(output_df)

        inference_button.click(
            fn=perform_inference,
            inputs=[input_sentence, dataset_ids],
            outputs=[output_df, output_text],
        )

demo.launch()