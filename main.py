from pyabsa import AspectPolarityClassification as APC, available_checkpoints
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import DatasetItem
from pyabsa import ModelSaveOption, DeviceTypeOption
import warnings
import json




warnings.filterwarnings("ignore")
# dataset = ATEPC.ATEPCDatasetList.Laptop14
# config = (
#     ATEPC.ATEPCConfigManager.get_atepc_config_english()
# )  # this config contains 'pretrained_bert', it is based on pretrained models
# config.model = ATEPC.ATEPCModelList.BERT_BASE_ATEPC
#
# ckpts = (
#     available_checkpoints()
# )  # This will show the available checkpoints and their detailed information
# find a suitable checkpoint and use the name:

# aspect_extractor = ATEPC.AspectExtractor(
#     checkpoint="english"
# )

# Extract aspects and save in JSON file
# res = aspect_extractor.extract_aspect(
#     inference_source=ATEPC.ATEPCDatasetList.Laptop14,
#     save_result=True,
#     pred_sentiment=True
# )

# Loads aspect data fron JSON file
res = json.load(open('Aspect Term Extraction and Polarity Classification.FAST_LCF_ATEPC.result.json'))

aspects = {
    "aspect": {
        "sentences": [""],
        "occurrences": 0
    },

}


for result in res:
    for i in range(len(result['aspect'])):
        # Check if aspect is repeated
        if result['aspect'][i] in aspects:
            # Check if aspect sentence is repeated
            if result['sentence'] not in aspects[result['aspect'][i]]['sentences']:
                aspects[result['aspect'][i]]['occurrences'] = aspects[result['aspect'][i]]['occurrences'] + 1
                aspects[result['aspect'][i]]['sentences'].append(result['sentence'])
                match result['sentiment'][i]:
                    case 'Positive': senScore = 1
                    case 'Negative': senScore = 0
                    case 'Neutral': senScore = 0.5
                    # Break code :)
                    case _: print("No sentiment")

                aspects[result['aspect'][i]]['sentiment'] += senScore
        else:
            match result['sentiment'][i]:
                case 'Positive':
                    senScore = 1
                case 'Negative':
                    senScore = 0
                case 'Neutral':
                    senScore = 0.5
                # Break code :)
                case _:
                    print("No sentiment")
            spe = {'sentences': [result['sentence']], 'occurrences': 1, 'sentiment': senScore}
            aspects[result['aspect'][i]] = spe

# Sort by descending order of highest occurences in review dataset
sorted_aspects = sorted(aspects.items(), key=lambda x: x[1]['occurrences'], reverse=True)

for aspect, aspect_info in sorted_aspects:
    if aspect_info['occurrences'] > 1:
        print(f"{aspect} appeared {aspect_info['occurrences']} times with an overall sentiment of {aspect_info['sentiment']/aspect_info['occurrences']}")

