''' TODO ways to push this further
- LM - nlp
- OOV - nlp
- punctuation model - nlp
- train on synthetic data, scrape your own data
- beam search decoding
- rnnt version, AED version
- bigger model - get an intuition for speed vs accuracy
- more data - get an intuition for speed vs accuracy
'''

from comet_ml import Experiment
with open('../comet_api_key.txt') as f:
    lines = f.readlines()
    experiment = Experiment(
        api_key=lines[0],
        project_name="michaelliang-dev",
        workspace="assemblyai",
)

import datasets
import transformers
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from model import Wav2Vec2ForCTC
import json

# Dataset
# libri_train = datasets.load_from_disk('data/libri_train')
libri_eval = datasets.load_from_disk('data/libri_eval')
# cv_train = datasets.load_from_disk('data/cv_train')
giga_train = datasets.load_from_disk('data/giga_train')
train_dataset = datasets.concatenate_datasets([giga_train])

# Vocab
# def extract_all_chars(batch):
#     all_text = " ".join(batch["labels"])
#     vocab = list(set(all_text))
#     return {"vocab": [vocab], "all_text": [all_text]}

# vocabs = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=["labels", "input_values"])
# vocab_list = vocabs["vocab"][0]
# vocab_dict = {v:k for k, v in enumerate(vocab_list)}
# vocab_dict["|"] = vocab_dict[" "] # space token
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict) # model can deal with unknown tokens
# vocab_dict["[PAD]"] = len(vocab_dict) # blank token

with open('vocab_en.json') as f:
    vocab = json.load(f)

# Tokenizer
''' Decodes model outputs to text
'''
tokenizer = transformers.Wav2Vec2CTCTokenizer("./vocab_en.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Feature extractor
''' Encodes speech signal to model input format

Source code: https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31

Padding, formatting inputs, normalising to zero mean and unit variance

'''
feature_extractor = transformers.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

# Processor
''' Wrapper of tokenizer and feature extractor
'''
processor = transformers.Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Preprocess dataset
def prepare_dataset(batch):
    batch["input_values"] = processor(batch["input_values"], sampling_rate=16000).input_values[0] # feature extractor (1-1 map)

    with processor.as_target_processor():
        batch["labels"] = processor(batch["labels"]) # tokenizer (1-1 map)

    return batch
    
train_dataset = train_dataset.map(prepare_dataset, num_proc=16)
libri_eval = libri_eval.map(prepare_dataset, remove_columns=libri_eval.column_names, num_proc=16)

# Data Collator
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: transformers.Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]['input_ids']} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Metrics
wer_metric = datasets.load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id # replace all -100's with pad_token_id (29)

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer":wer}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)

model.freeze_feature_encoder()

training_args = transformers.TrainingArguments(
    output_dir='outputs/',
    group_by_length=False,
    per_device_train_batch_size=32,
    dataloader_num_workers=16,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True, # false for CPU
    gradient_checkpointing=True, 
    save_steps=5000,
    eval_steps=5000,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=10,
    no_cuda=False, # train on CPU 
)

trainer= transformers.Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=libri_eval,
    tokenizer=processor.feature_extractor,
)

# TRAIN
trainer.train()

# EVAL
# model = transformers.Wav2Vec2ForCTC.from_pretrained(
#     "outputs/w2v2-base-libri-100-10-epoch/checkpoint-25000",
#     ctc_loss_reduction="mean",
#     pad_token_id=processor.tokenizer.pad_token_id
# )

# model.eval()
# trainer.evaluate()