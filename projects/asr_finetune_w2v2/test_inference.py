import torch
import torchaudio
import transformers
import pandas as pd
import jiwer
from tqdm import tqdm
from model import Wav2Vec2ForCTC

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

torch.set_printoptions(threshold=10_000)

tokenizer = transformers.Wav2Vec2CTCTokenizer(
    "./vocab_en.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)

model = transformers.Wav2Vec2ForCTC.from_pretrained("outputs/w2v2-base-libri-100").to(
    device
)
model.eval()

# Run through files - using our english test set
with open("test_tsv.txt") as f:
    test_tsvs = f.readlines()
    test_tsvs = [tsv.replace("\n", "") for tsv in test_tsvs]

for tsv in test_tsvs:
    print("testing", tsv)
    df = pd.read_csv(tsv, sep="\t")
    ground_truth_list = []
    hypothesis_list = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            path = row["paths"]
            # convert to audio array
            wav, sr = torchaudio.load(path)
            # upsample
            resample_rate = 16000
            resampler = torchaudio.transforms.Resample(
                sr, resample_rate, dtype=wav.dtype
            )
            resampled_waveform = resampler(wav)

            # chunk the audio up - hard cut every 30s
            chunks = resampled_waveform.split(30 * 16000, dim=1)

            predicted_sentence = ""
            word_offsets = []
            for chunk in chunks:
                # pred
                with torch.no_grad():
                    logits = model(chunk.to(device))[0][0]

                logits = torch.argmax(logits, dim=-1)

                preds = tokenizer.decode(logits, output_word_offsets=True)

                time_offset = model.config.inputs_to_logits_ratio / resample_rate

                word_offsets_chunk = [
                    {
                        "word": d["word"],
                        "start_time": round(d["start_offset"] * time_offset, 2),
                        "end_time": round(d["end_offset"] * time_offset, 2),
                    }
                    for d in preds.word_offsets
                ]
                word_offsets += word_offsets_chunk
                predicted_chunk = preds[0]
                predicted_sentence += predicted_chunk
            ground_truth_list.append(row["labels"])
            hypothesis_list.append(predicted_sentence)
        except Exception as e:
            print(e)

    # wer
    measures = jiwer.compute_measures(ground_truth_list, hypothesis_list)
    print("------", tsv, " OUTPUT --------")
    print(measures)
