import model_v2 as model
import torch
import pandas as pd
import glob
import torchaudio
import json
from tqdm import tqdm

import IPython

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open('labels.json') as f:
    labels = json.load(f)

# Load checkpoint
model = model.AudioTaggingModel().to(device)
model.load_state_dict(torch.load('outputs/audio_classification_v2.pt'))
model.eval()

# Run files through it
files = glob.glob("data/test/*.wav")

transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=80,
    win_length=50,
    hop_length=25
)


def get_mel_spec(audio_path):
    wav, sr = torchaudio.load(f, normalize=True)
    # resample
    if sr != 16000:
        resampler=torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000,
            dtype=wav.dtype
        )        
        wav = resampler(wav)

    if wav.shape[-1] > 16000 * 5: # if wav is longer than 5 seconds then we trim it to 5 seconds, we make all audio data 5 seconds
        wav = wav[0,:16000*5]
    audio_padded = torch.zeros(1, 16000 * 5, dtype=torch.float)
    audio_padded[0, :wav.shape[-1]] = wav

    mel_spec = transform(audio_padded)
    return mel_spec


data_list = []
P_THRESHOLD = 0.25

for f in tqdm(files):
    # get mel_spec
    mel_spec = get_mel_spec(f)
    mel_spec = mel_spec.to(device)

    # model outputs
    with torch.no_grad():
        outputs = model(mel_spec)

    fname = f.split('/')[-1]

    data = {
        "fname": fname
    }

    # add binary var to each class
    for idx, output in enumerate(outputs[0]):
        if output > P_THRESHOLD:
            predicted = 1
        else:
            predicted = 0

        label = list(labels.keys())[idx]

        data[label] = predicted
    
    data_list.append(data)

# Create pandas dataframe for submission
df = pd.DataFrame(data_list, columns=["fname","Accelerating_and_revving_and_vroom","Accordion","Acoustic_guitar","Applause","Bark","Bass_drum","Bass_guitar","Bathtub_(filling_or_washing)","Bicycle_bell","Burping_and_eructation","Bus","Buzz","Car_passing_by","Cheering","Chewing_and_mastication","Child_speech_and_kid_speaking","Chink_and_clink","Chirp_and_tweet","Church_bell","Clapping","Computer_keyboard","Crackle","Cricket","Crowd","Cupboard_open_or_close","Cutlery_and_silverware","Dishes_and_pots_and_pans","Drawer_open_or_close","Drip","Electric_guitar","Fart","Female_singing","Female_speech_and_woman_speaking","Fill_(with_liquid)","Finger_snapping","Frying_(food)","Gasp","Glockenspiel","Gong","Gurgling","Harmonica","Hi-hat","Hiss","Keys_jangling","Knock","Male_singing","Male_speech_and_man_speaking","Marimba_and_xylophone","Mechanical_fan","Meow","Microwave_oven","Motorcycle","Printer","Purr","Race_car_and_auto_racing","Raindrop","Run","Scissors","Screaming","Shatter","Sigh","Sink_(filling_or_washing)","Skateboard","Slam","Sneeze","Squeak","Stream","Strum","Tap","Tick-tock","Toilet_flush","Traffic_noise_and_roadway_noise","Trickle_and_dribble","Walk_and_footsteps","Water_tap_and_faucet","Waves_and_surf","Whispering","Writing","Yell","Zipper_(clothing)"])

df.to_csv('submission_v2.tsv', sep="\t")