import torch
import torchaudio

torch.random.manual_seed(3407)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Character-based Encoding
""" Encodes text string into character-symbols

We can also use TACOTRON2_WAVERNN_PHONE_LJSPEECH for phoneme based encoding.
"""
# symbols = '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'
# lookup = {s: i for i, s in enumerate(symbols)}
# symbols = set(symbols)

# def text_to_sequence(text):
#     text = text.lower()
#     return [lookup[s] for s in text if s in symbols]

text = "Hello world, I am A I become sentient. Tacotron 2 improves and simplifies the original architecture. While there are no major differences, let's see its key points. Mel spectrograms are generated and passed to the Vocoder as opposed to Linear-scale spectrograms.Parallel WaveNet is 1000 times faster than the original networks and can produce 20 seconds of audio in 1 second."  # diminishing quality on longer text sequences + there is a max step length on the decoder

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)

torchaudio.save("output.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
