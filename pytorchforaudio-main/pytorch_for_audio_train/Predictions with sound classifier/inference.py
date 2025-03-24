import torch
import torchaudio

from cnn import CNNNetwork
from sounddataset import SoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

"""
class_mapping match the file number of the file Sound/audio or look at Sound/Sound_README.txt
"""
# dataset mapping
class_mapping = [
    "bee_buzz",
    "elephant_trumpet",
    "kitten_bark",
    "dog_bark",
    "woodpecker_peck",
    "sloth_snore",
    "flamingo_alarm",
    "kangaroo_thump",
    "wolf_howl",
    "nightingale_song"
]


# prediction
def predict(model, input, target, class_mapping):
    # switch to evaluation mode
    model.eval()
    # disable gradient
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        # expected
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    cnn = CNNNetwork()
    # load model parameters
    state_dict = torch.load("feedforwardnet.pth")
    # load state_dict
    cnn.load_state_dict(state_dict)


    # load model parameters(for SD use)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # load dataset (for SD use)
    usd = SoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu"
                            )


    # select a sample(input & target from sd)
    index = 189
    input = usd[index][0]
    target =usd[index][1] 

    # Add a dimension to the begining of the innput
    input.unsqueeze_(0)    

    # prediction
    predicted, expected = predict(cnn, 
                                  input, 
                                  target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
