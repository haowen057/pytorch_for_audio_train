import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050


class _Keyword_Spotting_Service:
    
    model = None
    _mapping = [
        "bed", 
        "bird", 
        "cat", 
        "dog", 
        "down", 
        "eight", 
        "five", 
        "four", 
        "go", 
        "happy",
        "house", 
        "left", 
        "marvin", 
        "nine", 
        "no", 
        "off", 
        "on", 
        "one", 
        "right", 
        "seven",
        "sheila", 
        "six", 
        "stop", 
        "three", 
        "tree", 
        "two", 
        "up", 
        "wow", 
        "yes", 
        "zero"
    ]  
    
    _instance = None


    def predict(self, file_path):

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # 4D input for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal, 
                                         sr=sample_rate, 
                                         n_mfcc=num_mfcc, 
                                         n_fft=n_fft,
                                         hop_length=hop_length
                                         )
            MFCCs = MFCCs.T

            if MFCCs.shape[0] < 40:
                pad_width = 40 - MFCCs.shape[0]
                MFCCs = np.pad(MFCCs, ((0, pad_width), (0, 0)), mode='constant')
            elif MFCCs.shape[0] > 40:
                MFCCs = MFCCs[:40, :]

        return MFCCs
    


def Keyword_Spotting_Service():

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # create 2 instances for control group
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check the instances 
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("4- Making Predictions with the Speech Recognition System/test/left.wav")
    print(keyword)