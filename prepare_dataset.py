import librosa
import os
import json
import numpy as np

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050  # 1秒音频（假设采样率22.05kHz）
SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac')

# 新增参数：固定MFCC帧数（根据SAMPLES_TO_CONSIDER和hop_length计算）
N_FFT = 2048
HOP_LENGTH = 512
MAX_MFCC_FRAMES = (SAMPLES_TO_CONSIDER - N_FFT) // HOP_LENGTH + 1  # 计算结果为固定帧数

def preprocess_dataset(dataset_path, json_path, num_mfcc=13):
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            continue

        label = os.path.basename(dirpath)
        data["mapping"].append(label)
        print(f"\nProcessing: '{label}'")

        for f in filenames:
            if not f.lower().endswith(SUPPORTED_FORMATS):
                print(f"Skipping non-audio file: {f}")
                continue

            file_path = os.path.join(dirpath, f)
            try:
                # 1. 加载音频并统一长度
                signal, sample_rate = librosa.load(file_path, sr=None)
                if len(signal) > SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]  # 截断
                else:
                    # 短音频补零
                    signal = np.pad(signal, (0, max(0, SAMPLES_TO_CONSIDER - len(signal))), 
                                  mode="constant")

                # 2. 提取MFCC并固定帧数
                MFCCs = librosa.feature.mfcc(
                    y=signal,
                    sr=sample_rate,
                    n_mfcc=num_mfcc,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )

                # 3. 统一MFCC帧数（截断或填充）
                if MFCCs.shape[1] > MAX_MFCC_FRAMES:
                    MFCCs = MFCCs[:, :MAX_MFCC_FRAMES]  # 截断
                else:
                    # 填充到MAX_MFCC_FRAMES（末尾补0）
                    pad_width = ((0, 0), (0, MAX_MFCC_FRAMES - MFCCs.shape[1]))
                    MFCCs = np.pad(MFCCs, pad_width, mode="constant")

                data["MFCCs"].append(MFCCs.T.tolist())  # 转置为 (帧数, MFCC系数)
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print(f"Processed: {file_path} (label: {i-1}, frames: {MFCCs.shape[1]})")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print(f"\nData saved to {json_path}. MFCC shape: ({MAX_MFCC_FRAMES}, {num_mfcc})")

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)