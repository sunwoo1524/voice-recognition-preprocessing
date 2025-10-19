import librosa
import librosa.display
import numpy as np
import sounddevice as sd


# print("녹음이 시작되었습니다.\nEnter 키를 눌러 녹음을 종료하세요.")

# recorded_data = []

# def callback(indata, frames, time, status):
#     if status:
#         print(f"상태: {status}")
#     recorded_data.append(indata.copy())

# stream = sd.InputStream(samplerate=sr, channels=1, callback=callback, blocksize=1024)

# stream.start()
# input()
# stream.stop()
# stream.close()

# print("녹음이 종료되었습니다!")

# audio_data = np.concatenate(recorded_data, axis=0).flatten()

sr = 22050
n_fft = 2048
hop_length = 512
DURATION = 216 # record for 216 frame (about 5 sec)

print(f"녹음 중...")
recording = sd.rec(int((DURATION - 1) * (hop_length / sr) * sr), samplerate=sr, channels=1)
sd.wait()
print(f"약 {(DURATION - 1) * (hop_length / sr)}초 간 녹음함")

# convert to 1d array
audio_data = recording.flatten()

# optimum setting values for voice recognition (by Claude)
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio_data,
    sr=sr,
    n_fft=n_fft,
    hop_length=512,
    n_mels=128,
    fmax=8000
)

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

print(mel_spectrogram_db.shape)
print(mel_spectrogram_db)

# vislualize
# import matplotlib.pyplot as plt

# plt.subplot(2, 1, 1)
# librosa.display.waveshow(audio_data, sr=sr)
# plt.title('Audio Waveform')
# plt.xlabel('Time (sec)')
# plt.ylabel('Amplitude')

# plt.subplot(2, 1, 2)
# img = librosa.display.specshow(
#     mel_spectrogram_db,
#     sr=sr,
#     hop_length=512,
#     x_axis='time',
#     y_axis='mel',
#     fmax=8000,
#     cmap='viridis'
# )
# plt.colorbar(img, format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.xlabel('Time (sec)')
# plt.ylabel('frequency (Hz)')

# plt.tight_layout()
# plt.savefig('mel_spectrogram.png', dpi=300, bbox_inches='tight')
# plt.show()
