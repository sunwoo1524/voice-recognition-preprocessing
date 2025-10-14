import librosa
import librosa.display
import numpy as np
import sounddevice as sd

# mic: TODO

y, sr = librosa.load("./input.wav")

# print(len(y), sr, len(y)/sr)

# optimum setting values for voice recognition (by Claude)
mel_spectrogram = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmax=8000
)

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# print(mel_spectrogram_db.shape)

# vislualize
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
img = librosa.display.specshow(
    mel_spectrogram_db,
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='mel',
    fmax=8000,
    cmap='viridis'
)
plt.colorbar(img, format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (sec)')
plt.ylabel('frequency (Hz)')

plt.tight_layout()
plt.savefig('mel_spectrogram.png', dpi=300, bbox_inches='tight')
# plt.show()
