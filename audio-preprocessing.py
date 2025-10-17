import librosa
import librosa.display
import numpy as np
import sounddevice as sd
import threading

# mic: TODO
sr = 22050
recorded_data = []
is_recording = False

def record_audio():
    global recorded_data, is_recording
    
    def callback(indata, frames, time, status):
        if status:
            print(f"상태: {status}")
        if is_recording:
            recorded_data.append(indata.copy())
    
    with sd.InputStream(samplerate=sr, channels=1, callback=callback):
        while is_recording:
            sd.sleep(100)

print("녹음 중...")
print("엔터 키로 녹음 종료")

is_recording = True
recorded_data = []

record_thread = threading.Thread(target=record_audio)
record_thread.start()

is_recording = False
record_thread.join()

print("녹음 종료")

audio_data = np.concatenate(recorded_data, axis=0).flatten()

# optimum setting values for voice recognition (by Claude)
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio_data,
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
