import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import sklearn


audio_data = "C:\\Users\\Бобр Даша\\Desktop\\university\\3 КУРС\\5 сем\\Язык Python для работы с данными\\lab_5\\nukadeti (mp3cut.net).mp3"
y, sr = librosa.load(audio_data)


"""анализ сигнала"""

# сигнал в амплитудно-временной форме


plt.figure(figsize=(15, 5))
librosa.display.waveshow(y, sr=sr)


# используя преобразование Фурье, отрисовала частотный спектр
spectrum = fft(y)

frequencies = np.fft.fftfreq(len(spectrum), 1 / sr)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[: len(frequencies) // 2], np.abs(spectrum[: len(spectrum) // 2]))
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.title("Спектр частот")
plt.grid(True)


# отрисовала спектограмму сигнала

X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar()


"""выделение признаков"""

# вывела значение темпа и количество бит

y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
print("Detected Tempo: " + str(tempo) + " beats/min")
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_time_diff = np.ediff1d(beat_times)
beat_nums = np.arange(1, np.size(beat_times))

print(beat_nums)

"""
fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
ax.set_ylabel("Time difference (s)")
ax.set_xlabel("Beats")
g=sns.barplot(beat_time_diff,  palette="BuGn_d",ax=ax)
g=g.set(xticklabels=[])

вот в чем состояла ошибка:
Функция barplot() принимает от 0 до 1 позиционного аргумента, но было задано 2 позиционных аргумента 
(и 2 аргумента, относящихся только к ключевым словам)
"""

# получила и вывела в виде изображения и numpy массив мел-кепстральные коэффициенты

mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfccs, x_axis="time")
plt.colorbar()
plt.title("MFCC")


# получила и вывела изображение спектрального центроида на одном изображении с сигналом в амплитудно-временной форме

cent = librosa.feature.spectral_centroid(y=y, sr=sr)

spectral_centroids = cent[0]
spectral_centroids.shape

plt.figure(figsize=(15, 5))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


def normalize(y, axis=0):
    return sklearn.preprocessing.minmax_scale(y, axis=axis)


librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color="r")

plt.show()


# Что такое гармоническая и перкусионная части сигнала, зачем они нужны

"""
Гармоническая составляющая сигнала представляется его элементами, известными как гармоники, которые определяются амплитудами, 
начальными фазами и частотами, кратными основной частоте. 
Перкуссионная часть сигнала — это его составляющая, включающая ударные колебания.
Обе составляющие — гармоническая и перкуссионная — играют ключевую роль в музыкальном восприятии, композиции и звукозаписи.
Они позволяют создавать богатые и многослойные звуковые текстуры
Гармоники создают тональность и характерный тембр звука.
Понимание гармонической структуры сигнала помогает в анализе его частотного спектра. 
Перкуссионные звуки добавляют текстуру и глубину к звуку. 
Перкуссионные составляющие сигнала отвечают за ритмические элементы и акценты, создаваемые ударами или щелчками. 
"""
