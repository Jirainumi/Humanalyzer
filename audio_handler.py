import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import time
import tempfile


class AudioHandler:
    """Sesli komut alma ve sonuçları seslendirme işlemleri için sınıf"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.temp_dir = tempfile.gettempdir()

    def listen_command(self):
        """Mikrofondan sesli komut dinler ve metne çevirir"""
        with sr.Microphone() as source:
            print("Dinleniyor...")
            audio = self.recognizer.listen(source)

        try:
            command = self.recognizer.recognize_google(audio, language="tr-TR")
            print(f"Algılanan komut: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Anlaşılamadı")
            return None
        except sr.RequestError:
            print("Google servisi bağlantı hatası")
            return None

    def speak_text(self, text, lang="tr"):
        """Metni seslendirir"""
        try:
            # Geçici dosya oluştur
            tts = gTTS(text=text, lang=lang)
            temp_file = os.path.join(self.temp_dir, f"temp_audio_{int(time.time())}.mp3")
            tts.save(temp_file)

            # Ses dosyasını çal
            playsound.playsound(temp_file)

            # Geçici dosyayı sil
            try:
                os.remove(temp_file)
            except:
                pass

        except Exception as e:
            print(f"Seslendirme hatası: {e}")