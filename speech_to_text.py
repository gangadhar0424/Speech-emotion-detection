import speech_recognition as sr

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Transcribed text: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

if __name__ == "__main__":
    audio_path = r"C:\Users\ganga\OneDrive\Desktop\Speech-emotion-detection\dataset\Actor_01\03-01-01-01-02-01-01.wav"
    text = speech_to_text(audio_path)
    print(f"Result: {text}")
