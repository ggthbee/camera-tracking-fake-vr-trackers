import cv2
import numpy as np

# RTMP-Stream-URL
stream_url = "rtmp://localhost/live/test"

# Initialisiere Videoaufnahme mit FFmpeg und Low-Latency-Optionen
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # Minimale Puffergröße
cap.set(cv2.CAP_PROP_FPS, 30)  # Stelle sicher, dass die Bildrate mit OBS übereinstimmt

if not cap.isOpened():
    print("Fehler: RTMP-Stream konnte nicht geöffnet werden. Stelle sicher, dass OBS streamt und FFmpeg installiert ist.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Frame konnte nicht gelesen werden")
        break

    # Beispielverarbeitung: In Graustufen umwandeln
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Verarbeiteten Frame anzeigen
    cv2.imshow('OBS Stream', frame_gray)

    # Beenden mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()