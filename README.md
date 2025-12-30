# Selbstfahrendes Auto

## Einleitung
Ein Projekt für mein Fachreferat der FOS/BOS Bad Neustadt. Es werden die Effektivität von Kamera, LiDAR und einem fusionierten System in verschiedenen Situationen verglichen.

## Instalation

Folge dieser [Anleitung](https://carla.readthedocs.io/en/latest/start_quickstart/) um Carla zu instalieren

Lade pyhon3.12.10 herunter ([Windows](https://www.python.org/downloads/windows/), [MacOS](https://www.python.org/downloads/macos/)) und installiere es

Lade dieses Repository herunter und entpacke es.

Installiere alle Abhängigkeiten:

```bash
python3.12 -m pip install tensorflow pandas scikit-learn matplotlib carla pydot opencv-python 
```

Falls du eine NVIDIA-Grafikkarte hast, installiere das [CUDA-Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
python3.12 -m pip install 'tensorflow[and-cuda]' 
```

## Benutzung

### Trainieren

Um das Trainieren eines Modells zu starten, starte erst Carla, wie in der oben genannten Anleitung beschrieben.

Führe dann erst das ```setup.py``` Skript aus, um die richtige Karte auszuwählen. Verbinde dann einen Controller und starte das Skript ```record_advanced.py``` im Ordner ```training```. Fahre etwas herum, um Daten zu sammeln. Für ein "Highway-Model" solltest du etwa 20 Minuten fahren. Wenn du auch durch die Stadt fährst, etwa 1–2 Stunden.

Sobald du fertig bist, kannst du die verschiedenen ```prepocessing_<Model>.py``` Skripte starten, um die Daten für die verschiedenen KI-Modelle zu verarbeiten.

Um die Modelle zu trainieren, starte die ```mode_<Model>.py``` Skripte. Vor allem das Kamera- und Fusions-Modell kann abhängig von den gesammelten Daten mehrere Stunden dauern.

### Fahren

Kopiere die ```best_model_<Model>.h``` Dateien in den ```ai-driving``` Ordner.

Im ```ai-driving``` Ordner, starte das ```carla_control.py``` Skript. Hierfür muss wieder ein Controller verbunden sein. Wähle das Modell aus und wechsle den Modus von „Manuell“ auf „AI“.

### Visualisierung

Im Hauptordner können verschiedene Visualisierungen mit den ```visual_<Model>.py``` Skripten und dem ```lidar_pointcloud.py``` Skript erstellt werden. Führe sie einfach aus. Für ein gutes Ergebnis des letzten Skriptes solltest du ```record_advanced.py``` starten und, wenn du eine gute Position hast, stillstehenbleiben und in den „Visualizer-Modi“ wechseln.
