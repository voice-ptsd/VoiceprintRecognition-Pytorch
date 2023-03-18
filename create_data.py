import json
import os
from typing import Union, Dict, List

folders: List[str] = [
    "dataset/aidatatang_200zh/corpus/train",
    "dataset/aidatatang_200zh/corpus/dev",
    "dataset/aidatatang_200zh/corpus/test",
]
files: Dict[int, Dict[str, Union[int, List]]] = {}
speakers: Dict = {}
sounds: int = 0

label: os.DirEntry
audio: os.DirEntry

for folder in folders:
    for label in os.scandir(folder):
        if not label.is_dir():
            continue
        # 新人入
        if label.name not in speakers:
            speakers[label.name] = len(speakers)
        if speakers[label.name] not in files:
            files[speakers[label.name]] = {
                "sounds_count": 0,
                "sounds": [],
            }
        for audio in os.scandir(label):
            if not audio.name.endswith(".wav"):
                continue
            sounds += 1
            files[speakers[label.name]]["sounds_count"] += 1
            files[speakers[label.name]]["sounds"].append(audio.path.replace("\\", "/"))
print('::total speakers: %d\n::total sounds: %d' % (len(speakers), sounds))
data = {
    "total_speakers": len(speakers),
    "total_sounds": sounds,
    "speakers": {**speakers, **{value: key for key, value in speakers.items()}},
    "sounds": files,
}

json.dump(data, open("dataset/data.json", "w", encoding="utf-8"), indent=4)
