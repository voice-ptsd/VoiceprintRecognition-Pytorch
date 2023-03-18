import json
import math
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

train_file = open("dataset/train_list.txt", "w", encoding="utf-8")
test_file = open("dataset/test_list.txt", "w", encoding="utf-8")

for folder in folders:
    for label in os.scandir(folder):
        if not label.is_dir():
            continue
        # 新的说话人创建映射
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

for i in data['sounds'].keys():
    s = math.ceil(data['sounds'][i]['sounds_count'] * 0.8)
    for sound in data['sounds'][i]['sounds'][0:s]:
        train_file.write("%s\t%d\n" % (sound, i))
    for sound in data['sounds'][i]['sounds'][s:]:
        test_file.write("%s\t%d\n" % (sound, i))
train_file.close()
test_file.close()
