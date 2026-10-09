"""Public frozen parameters and action labels, without local machine paths."""
from importlib.resources import files
import json

ACTION_LABELS = [
    "右臂向左划", "右臂向右划", "右手挥手", "双手前拍", "右臂投掷",
    "交叉双臂于胸前", "篮球投篮", "右手画X", "右手顺时针画圆", "右手逆时针画圆",
]


def device_models():
    resource = files("photo_synapse_rc").joinpath("resources/device_models.json")
    return json.loads(resource.read_text(encoding="utf-8"))["models"]
