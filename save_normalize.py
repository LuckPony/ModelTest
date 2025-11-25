import json


def save_normalize(noisy_img, save_path="minmax.json"):
    stats = {}
    for f_id in range(len(noisy_img)):
        stats[f_id] = {
            "noisy": {
                "min": float(noisy_img[f_id].min()),
                "max": float(noisy_img[f_id].max()),
            },
        }
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved normalization stats to {save_path}")
