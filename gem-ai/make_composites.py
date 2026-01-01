import os
from PIL import Image

RAW_DIR = "raw"
OUT_DIR = "composites"
ANGLES = ["0.jpg", "90.jpg", "180.jpg", "270.jpg"]

def make_one(stone_dir, out_path, tile=256):
    # tile = each small image size (tile x tile)
    imgs = []
    for a in ANGLES:
        p = os.path.join(stone_dir, a)
        im = Image.open(p).convert("RGB").resize((tile, tile))
        imgs.append(im)

    # 2x2 grid: [0,90]
    #           [180,270
    grid = Image.new("RGB", (tile*2, tile*2))
    grid.paste(imgs[0], (0, 0))
    grid.paste(imgs[1], (tile, 0))
    grid.paste(imgs[2], (0, tile))
    grid.paste(imgs[3], (tile, tile))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path, quality=95)

def main():
    for label in ["real", "synthetic"]:
        label_in = os.path.join(RAW_DIR, label)
        label_out = os.path.join(OUT_DIR, label)
        if not os.path.exists(label_in):
            continue

        for stone_id in os.listdir(label_in):
            stone_dir = os.path.join(label_in, stone_id)
            if not os.path.isdir(stone_dir):
                continue

            # check all 4 images exist
            ok = all(os.path.exists(os.path.join(stone_dir, a)) for a in ANGLES)
            if not ok:
                print("Missing angles in:", stone_dir)
                continue

            out_path = os.path.join(label_out, f"{stone_id}.jpg")
            make_one(stone_dir, out_path)

    print("Done. Composites saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
