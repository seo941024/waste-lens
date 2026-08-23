"""data/raw/<class_name>/*.jpg 를 train/val/test로 분할한다.

계획서 5.3: Train 70% / Val 15% / Test 15%.
같은 물체를 연속 촬영한 이미지가 split을 넘나드는 데이터 누수를 막기 위해
--group-by-prefix 옵션으로 파일명 앞부분이 같은 이미지를 한 split에 묶을 수 있다.

사용:
    python -m src.split_data
    python -m src.split_data --group-by-prefix --prefix-sep _
"""
import argparse
import random
import shutil
from collections import defaultdict

from configs.classes import CLASSES
from configs.config import RAW_DIR, SEED, SPLIT_RATIO, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.dataset import IMG_EXTS

SPLIT_DIRS = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}


def group_key(path, group_by_prefix, sep):
    if not group_by_prefix:
        return path.name
    return path.stem.rsplit(sep, 1)[0] if sep in path.stem else path.stem


def split_class(class_name, group_by_prefix, sep, copy, rng):
    src_dir = RAW_DIR / class_name
    if not src_dir.is_dir():
        print(f"  [skip] {class_name}: data/raw에 폴더 없음")
        return None

    images = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"  [skip] {class_name}: 이미지 0장")
        return None

    groups = defaultdict(list)
    for path in images:
        groups[group_key(path, group_by_prefix, sep)].append(path)

    keys = list(groups)
    rng.shuffle(keys)

    n_val = int(len(keys) * SPLIT_RATIO["val"])
    n_test = int(len(keys) * SPLIT_RATIO["test"])
    if len(keys) >= 3:  # 그룹이 적어도 val/test가 비지 않도록 보정
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    n_train = len(keys) - n_val - n_test
    assignment = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }

    counts = {}
    for split, split_keys in assignment.items():
        dest = SPLIT_DIRS[split] / class_name
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for key in split_keys:
            for path in groups[key]:
                target = dest / path.name
                if copy:
                    shutil.copy2(path, target)
                else:
                    shutil.move(str(path), target)
                moved += 1
        counts[split] = moved

    print(f"  {class_name}: train {counts['train']} / val {counts['val']} / test {counts['test']}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-by-prefix", action="store_true",
                        help="파일명 접두사가 같은 이미지를 같은 split에 묶는다 (데이터 누수 방지)")
    parser.add_argument("--prefix-sep", default="_", help="접두사 구분자")
    parser.add_argument("--move", action="store_true", help="복사 대신 이동")
    args = parser.parse_args()

    rng = random.Random(SEED)
    print(f"원본: {RAW_DIR}")

    total = {"train": 0, "val": 0, "test": 0}
    for class_name in CLASSES:
        counts = split_class(class_name, args.group_by_prefix, args.prefix_sep,
                             copy=not args.move, rng=rng)
        if counts:
            for split in total:
                total[split] += counts[split]

    print(f"\n합계 → train {total['train']} / val {total['val']} / test {total['test']}")


if __name__ == "__main__":
    main()
