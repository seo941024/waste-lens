"""AI-Hub 원천데이터를 data/train, data/val, data/test 로 정리한다.

AI-Hub 「생활 폐기물 이미지」(dataSetSn=140) 원천데이터의 실제 구조:

    생활 폐기물 이미지/
      Training/[T원천]도기류_화분_화분/14_X001_C014_1209/14_X001_C014_1209_0.jpg
                                       └ 건(물체) 폴더    └ 같은 물체 5장(_0~_4)
      Validation/[V원천]도기류_화분_화분/...

세 가지가 여기서 결정된다:

  1) 품목명은 폴더명 마지막 토큰이다. `[T원천]<중분류>_<품목>_<품목>` 에서 <품목>을 읽는다.
     라벨 JSON은 원천데이터에 포함되지 않으므로 파싱할 것이 없다.
  2) 건(물체) ID는 이미지의 부모 폴더명이다. 같은 건 5장이 흩어지면 데이터 누수가 되므로
     항상 건 단위로 묶어서 split에 배정한다.
  3) Training/Validation 이 이미 나뉘어 있다. Training 은 그대로 train 이 되고,
     Validation 은 건 단위로 절반씩 갈라 val 과 test 가 된다.
     (AI-Hub 가 test 셋을 따로 주지 않기 때문)

먼저 확인:
    python -m src.prepare_raw --source "C:/Users/.../생활 폐기물 이미지" --dry-run

문제없으면 --dry-run 을 빼고 실행한다. 클래스 불균형이 크므로
--limit 으로 클래스당 건수를 맞추는 것을 권장한다.
"""
import argparse
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image

from configs.classes import CLASSES, CLASS_KOR_NAME
from configs.config import SEED, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.dataset import IMG_EXTS

# 한글 품목명 -> 영문 클래스 슬러그. AI-Hub 표기 흔들림을 별칭으로 흡수한다.
KOR_TO_CLASS = {CLASS_KOR_NAME[c]: c for c in CLASSES}
KOR_TO_CLASS.update({
    "이동전화단말기": "mobile_phone",   # AI-Hub 폴더명 표기
    "포장용기": "styrofoam_tray",       # [T원천]스티로폼_포장용기_포장용기
    "후라이팬": "fry_pan",
    "전기후라이팬": "electric_fry_pan",
    "옷걸이(철)": "steel_hanger",
    "철제옷걸이": "steel_hanger",
    "에어캡": "bubble_wrap",
    "뽁뽁이": "bubble_wrap",
    "스티로폼포장용기": "styrofoam_tray",
    "일회용컵": "disposable_cup",
    "음료수컵": "disposable_cup",
    "캔": "beverage_can",
    "핸드폰": "mobile_phone",
    "스마트폰": "mobile_phone",
    "다리미": "electric_iron",
    "밥솥": "electric_rice_cooker",
    "진공청소기": "vacuum_cleaner",
    "종이팩": "beverage_carton",
    "우유팩": "beverage_carton",
})


def norm(name):
    """자모 분리(NFD) 저장분을 NFC로 통일하고 공백을 제거해 비교한다."""
    return unicodedata.normalize("NFC", str(name)).strip().replace(" ", "")


NORM_LOOKUP = {norm(k): v for k, v in KOR_TO_CLASS.items()}
NORM_LOOKUP.update({norm(c): c for c in CLASSES})


def slug_from_folder(folder_name):
    """'[T원천]도기류_화분_화분' -> 'flower_pot'. 못 알아보면 None."""
    name = norm(re.sub(r"^\[[^\]]*\]", "", folder_name))  # [T원천] 접두사 제거
    for token in reversed(name.split("_")):               # 마지막 토큰이 품목명
        slug = NORM_LOOKUP.get(token)
        if slug:
            return slug
    return NORM_LOOKUP.get(name)


def collect(split_dir):
    """<클래스 폴더>/<건 폴더>/*.jpg 를 훑어 {슬러그: {건ID: [경로,...]}} 를 만든다."""
    buckets = defaultdict(lambda: defaultdict(list))
    unmatched = defaultdict(int)

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        slug = slug_from_folder(class_dir.name)
        if slug is None:
            unmatched[class_dir.name] += 1
            continue
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
                # 건 ID = 부모 폴더명. 건 폴더가 없는 배치면 파일명으로 대체한다.
                group_id = (image_path.parent.name
                            if image_path.parent != class_dir else image_path.stem)
                buckets[slug][group_id].append(image_path)

    return buckets, unmatched


def shrink(image, max_side):
    """긴 변이 max_side를 넘으면 줄인다.

    학습은 224px로 하므로 원본 1920x1080을 그대로 두면 매 epoch마다
    쓸데없이 큰 JPEG를 디코딩하게 되어 GPU가 논다. 전처리에서 한 번만 줄여둔다.
    """
    if not max_side or max(image.size) <= max_side:
        return image
    ratio = max_side / max(image.size)
    new_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(new_size, Image.BILINEAR)


def save_image(src_path, dest_path, max_side):
    """max_side로 축소해 저장한다. 축소가 필요 없으면 재인코딩 없이 복사."""
    if not max_side:
        shutil.copy2(src_path, dest_path)
        return
    with Image.open(src_path) as image:
        if max(image.size) <= max_side:
            shutil.copy2(src_path, dest_path)
            return
        shrink(image.convert("RGB"), max_side).save(dest_path, quality=92)


def write_split(slug, groups, dest_root, max_side, dry_run):
    """건 단위로 받은 이미지들을 <dest_root>/<slug>/ 에 저장하고 장수를 돌려준다."""
    n_images = sum(len(v) for v in groups.values())
    if dry_run or not n_images:
        return n_images

    dest = dest_root / slug
    dest.mkdir(parents=True, exist_ok=True)
    for group_id, paths in sorted(groups.items()):
        for k, image_path in enumerate(sorted(paths), 1):
            target = dest / f"{slug}-{group_id}_{k:02d}.jpg"
            try:
                save_image(image_path, target, max_side)
            except OSError as exc:
                print(f"  [실패] {image_path.name}: {exc}")
    return n_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Training/ 과 Validation/ 을 담고 있는 최상위 폴더")
    parser.add_argument("--max-side", type=int, default=512,
                        help="저장 시 긴 변 최대 길이. 학습(224px) 대비 충분하면서 "
                             "디코딩을 크게 줄인다. 0이면 축소 안 함")
    parser.add_argument("--limit", type=int, default=0,
                        help="클래스당 최대 '건수' (0=제한없음). 클래스 불균형 완화용")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    train_src, val_src = source / "Training", source / "Validation"
    for path in (train_src, val_src):
        if not path.is_dir():
            raise SystemExit(f"폴더를 찾을 수 없습니다: {path}")

    train_buckets, unmatched = collect(train_src)
    val_buckets, unmatched_v = collect(val_src)
    unmatched.update(unmatched_v)

    rng = random.Random(SEED)
    print(f"\n{'클래스':<24}{'train':>9}{'val':>8}{'test':>8}")
    print("-" * 49)
    total = {"train": 0, "val": 0, "test": 0}

    for slug in CLASSES:
        train_groups = dict(train_buckets.get(slug, {}))
        val_groups = dict(val_buckets.get(slug, {}))
        if not train_groups and not val_groups:
            continue

        # 건 단위로 자른다. 같은 물체 5장이 통째로 남거나 통째로 빠진다.
        if args.limit and len(train_groups) > args.limit:
            keep = rng.sample(sorted(train_groups), args.limit)
            train_groups = {g: train_groups[g] for g in keep}

        # AI-Hub는 test를 따로 주지 않으므로 Validation을 건 단위로 반씩 나눈다.
        val_keys = sorted(val_groups)
        rng.shuffle(val_keys)
        half = len(val_keys) // 2
        test_groups = {g: val_groups[g] for g in val_keys[:half]}
        val_groups = {g: val_groups[g] for g in val_keys[half:]}

        counts = {
            "train": write_split(slug, train_groups, TRAIN_DIR, args.max_side, args.dry_run),
            "val": write_split(slug, val_groups, VAL_DIR, args.max_side, args.dry_run),
            "test": write_split(slug, test_groups, TEST_DIR, args.max_side, args.dry_run),
        }
        for split, n in counts.items():
            total[split] += n
        print(f"{slug:<24}{counts['train']:>9}{counts['val']:>8}{counts['test']:>8}")

    print("-" * 49)
    print(f"{'합계':<24}{total['train']:>9}{total['val']:>8}{total['test']:>8}")

    missing = [c for c in CLASSES if c not in train_buckets and c not in val_buckets]
    if missing:
        print(f"\n데이터 없는 클래스 {len(missing)}개: {', '.join(missing)}")

    if unmatched:
        print("\n클래스로 매칭하지 못한 폴더 (KOR_TO_CLASS에 별칭 추가 검토):")
        for name in sorted(unmatched):
            print(f"    {name}")

    if args.dry_run:
        print("\n(dry-run: 실제로 옮기지 않았습니다)")
    else:
        print("\n다음: python -m src.train --name baseline --no-augment --epochs 10")


if __name__ == "__main__":
    main()
