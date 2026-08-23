"""AI-Hub 원본을 data/raw/<영문 클래스명>/ 으로 정리한다.

AI-Hub 「생활 폐기물 이미지」의 특성 두 가지를 반영한다:

  1) 폴더는 중분류(고철류·전자제품 등) 단위이고, 우리가 필요한 세부 품목명은
     라벨 JSON 안에 들어 있다. --detail-key 로 그 키를 지정한다.
  2) 1건 = 같은 물체 5장이다. 이 5장이 train/test로 흩어지면 데이터 누수가 되므로
     파일명을 <클래스>_<건ID>_<장번호>.jpg 로 만들어 이후 split_data 가
     --group-by-prefix 로 한 split에 묶을 수 있게 한다.

먼저 구조 파악:
    python -m src.inspect_source --source "D:/aihub/생활폐기물"

폴더명이 곧 품목명인 경우(folder 모드):
    python -m src.prepare_raw --source "D:/aihub" --mode folder --dry-run

라벨 JSON에서 품목명을 읽는 경우(json 모드, AI-Hub 기본):
    python -m src.prepare_raw --source "D:/aihub" --mode json \
        --detail-key DETAILS --group-key ID --crop-bbox --dry-run

확인 후 --dry-run 을 빼고 다시 실행한다.
"""
import argparse
import json
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image

from configs.classes import CLASSES, CLASS_KOR_NAME
from configs.config import RAW_DIR
from src.dataset import IMG_EXTS

# 한글 품목명 -> 영문 클래스 슬러그. AI-Hub DETAILS 표기 흔들림을 별칭으로 흡수한다.
KOR_TO_CLASS = {CLASS_KOR_NAME[c]: c for c in CLASSES}
KOR_TO_CLASS.update({
    "후라이팬": "fry_pan",
    "전기후라이팬": "electric_fry_pan",
    "옷걸이(철)": "steel_hanger",
    "철제옷걸이": "steel_hanger",
    "에어캡": "bubble_wrap",
    "뽁뽁이": "bubble_wrap",
    "스티로폼포장용기": "styrofoam_tray",
    "스티로폼완충재": "styrofoam_tray",
    "일회용컵": "disposable_cup",
    "음료수컵": "disposable_cup",
    "캔": "beverage_can",
    "알루미늄캔": "beverage_can",
    "핸드폰": "mobile_phone",
    "스마트폰": "mobile_phone",
    "다리미": "electric_iron",
    "밥솥": "electric_rice_cooker",
    "전기청소기": "vacuum_cleaner",
    "진공청소기": "vacuum_cleaner",
    "종이팩": "beverage_carton",
    "음료수팩": "beverage_carton",
    "우유팩": "beverage_carton",
})


def norm(name):
    """자모 분리(NFD) 저장분을 NFC로 통일하고 공백을 제거해 비교한다."""
    return unicodedata.normalize("NFC", str(name)).strip().replace(" ", "")


NORM_LOOKUP = {norm(k): v for k, v in KOR_TO_CLASS.items()}
NORM_LOOKUP.update({norm(c): c for c in CLASSES})


def load_json(path):
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def deep_get(obj, key, depth=0):
    """중첩 dict/list 어디에 있든 key 의 첫 값을 찾는다."""
    if depth > 6:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = deep_get(value, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_get(item, key, depth + 1)
            if found is not None:
                return found
    return None


def parse_bbox(raw, width, height):
    """bbox를 (left, top, right, bottom)으로 정규화. 형식이 애매하면 None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            if all(k in raw for k in keys):
                raw = [raw[k] for k in keys]
                break
        else:
            if all(k in raw for k in ("x", "y", "w", "h")):
                x, y, w, h = (raw[k] for k in ("x", "y", "w", "h"))
                raw = [x, y, x + w, y + h]
            else:
                return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None

    try:
        a, b, c, d = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None

    # x2<x1 이면 xywh 형식으로 간주
    if c <= a or d <= b:
        c, d = a + c, b + d

    left, top = max(0, int(a)), max(0, int(b))
    right, bottom = min(width, int(c)), min(height, int(d))
    if right - left < 16 or bottom - top < 16:
        return None
    return left, top, right, bottom


def find_image_for(json_path, source):
    """라벨 JSON에 대응하는 원본 이미지를 찾는다."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        sibling = json_path.with_suffix(ext)
        if sibling.exists():
            return sibling
    matches = list(source.rglob(json_path.stem + ".jpg"))
    return matches[0] if matches else None


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


def save_image(src_path, dest_path, bbox_raw, margin, max_side=0):
    """bbox가 있으면 여유를 두고 크롭하고, max_side로 축소해 저장한다."""
    with Image.open(src_path) as image:
        image = image.convert("RGB")

        box = parse_bbox(bbox_raw, image.width, image.height) if bbox_raw is not None else None
        if box is not None:
            left, top, right, bottom = box
            pad_x = int((right - left) * margin)
            pad_y = int((bottom - top) * margin)
            image = image.crop((max(0, left - pad_x), max(0, top - pad_y),
                                min(image.width, right + pad_x),
                                min(image.height, bottom + pad_y)))

        resized = shrink(image, max_side)
        if box is None and resized is image:
            # 크롭도 축소도 없으면 재인코딩 없이 원본을 그대로 복사
            shutil.copy2(src_path, dest_path)
            return False

        resized.save(dest_path, quality=92)
    return box is not None


def collect_from_json(source, detail_key, group_key, bbox_key):
    """라벨 JSON을 훑어 (슬러그 -> [(이미지경로, 건ID, bbox), ...]) 를 만든다."""
    buckets = defaultdict(list)
    unmatched = defaultdict(int)
    json_files = list(source.rglob("*.json"))
    print(f"라벨 JSON {len(json_files)}개 스캔 중...")

    for i, json_path in enumerate(json_files, 1):
        if i % 5000 == 0:
            print(f"  {i}/{len(json_files)}")
        data = load_json(json_path)
        if data is None:
            continue

        detail = deep_get(data, detail_key)
        if detail is None:
            continue
        slug = NORM_LOOKUP.get(norm(detail))
        if slug is None:
            unmatched[norm(detail)] += 1
            continue

        image_path = find_image_for(json_path, source)
        if image_path is None:
            continue

        group_id = deep_get(data, group_key) if group_key else None
        bbox = deep_get(data, bbox_key) if bbox_key else None
        buckets[slug].append((image_path, str(group_id or json_path.stem), bbox))

    return buckets, unmatched


def collect_from_folder(source):
    """폴더명이 곧 품목명인 경우."""
    buckets = defaultdict(list)
    for path in source.rglob("*"):
        if not path.is_dir():
            continue
        slug = NORM_LOOKUP.get(norm(path.name))
        if not slug:
            continue
        for image_path in sorted(path.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
                buckets[slug].append((image_path, image_path.stem, None))
    return buckets, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="AI-Hub 압축 해제 최상위 폴더")
    parser.add_argument("--mode", choices=["json", "folder"], default="json")
    parser.add_argument("--detail-key", default="DETAILS", help="세부 품목명이 담긴 JSON 키")
    parser.add_argument("--group-key", default="ID", help="같은 물체 5장을 묶는 건 ID 키")
    parser.add_argument("--bbox-key", default="BOX", help="바운딩박스 키")
    parser.add_argument("--crop-bbox", action="store_true", help="bbox로 크롭해 저장")
    parser.add_argument("--margin", type=float, default=0.08, help="크롭 여유 비율")
    parser.add_argument("--max-side", type=int, default=512,
                        help="저장 시 긴 변 최대 길이. 학습(224px) 대비 충분하면서 "
                             "디코딩을 크게 줄인다. 0이면 축소 안 함")
    parser.add_argument("--limit", type=int, default=0, help="클래스당 최대 장수 (0=제한없음)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"경로를 찾을 수 없습니다: {source}")

    if args.mode == "json":
        buckets, unmatched = collect_from_json(
            source, args.detail_key, args.group_key,
            args.bbox_key if args.crop_bbox else None)
    else:
        buckets, unmatched = collect_from_folder(source)

    print(f"\n{'클래스':<24}{'장수':>8}{'건수':>8}")
    print("-" * 40)
    total, cropped = 0, 0

    for slug in CLASSES:
        items = buckets.get(slug)
        if not items:
            continue

        # 같은 건(물체)끼리 묶어 정렬 -> 파일명에 건ID가 들어가 split에서 묶인다
        by_group = defaultdict(list)
        for image_path, group_id, bbox in items:
            by_group[group_id].append((image_path, bbox))

        if args.limit:
            keep, count = {}, 0
            for group_id, group_items in by_group.items():
                if count >= args.limit:
                    break
                keep[group_id] = group_items
                count += len(group_items)
            by_group = keep

        n_images = sum(len(v) for v in by_group.values())
        total += n_images
        print(f"{slug:<24}{n_images:>8}{len(by_group):>8}")

        if args.dry_run:
            continue

        dest = RAW_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        for g, (group_id, group_items) in enumerate(sorted(by_group.items()), 1):
            for k, (image_path, bbox) in enumerate(group_items, 1):
                target = dest / f"{slug}-g{g:05d}_{k:02d}.jpg"
                try:
                    if save_image(image_path, target, bbox, args.margin, args.max_side):
                        cropped += 1
                except OSError as exc:
                    print(f"  [실패] {image_path.name}: {exc}")

    print("-" * 40)
    print(f"{'합계':<24}{total:>8}")
    if cropped:
        print(f"bbox 크롭 적용: {cropped}장")

    missing = [c for c in CLASSES if c not in buckets]
    if missing:
        print(f"\n데이터 없는 클래스 {len(missing)}개: {', '.join(missing)}")

    if unmatched:
        print("\n매칭 실패한 품목명 상위 20개 (KOR_TO_CLASS에 별칭 추가 검토):")
        for name, count in sorted(unmatched.items(), key=lambda x: -x[1])[:20]:
            print(f"    {name}  ({count}건)")

    if args.dry_run:
        print("\n(dry-run: 실제로 옮기지 않았습니다)")
    else:
        print("\n다음: python -m src.split_data --group-by-prefix --prefix-sep _")


if __name__ == "__main__":
    main()
