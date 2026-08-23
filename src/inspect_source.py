"""다운로드한 AI-Hub 원본이 실제로 어떤 구조인지 조사한다.

라벨이 JSON인지 XML인지, 폴더명에 품목명이 있는지, 1건 5장이 파일명에서
어떻게 구분되는지를 미리 알 수 없으므로 일단 열어보고 보고만 한다.
아무것도 옮기거나 고치지 않는다.

사용:
    python -m src.inspect_source --source "D:/aihub"
"""
import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".json", ".xml", ".csv", ".txt", ".yaml", ".yml"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".7z", ".egg", ".alz", ".part"}


def read_text(path, limit=4000):
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return path.read_text(encoding=encoding)[:limit]
        except (UnicodeDecodeError, OSError):
            continue
    return None


def walk_keys(obj, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                walk_keys(value, path, depth + 1, out)
            else:
                out.append((path, value))
    elif isinstance(obj, list) and obj:
        # [10,10,30,30] 같은 스칼라 리스트는 통째로 값 취급해야 bbox 키를 놓치지 않는다
        if all(not isinstance(v, (dict, list)) for v in obj):
            out.append((prefix, obj))
        else:
            walk_keys(obj[0], f"{prefix}[0]", depth + 1, out)
    return out


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def report_inventory(source):
    section("1) 파일 종류별 개수")
    ext_counter = Counter()
    total_bytes = 0
    for path in source.rglob("*"):
        if path.is_file():
            ext_counter[path.suffix.lower()] += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    if not ext_counter:
        print("  파일이 없습니다. 경로를 확인하세요.")
        return ext_counter

    for ext, count in ext_counter.most_common(20):
        tag = ""
        if ext in IMG_EXTS:
            tag = "  <- 이미지"
        elif ext in LABEL_EXTS:
            tag = "  <- 라벨 후보"
        elif ext in ARCHIVE_EXTS:
            tag = "  <- 압축(아직 안 풀림)"
        print(f"  {ext or '(확장자없음)':<14}{count:>9}{tag}")
    print(f"\n  합계 {sum(ext_counter.values())}개 파일, {total_bytes / 1e9:.2f} GB")
    return ext_counter


def report_tree(source, max_depth=4, max_lines=100):
    section(f"2) 폴더 구조 (최대 {max_depth}단계)")
    shown = 0
    for path in sorted(source.rglob("*")):
        if not path.is_dir():
            continue
        rel = path.relative_to(source)
        if len(rel.parts) > max_depth:
            continue
        n_img = sum(1 for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
        suffix = f"   (이미지 {n_img}장)" if n_img else ""
        print("  " * (len(rel.parts) - 1) + f"[{path.name}]{suffix}")
        shown += 1
        if shown >= max_lines:
            print("  ... (생략)")
            break
    if shown == 0:
        print("  하위 폴더가 없습니다.")


def report_archives(source):
    archives = [p for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS]
    if not archives:
        return
    section("3) 아직 압축이 안 풀린 파일")
    for path in archives[:15]:
        size = path.stat().st_size / 1e9
        print(f"  {path.relative_to(source)}  ({size:.2f} GB)")
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()[:8]
                print(f"     안의 파일 예시: {names}")
            except (zipfile.BadZipFile, OSError) as exc:
                print(f"     (열 수 없음: {exc}) — 분할 압축이면 모두 받은 뒤 합쳐야 합니다")
    print(f"\n  총 {len(archives)}개. 압축을 먼저 풀어야 조사가 가능합니다.")


def report_filenames(source):
    section("4) 이미지 파일명 패턴 (1건=5장 확인용)")
    images = []
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            images.append(path)
            if len(images) >= 3000:
                break
    if not images:
        print("  이미지가 없습니다.")
        return

    print("  예시 10개:")
    for path in images[:10]:
        print(f"    {path.name}")

    # 마지막 숫자 토큰을 떼면 같은 물체끼리 묶이는지 확인
    groups = defaultdict(list)
    for path in images:
        stem = path.stem
        key = re.sub(r"[_\-\s]?\d+$", "", stem)
        groups[key].append(stem)

    sizes = Counter(len(v) for v in groups.values())
    print("\n  '파일명 끝 숫자 제거' 기준 그룹 크기 분포:")
    for size, count in sorted(sizes.items())[:10]:
        marker = "   <- 1건 5장 패턴!" if size == 5 else ""
        print(f"    {size}장짜리 그룹  {count}개{marker}")

    sample = next((k for k, v in groups.items() if len(v) == 5), None)
    if sample:
        print(f"\n  5장 묶임 예시: {sample}")
        for stem in sorted(groups[sample]):
            print(f"    {stem}")
        print("  -> split_data 의 --group-by-prefix 로 이 5장을 한 split에 묶을 수 있습니다.")
    else:
        print("\n  5장 그룹이 안 보입니다. 건 ID는 라벨 파일 안에 있을 수 있습니다.")


def report_labels(source, samples):
    label_files = [p for p in source.rglob("*")
                   if p.is_file() and p.suffix.lower() in LABEL_EXTS]
    section("5) 라벨 파일")
    if not label_files:
        print("  라벨 파일(JSON/XML/CSV)이 없습니다.")
        print("  -> 라벨링데이터 압축을 따로 받아야 하거나,")
        print("     품목명이 폴더명에만 있을 수 있습니다. 위 2)번 폴더 구조를 확인하세요.")
        return

    by_ext = Counter(p.suffix.lower() for p in label_files)
    print(f"  {dict(by_ext)}")

    json_files = [p for p in label_files if p.suffix.lower() == ".json"]
    if not json_files:
        print("\n  JSON이 아닌 라벨입니다. 아래 내용을 보고 형식을 확인하세요.")
        for path in label_files[:samples]:
            print(f"\n  --- {path.relative_to(source)} ---")
            text = read_text(path, 1200)
            print("  " + (text or "(읽기 실패)").replace("\n", "\n  "))
        return

    print(f"\n  JSON {len(json_files)}개. 전문 {samples}개:")
    for path in json_files[:samples]:
        print(f"\n  --- {path.relative_to(source)} ---")
        text = read_text(path, 100000)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            print("  (파싱 실패)")
            continue
        dumped = json.dumps(data, ensure_ascii=False, indent=2)
        print("  " + dumped[:2000].replace("\n", "\n  "))
        if len(dumped) > 2000:
            print("  ... (생략)")

    # 키별 값 분포 -> 어떤 키가 품목명/건ID/bbox 인지 추론
    section("6) JSON 키별 값 분포 (품목명·건ID·bbox 키 찾기)")
    scan = json_files[:400]
    values_by_key = defaultdict(Counter)
    for path in scan:
        text = read_text(path, 100000)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        for key, value in walk_keys(data):
            values_by_key[key][str(value)[:36]] += 1

    print(f"  ({len(scan)}개 스캔)\n")
    for key, counter in sorted(values_by_key.items()):
        uniq = len(counter)
        common = ", ".join(f"{v}({n})" for v, n in counter.most_common(4))
        hint = ""
        low = key.lower()
        if uniq <= 60 and any(t in low for t in ("detail", "class", "name", "item", "type")):
            hint = "   <<< 품목명 키 후보"
        elif uniq > len(scan) * 0.5 and any(t in low for t in ("id", "no", "seq")):
            hint = "   <<< 건 ID 키 후보"
        elif any(t in low for t in ("box", "bbox", "coord", "point", "polygon", "rect")):
            hint = "   <<< bbox 키 후보"
        print(f"  {key}  [고유값 {uniq}]{hint}")
        print(f"      {common}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="다운로드/압축해제한 최상위 폴더")
    parser.add_argument("--samples", type=int, default=2, help="전문 출력할 라벨 파일 수")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"경로를 찾을 수 없습니다: {source}")

    print(f"조사 대상: {source.resolve()}")
    report_inventory(source)
    report_tree(source)
    report_archives(source)
    report_filenames(source)
    report_labels(source, args.samples)

    section("정리: 다음 단계에 필요한 정보")
    print("  1. 세부 품목명(프라이팬 등)이 어디에 있는가?")
    print("       폴더명   -> prepare_raw --mode folder")
    print("       라벨 JSON -> prepare_raw --mode json --detail-key <키>")
    print("  2. 같은 물체 5장을 묶는 기준은?  파일명 / JSON의 ID 키")
    print("  3. bbox 키와 좌표 형식 (x1y1x2y2 / xywh)")
    print("\n  위 3개를 확인해 prepare_raw 옵션에 넣으세요.")


if __name__ == "__main__":
    main()
