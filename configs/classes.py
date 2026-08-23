# 폐기물 배출 안내 AI - 학습 클래스 정의 (총 18종)
# AIHub 「생활 폐기물 이미지」 데이터셋의 DETAILS(세부 물품) 기준
CLASSES = [
    "fry_pan",              # 1. 프라이팬
    "electric_fry_pan",     # 2. 전기프라이팬
    "steel_hanger",         # 3. 철옷걸이
    "cutting_board",        # 4. 도마
    "ttukbaegi",            # 5. 뚝배기
    "flower_pot",           # 6. 화분
    "sealed_container",     # 7. 밀폐용기
    "bubble_wrap",          # 8. 에어캡(뽁뽁이)
    "styrofoam_tray",       # 9. 스티로폼 포장용기
    "pet_bottle",           # 10. 페트병
    "disposable_cup",       # 11. 일회용 음료수잔
    "beverage_can",         # 12. 음료수캔
    "mobile_phone",         # 13. 휴대전화
    "electric_iron",        # 14. 전기다리미
    "electric_rice_cooker", # 15. 전기밥솥
    "electric_fan",         # 16. 선풍기
    "vacuum_cleaner",       # 17. 청소기
    "beverage_carton",      # 18. 음료수곽
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# 한글 표시명 (웹 데모/리포트용)
CLASS_KOR_NAME = {
    "fry_pan": "프라이팬",
    "electric_fry_pan": "전기프라이팬",
    "steel_hanger": "철옷걸이",
    "cutting_board": "도마",
    "ttukbaegi": "뚝배기",
    "flower_pot": "화분",
    "sealed_container": "밀폐용기",
    "bubble_wrap": "에어캡(뽁뽁이)",
    "styrofoam_tray": "스티로폼 포장용기",
    "pet_bottle": "페트병",
    "disposable_cup": "일회용 음료수잔",
    "beverage_can": "음료수캔",
    "mobile_phone": "휴대전화",
    "electric_iron": "전기다리미",
    "electric_rice_cooker": "전기밥솥",
    "electric_fan": "선풍기",
    "vacuum_cleaner": "청소기",
    "beverage_carton": "음료수곽",
}
