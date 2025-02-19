# config.py

class LottoConfig:
    # 상수 정의
    CORNER_NUMBERS = {
        '좌측 상단': [1, 2, 8, 9],
        '우측 상단': [6, 7, 13, 14],
        '좌측 하단': [29, 30, 36, 37, 43, 44],
        '우측 하단': [34, 35, 41, 42]
    }

    BALL_COLORS = {
        '노랑(🟡)': range(1, 11),      # 1-10
        '파랑(🔵)': range(11, 21),     # 11-20
        '빨강(🔴)': range(21, 31),     # 21-30
        '검정(⚫)': range(31, 41),     # 31-40
        '초록(🟢)': range(41, 46)      # 41-45
    }

    COMPOSITE_NUMBERS = {1, 4, 8, 10, 14, 16, 20, 22, 25, 26, 28, 32, 34, 35, 38, 40, 44}
    PERFECT_SQUARES = {1, 4, 9, 16, 25, 36}
    PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

    # 동형수 그룹 정의
    MIRROR_NUMBER_GROUPS = [
        {12, 21}, {13, 31}, {14, 41},
        {23, 32}, {24, 42}, {34, 43},
        {6, 9}
    ]

    # 배수 정의
    MULTIPLES = {
        '3의 배수': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
        '4의 배수': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        '5의 배수': [5, 10, 15, 20, 25, 30, 35, 40, 45]
    }

    # 조합 제외 기준
    FILTER_CRITERIA = {
        'sum_range': {'min': 100, 'max': 175},        # 총합구간
        'ac_range': {'min': 7, 'max': 10},            # AC값 범위
        'odd_even_exclude': [{0, 6}, {6, 0}],         # 제외할 홀짝 비율
        'high_low_exclude': [{0, 6}, {6, 0}],         # 제외할 고저 비율
        'same_last_digit': {'min': 2, 'max': 3},      # 동일 끝수 개수
        'last_digit_sum': {'min': 15, 'max': 35},     # 끝수 총합
        'consecutive_numbers': {
            'none': True,           # 연속번호 없음 허용
            'pairs': [1, 2]         # 허용되는 연속번호 쌍의 개수
        },
        'number_counts': {
            'prime_numbers': {'min': 1, 'max': 3},             # 소수 개수 범위
            'composite_numbers': {'min': 1, 'max': 3},         # 합성수 개수 범위
            'multiples_of_3': {'min': 1, 'max': 3},           # 3의 배수 개수 범위
            'multiples_of_4': {'min': 1, 'max': 2},           # 4의 배수 개수 범위
            'multiples_of_5': {'min': 0, 'max': 2},           # 5의 배수 개수 범위
            'double_numbers': {'min': 0, 'max': 2},           # 쌍수 개수 범위
            'corner_numbers': {'min': 1, 'max': 4}            # 모서리 번호 개수 범위
        },
        'number_range': {
            'start_number_max': 15,  # 시작번호 최대값
            'end_number_min': 35     # 끝번호 최소값
        },
        'section_numbers': {
            'min': 0,               # 구간별 최소 번호 개수
            'max': 3                # 구간별 최대 번호 개수
        },
        'colors': {
            'min': 3,               # 최소 사용 색상 개수
            'max': 5                # 최대 사용 색상 개수
        }
    }