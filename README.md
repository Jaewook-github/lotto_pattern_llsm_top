## DIRECTORY
```
G:lotto_pattern_llsm_top_number
│  config.py # 상수와 필터링 설정
│  lotto.db # 로또 데이터 베이스
│  lotto_analyzer.py # 로또 분석 기법
│  lotto_generator.py # 예측 번호 생성(게임당 6개 번호)
│  lotto_generator_old.py
│  lotto_pattern_llsm.py
│  main.py # 머신러닝과 학습을 통한 번호 생성(번호 6 ~ 45개 생성)
│  number_test.py # 생성된 예측번호를 필터링 조건에 맞는지 검증
│  README.md
│
├─backups
├─logs # main.py 실행시 로그 파일 생성
│      lotto_prediction_20250222-1.log
│
├─models
├─modules # 머신러닝에 필요한 모듈들
│  │  database_manager.py
│  │  file_manager.py
│  │  gui.py
│  │  log_manager.py
│  │  lotto_analyzer.py
│  │  lotto_analyzer_old.py
│  │  model_state.py
│  └─ __init__.py
│
└─predictions
   └─ lotto_prediction_20250222_104145.txt
```
# 로또 패턴 분석 및 예측 시스템 v3.0

## 목차
1. [개요](#1-개요)
2. [시스템 구성](#2-시스템-구성)
3. [주요 기능](#3-주요-기능)
4. [설정 가이드](#4-설정-가이드)
5. [패턴 분석 방법](#5-패턴-분석-방법)
6. [예측 알고리즘](#6-예측-알고리즘)
7. [결과 분석](#7-결과-분석)
8. [기술 상세](#8-기술-상세)
9. [주의사항](#9-주의사항)
10. [부록](#10-부록)

## 1. 개요

### 1.1 시스템 소개
이 시스템은 과거 로또 당첨 번호의 패턴을 분석하여 미래의 번호를 예측하는 프로그램입니다. 
다양한 통계적 방법과 패턴 인식 알고리즘을 사용하여 번호를 선택합니다.

### 1.2 주요 특징
- 패턴 기반 번호 선택
- 사용자 정의 가중치 시스템
- 상세한 분석 및 시각화
- 데이터베이스 기반 이력 관리
- 결과 저장 및 분석 기능

### 1.3 시스템 요구사항
- Python 3.8 이상
- 필요 라이브러리:
  - tkinter
  - pandas
  - numpy
  - matplotlib
  - sqlite3
  - scikit-learn

## 2. 시스템 구성

### 2.1 핵심 컴포넌트
1. **분석 엔진**
   - LottoAnalyzer: 번호 패턴 분석
   - PredictionAnalyzer: 예측 결과 분석
   - LearningAnalyzer: 학습 과정 분석

2. **데이터 관리**
   - DatabaseManager: 데이터베이스 연동
   - FileManager: 파일 관리
   - LogManager: 로깅 시스템

3. **사용자 인터페이스**
   - LottoPredictionGUI: 메인 GUI
   - VisualizationManager: 시각화 관리
   - ResultVisualizer: 결과 표시

### 2.2 데이터 구조
```python
class ModelState:
    """모델 상태 관리 클래스"""
    numbers_memory: Dict[int, float]  # 번호별 가중치
    number_stats: Dict[int, int]      # 번호별 통계
    pattern_history: Dict[tuple, list] # 패턴 이력
```

## 3. 주요 기능

### 3.1 패턴 분석 시스템
#### 3.1.1 기본 패턴 분석
- **2개 번호 패턴**
 - 두 번호가 함께 출현하는 빈도 분석
 - 가중치: 35%
 - 분석 방법: 연속된 두 번호의 출현 빈도 및 최근성 계산

- **3개 번호 패턴**
 - 세 번호의 동시 출현 패턴 분석
 - 가중치: 25%
 - 특징: 더 복잡한 패턴 인식 가능

- **연속성 패턴**
 - 연속된 번호의 출현 분석
 - 가중치: 15%
 - 예: 1-2, 15-16과 같은 연속 번호

#### 3.1.2 고급 패턴 분석
```python
def analyze_patterns(self, historical_data: pd.DataFrame):
   """
   패턴 분석 수행
   - historical_data: 과거 당첨 번호 데이터
   - 반환: 패턴 분석 결과
   """
   for idx, row in historical_data.iterrows():
       numbers = sorted([row[f'num{i}'] for i in range(1, 7)])
       
       # 기본 통계 업데이트
       self._update_basic_stats(numbers)
       
       # 패턴 분석
       self._analyze_pair_patterns(numbers)
       self._analyze_triple_patterns(numbers)
       self._analyze_sequence_patterns(numbers)
```
### 3.2 가중치 시스템
#### 3.2.1 가중치 구성
```Python
pattern_weights = {
    'pair': 0.35,      # 2개 번호 패턴
    'triple': 0.25,    # 3개 번호 패턴
    'sequence': 0.15,  # 연속성
    'frequency': 0.15, # 출현 빈도
    'recency': 0.10    # 최근성
}
```
#### 3.2.2 가중치 조정
- 사용자 정의 가능
- 자동 정규화 기능
- 유효성 검사 시스템

### 3.3 시각화 시스템
#### 번호별 출현 빈도 그래프
- 막대 그래프 형태
- 평균선 표시
- 표준편차 영역 표시
#### 패턴 히트맵
- 번호 간 연관성 시각화
- 컬러 스케일로 강도 표현
#### 트렌드 분석 그래프
- 시계열 데이터 분석
- 이동평균선 표시

## 4. 설정 가이드

### 4.1 기본 설정
#### 4.1.1 번호 선택 설정
```python
# 설정 예시
settings = {
   "선택 번호 개수": 6,     # 기본값, 1-45 사이 설정 가능
   "학습 회차": 100,       # 전체/10/30/50/100/200 선택 가능
   "학습률": 0.1,         # 0-1 사이 값
   "학습 반복": 100       # 반복 학습 횟수
}
```
#### 4.1.2 권장 설정값
| 설정 항목 |권장값|설명|
|-------|---|---|
| 학습 회차 |100|최근 100회차 데이터 사용|
| 학습률   |0.1|안정적인 학습을 위한 값|
| 학습 반복 |100|충분한 학습을 위한 반복 횟수|

### 4.2 패턴 가중치 설정
#### 4.2.1 세부 가중치 조정
|패턴 유형|기본값|권장 범위|영향|
|-----|---|---|---|
|2개 번호|35%|30-40%|기본 패턴 인식|
|3개 번호|25%|20-30%|복합 패턴 인식|
|연속성|15%|10-20%|연속 번호 출현|
|출현 빈도|15%|10-20%|자주 나오는 번호|
|최근성|10%|5-15%|최근 추세 반영|

### 4.3 분석 설정
#### 4.3.1 패턴 인식 임계값
```Python
thresholds = {
    "패턴 강도 임계값": 0.8,    # 상위 20% 패턴만 선택
    "연속성 임계값": 2,        # 최대 연속 번호 개수
    "최소 출현 빈도": 5        # 최소 출현 횟수
}
```

## 5. 패턴 분석 방법

### 5.1 기본 패턴 분석
#### 5.1.1 출현 빈도 분석
```Python
def analyze_frequency(self, numbers: list) -> dict:
    """번호별 출현 빈도 분석"""
    frequency = {
        'total_count': len(numbers),
        'number_counts': {},
        'statistics': {
            'mean': np.mean(numbers),
            'std': np.std(numbers),
            'range': max(numbers) - min(numbers)
        }
    }
    return frequency
```

#### 5.1.2 연속성 분석
```Python
def analyze_sequence(self, numbers: list) -> list:
    """연속 번호 패턴 분석"""
    sequences = []
    for i in range(len(numbers)-1):
        if numbers[i+1] - numbers[i] == 1:
            sequences.append((numbers[i], numbers[i+1]))
    return sequences
```

### 5.2 고급 패턴 분석
#### 5.2.1 구간별 분석
- 번호를 구간별로 나누어 분석
    - 1-10, 11-20, 21-30, 31-40, 41-45
- 각 구간의 출현 비율 계산
- 구간 조합 패턴 분석

## 6. 예측 알고리즘

### 6.1 기본 예측 프로세스
```python
class LottoAnalyzer:
   def select_numbers_by_count(self, count: int) -> list:
       """
       패턴 기반 번호 선택 프로세스
       
       Args:
           count (int): 선택할 번호 개수
       
       Returns:
           list: 선택된 번호 리스트
       """
       selected = []
       while len(selected) < count:
           scores = self._calculate_number_scores(selected)
           next_num = self._select_next_number(scores)
           selected.append(next_num)
       return sorted(selected)
```


