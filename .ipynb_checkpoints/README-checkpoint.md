# **용인지역 수위 예측 프로젝트 rebuild**
### **Deployment & Documentation & License**
<p align="left">
<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></a>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a> 
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>
<img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white"/></a></p> 

## 1. Introduce 
    
- **프로젝트명** : 

    🎯 **목표** : 용인시 내 유량계 설치 지점별 수위 예측


## 2. Structure 
<pre>
├── 📁 <b>data</b>
│   ├── 📁 <b>preprocessed</b>: 전처리 완료 데이터 
│   │   ├── 용인_강우량계_설치정보.csv
│   │   ├── 용인_하수도유량계_설치정보_위경도포함.csv
│   │   ├── 용인_강우량계_데이터(분단위)_201901-202207.parquet
│   │   └── 용인_하수도유량계_데이터_지역통합.parquet
│   └── 📁 <b>raw</b>: 원천 데이터 
│       ├── 📁 용인_기상데이터
│       │   ├── 용인_강우량계_데이터(분단위)
│       │   └── ...
│       └── 📁 용인_상하수도_데이터
│           ├── 📁 상수도데이터
│           │   ├── 상수_유량및수압데이터_지역별(분단위)
│           │   ├── 용인_상수도_유량계실,수압계실_설치정보.xlsx
│           │   └── 용인_상수도사용량-2017-2021.xlsx
│           └── 📁 하수도데이터
│               ├── 하수_유량계데이터_지역별(분단위)
│               └── 용인_하수도유량계_설치정보.csv
├── 📁 <b>src</b>: 전처리 및 모델 생성 전 과정
│   ├── model.py
│   └── preprocessing.ipynb
├── requirements.txt
└── README.md
</pre>

---
이하 수정하지 않음

## 3. Installation 
### Dependencies
- Conda 22.9.0
- Python 3.9.13

### User installation
|프로그램|다운로드 주소|
|:---:|:---:|
|Python|[**<U>Link</U>**](https://www.python.org/downloads/)|
|Anaconda|[**<U>Link</U>**](https://www.anaconda.com/products/distribution)|
※설치 후 Anaconda Prompt 실행해야 함

<br>
1) conda 가상환경 생성

~~~python
# [옵션] somang 위치에 원하는 환경명 입력
conda create -n somang python=3.9.13   
conda activate somang
pip install -U pip
~~~
2) 프로젝트 폴더 다운로드
~~~python
# 원하는 위치에 설치 레포지토리 clone(예시)
## 최상위 폴더로 이동
cd /    
## somang_project 폴더 생성
mkdir somang_project    
cd somang_project
git clone https://gitlab.com/Ashbee/somang.git
~~~

3) 관련 패키지 설치 및 경로 설정
~~~python
pip install -r requirements.txt  
python setup.py install   
# 패키지를 수정하며 작업하는 경우 아래 명령어 실행
# python setup.py develop
~~~

## 4. Research 
모델 생성 및 평가, *Anomaly Detection 모델 위주 사용
### 이상탐지(Anomaly Detection)란?
- **접목 배경** 

   - 일반적인 머신러닝 분류의 경우, 분류하려는 대상의 데이터 수가 일정 수준 확보되어야 함(목표값의 비율이 다소 언밸런스한 경우에도 Data Augumentation을 적용해 해소 가능).
   
   - 하지만 금융 사기거래, 제조 공장에서의 결품 발생 등의 사례에서는 분류 대상의 데이터 비율에 극단적으로 큰 차이가 존재함. 따라서 Data Augumentation도 적용할 수 없으며, 일반적인 분류 문제로 해결하기 어려움

- **이상탐지 모델링의 특장점 및 사례**
   - 특장점: 데이터가 많은 쪽을 학습해서 극소수의 데이터로 이루어진 이상 데이터를 찾는 데 효과적

   - 예시: Probabilistic, Neural Network Base 모델 및 두 가지 방법이 결합된 모델이 있으며, ECOD, COPOD, VAE, DAGMM 등이 대표적임

### *References

|Model|Type|Algorithm|Year|Ref|
|:---:|:---:|:---|:---:|:---:|
|ECOD|Probabilistic|Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions|2022|[**<U>Link</U>**](https://arxiv.org/pdf/2201.00382.pdf)|
|SUOD|Outlier Ensembles|SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection (Acceleration)|2021|[**<U>Link</U>**](https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf)|
|COPOD|Probabilistic|COPOD: Copula-Based Outlier Detection|2020|[**<U>Link</U>**](https://www.andrew.cmu.edu/user/yuezhao2/papers/20-icdm-copod.pdf)|
|DAGMM|Autoencoder & Probabilistic|DEEP AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION|2018|[**<U>Link</U>**](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)|
|VAE|Autoencoder|Auto-Encoding Variational Bayes|2014|[**<U>Link</U>**](https://arxiv.org/pdf/1312.6114.pdf)|


## 5. Research Code
### 우울증 탐지모델
~~~python
#cd src // source folder로 이동 후 실행 
python depress.py
~~~
Example : 데이터 위치 변경, 전처리(시간변수 생성, 정규화), AE모델 선택 및 하이퍼파라미터 변경
~~~python
python depress.py --time_features True --normalize std --model ae --epochs 5 --dropout_rate 0.3
~~~
---

### 위급상황 탐지모델
~~~python
#cd src // source folder로 이동 후 실행 
python emergency.py
~~~
Example : 파일 Path 지정(우울증 탐지모델에서 생성된 Anomaly Score), VAE모델 선택 및 하이퍼파라미터 변경
~~~python
python emergency.py --model vae --epochs 5 --dropout_rate 0.3
~~~
**모델 학습 및 평가 config 정보 확인 : [<U>우울증 탐지모델</U>](src/depress.py), [<U>위급상황 탐지모델</U>](src/emergency.py)**

---

#### 주요 모델링 Jupyter Notebook
|Model|우울증 탐지 모델 튜토리얼|위급상황 탐지 모델 튜토리얼|
|:---:|:---:|:---:|
|VAE|[**<U>Code</U>**](examples/4_model/detect_depress/model_vae.ipynb)|[**<U>Code</U>**](examples/4_model/detect_emergency/model_vae.ipynb)|
|DAGMM|[**<U>Code</U>**](examples/4_model/detect_depress/model_dagmm.ipynb)|[**<U>Code</U>**](examples/4_model/detect_emergency/model_dagmm.ipynb)|
|**ECOD**|[**<U>Code</U>**](examples/4_model/detect_depress/model_ecod.ipynb)|[**<U>Code</U>**](examples/4_model/detect_emergency/model_ecod.ipynb)|
|**COPOD**|[**<U>Code</U>**](examples/4_model/detect_depress/model_copod.ipynb)|[**<U>Code</U>**](examples/4_model/detect_emergency/model_copod.ipynb)|
|**LGBM**|[**<U>Code</U>**](examples/4_model/detect_depress/model_lgbm.ipynb)|X|


※ 전처리, 모델 평가에 대한 세부 코드는 [**<U>src</U>**](src)폴더 내에서 확인 가능

➕ *`2022.08.31` 우울증, 정상 데이터 모두 충분하다고 판단 → 이진분류(Binary Classfication) 모델 추가로 적용 
<br>
## 6. Service

- 우울증 탐지, 위급상황 탐지 모델을 사용한 추론 결과를 공유하는 웹 페이지
- [**<U>상세보기</U>**<-클릭](webservice/README.md)


## 7. labq_solution 

- 프로젝트 수행을 위해 사용한 패키지
    - [Source Code](labq)
    - [Documentation](docs/_build/html/index.html/)
<br><br>

## 8. Project History
2022년 AI바우처 사업 의료분과 `디지털 바이오마커 기반 AI 비대면 정신건강 케어 서비스`

- 협약 체결 : 2022년 4월
- 데이터 수집 및 모델 개발 : 2022년 4월 ~ 2022년 9월
- 서비스 개발 : 2022년 9월 ~ 2022년 10월
- 사업 종료 : 2022년 10월