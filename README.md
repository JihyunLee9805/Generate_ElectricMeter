# Generate_ElectricMeter

### 1) 목적: 계량기 숫자 부분의 화질이 뛰어난 가짜 전기 계량기 이미지 생성
### 2) 구현 방법: 
      1. DCGAN 모델로 흑백 숫자 이미지와 흑백 전기 계량기(숫자 부분이 검은 박스가 쳐진) 이미지를 생성
      2. Auto-Encoder와 Residual Block을 혼합한 모델로 가짜 흑백 계량기 이미지를 채색하여 컬러 전기 계량기 이미지 생성
       
### 3) 출판 논문: SSN 1975-8359 [Print] / ISSN 2287-4364 [Online]
The Transactions of the Korean Institute of Electrical Engineers, vol. 69, no. 12, pp. 1943~1949, 2020 https://doi.org/10.5370/KIEE.2020.69.12.1943
