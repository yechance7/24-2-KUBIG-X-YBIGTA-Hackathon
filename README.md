# 24-2-YBIGTA_hackathon

## 📁 프로젝트 폴더 구조

```plaintext
24-2-YBIGTA_hackathon/
│-- README.md
│-- EDA.ipynb
├── data/
│   │-- avazu_test_data.csv
│   │-- avazu_train_data.csv
│   │-- sample_submission_hackathon.csv
├── model/
│   ├── boosting/                  
│   │   │-- catboost_baseline.py   
│   │   │-- LGBM.py  
│   │   │-- LGBM_config.yaml     
│   │   └──      
├── ensemble/
│   │-- ensemble.py
│   │-- similarity.py
│   ├── before/ # 앙상블 할 csv파일 위치
│   ├── after/
├── convert/ # 회귀문제 결과를 이진분류 결과로 바꿔줌
│   │-- convert.py # 
│   ├── before/ # 바꿔줄 csv파일 위치
│   ├── after/
├── submission/ # 결과물들
├── bestmodel/ # 결과물들중 좋은 앙상블 재료들


