import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from sklearn.utils import resample

# 현재 시간 구하기 (파일명에 사용할 시간)
current_time = time.strftime("%y%m%d_%H%M%S")

# 데이터 로딩
print("###### LOAD DATA ######")
train_data = pd.read_csv('data/avazu_train_data.csv')
test_data = pd.read_csv('data/avazu_test_data.csv')

# 범주형 변수 지정
categorical_features = ['banner_pos', 'site_id', 'site_domain', 'site_category', 
                        'app_id', 'app_domain', 'app_category', 'device_id', 
                        'device_ip', 'device_model', 'device_type', 'device_conn_type'] + [f'C{i}' for i in range(14, 22)]

# 클래스 0과 클래스 1으로 분리
train_data_class_0 = train_data[train_data['click'] == 0]
train_data_class_1 = train_data[train_data['click'] == 1]

# 클래스 0을 클래스 1의 개수에 맞게 언더샘플링
train_data_class_0_under = resample(train_data_class_0, 
                             replace=False,     # 중복 없이 샘플링
                             n_samples=len(train_data_class_1),  # 클래스 1의 개수에 맞게 샘플링
                             random_state=42)   # 재현성 있는 결과를 위해 시드 설정

# 샘플링된 데이터와 클래스 1 데이터를 합침
train_data_balanced = pd.concat([train_data_class_0_under, train_data_class_1])

# 섞기
train_data = train_data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(train_data['click'].value_counts()) 

# 언더 셈플링
train_data = train_data_balanced.sample(frac=0.2, random_state=42)
print(train_data['click'].value_counts()) 

# 데이터 전처리
print("###### PREPROCESSING DATA ######")
for col in categorical_features:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

X = train_data.drop(columns=[train_data.columns[0], 'click'])
y = train_data['click']
test_data = test_data.drop(columns=[test_data.columns[0]])

print("###### SPLIT DATA ######")
# 학습, 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("###### LOAD MODEL ######")
# LGBM 모델 초기화
model = LGBMRegressor(n_estimators=500, 
                      learning_rate=0.1, 
                      max_depth=6, 
                      categorical_feature=categorical_features,
                      random_state=42)

# 학습
print("###### TRAIN MODEL ######")
model.fit(X_train, y_train, categorical_feature=categorical_features)  

# 예측 및 평가
print("###### PREDICT MODEL ######")
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature Importance 출력
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance  # 인덱싱 제거
}).sort_values(by='Importance', ascending=False)

# Feature Importance 터미널에 출력
print("\nFeature Importance:")
print(importance_df)

# 테스트 데이터 예측
print("###### MAKING RESULT ######")
test_pred = model.predict(test_data)

# 제출 파일 작성
print("###### SUBMISSION ######")
submission = pd.DataFrame({'id': test_data['id'], 'click': test_pred})
submission.to_csv(f'{mse}submission/{current_time}_lgbm.csv', index=False)

# python model/boosting/LGBM_baseline.py