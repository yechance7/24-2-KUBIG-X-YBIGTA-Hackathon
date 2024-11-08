import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import wandb
from wandb import config

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

# wandb 설정
wandb.init(project="avazu_click_prediction", config={
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "objective": "regression",  # 회귀 문제로 설정
    "metric": "l2"  # 회귀의 경우 L2 (MSE) 사용
})
config = wandb.config

print("###### LOAD MODEL ######")
# LightGBM 회귀 모델 초기화
model = lgb.LGBMRegressor(
    n_estimators=config.iterations,
    learning_rate=config.learning_rate,
    max_depth=config.depth,
    categorical_feature=categorical_features.tolist(),
    random_state=42
)

# 학습
print("###### TRAIN MODEL ######")
model.fit(X_train, y_train)

# 예측 및 평가
print("###### PREDICT MODEL ######")
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature Importance 출력
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
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
submission.to_csv(f'submission/{mse}_{current_time}_lgbm_regressor.csv', index=False)

wandb.log({"mse": mse})
wandb.finish()


# wandb sweep LGBM_config.yaml
# wandb agent <SWEEP_ID>
