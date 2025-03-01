#데이터 불러오기
# Dataset 1: Seoul Bike
import pandas as pd
dia = pd.read_csv("Diabetes.csv")
print(dia.info())

#Q3. 데이터 통계량 계산

import scipy.stats as stats
skewness = dia.apply(stats.skew)
kurtosis = dia.apply(stats.kurtosis)

# 통계량 데이터프레임 생성
statistics_df = pd.DataFrame({
    'Mean': dia.mean(),
    'Std Dev': dia.std(),
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

# 통계량 출력
print(statistics_df)


import matplotlib.pyplot as plt
# Outcome(당뇨병 발생 여부)를 제외한 Numerical 변수에 대해 각 열에 대한 상자 그림 그리기
for column in dia.columns[:8]:
    plt.figure(figsize=(6, 4))
    dia[column].plot(kind='box')
    plt.title(f'Box plot of {column}')
    plt.show()


#Q4. 이상치 제거
def remove_outliers(df):

    for col in df.columns[:8]: #Outcome(당뇨병 유무)제외한 열에 대해 이상치 제거 수행
        IQR=df[col].quantile(0.75)-df[col].quantile(0.25)
        lower_bound = df[col].quantile(0.25)-1.5*IQR
        upper_bound = df[col].quantile(0.75)+1.5*IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df #이상치를 제거한 수정된 데이터 프레임 반환

# remove_outliers 함수를 호출하여 이상치를 제거한 새로운 데이터프레임을 생성
new_data = remove_outliers(dia)
new_data=new_data.dropna() #이상치 제거한 데이터프레임의 결측치 제거

# 이상치 제거 전 후의 데이터프레임 출력
print(f'Before remove outliers:\n{dia}')
print(f'\nAfter remove outliers:\n{new_data}')
print(new_data.head(5))


#Q5. 상관성 분석
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))
sns.set(style='whitegrid') #그래프의 배경을 흰색, 격자 라인이 있는 스타일로 지정
sns.pairplot(new_data) #각 열의 조합에 대한 산점도 그리기
plt.show()


#heatmap
corr= new_data.corr()
# 대각 성분 아래만 마스크 생성
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(20, 16), dpi=200)
sns.heatmap(corr, mask=mask, fmt='.2f', annot=True, annot_kws={"size": 10})
plt.title('Correlation of Variable')
plt.show()


#데이터 표준화 전처리
from sklearn.preprocessing import scale

dia_input_scaled= scale(new_data.iloc[:,:8])
dia_target = new_data.iloc[:,-1] #종속 변수


#1. Age 제외하고 Logistic Regression 수행
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
seed=12345
test_size=0.3


# 설명 변수들에서 Age 변수 제거
dia_input_scaled_age_removed = dia_input_scaled[:, :-1]

# 모델 수행을 위해 train-test split
dia_trn_age_removed, dia_test_age_removed = train_test_split(dia_input_scaled_age_removed, test_size=test_size, random_state=seed)
y_trn, y_test = train_test_split(dia_target, test_size=test_size, random_state=seed)

# 모델 설정
full_lr_age_removed = LogisticRegression(solver='liblinear', random_state=seed, max_iter=int(1e+5))

# trainset으로 모델 훈련
full_lr_age_removed.fit(dia_trn_age_removed, y_trn)

# 각 feature의 p-value 확인
x_trn_with_const_age_removed = sm.add_constant(dia_trn_age_removed)
logit_model_age_removed = sm.Logit(y_trn, x_trn_with_const_age_removed)
result_age_removed = logit_model_age_removed.fit()

# 설명 변수들의 이름 가져오기
variable_names = new_data.columns[:-2].tolist()

#결과 출력
print("Age 변수 제외한 경우:")
print(result_age_removed.summary(xname=['Intercept']+variable_names))

# 설명 변수들에서 Pregnancies 변수 제거
dia_input_scaled_preg_removed = dia_input_scaled[:, 1:]

# 모델 수행을 위해 train-test split
dia_trn_preg_removed, dia_test_preg_removed = train_test_split(dia_input_scaled_preg_removed, test_size=test_size, random_state=seed)
y_trn, y_test = train_test_split(dia_target, test_size=test_size, random_state=seed)

# 모델 설정
full_lr_preg_removed = LogisticRegression(solver='liblinear', random_state=seed, max_iter=int(1e+5))

# trainset으로 모델 훈련
full_lr_preg_removed.fit(dia_trn_preg_removed, y_trn)

# 각 feature의 p-value 확인
x_trn_with_const_preg_removed = sm.add_constant(dia_trn_preg_removed)
logit_model_preg_removed = sm.Logit(y_trn, x_trn_with_const_preg_removed)
result_preg_removed = logit_model_preg_removed.fit()

# 설명 변수들의 이름 가져오기
variable_names = new_data.columns[1:-1].tolist()

#결과 출력
print("Pregnancies 변수 제외한 경우:")
print(result_preg_removed.summary(xname=['Intercept']+variable_names))

#6 LLR 모델 학습
#Q 6-1.
# 모델 수행을 위해 train-test split
dia_trn, dia_test = train_test_split(dia_input_scaled, test_size=test_size, random_state=seed)
y_trn, y_test = train_test_split(dia_target, test_size=test_size, random_state=seed)

# 모델 설정
full_lr= LogisticRegression(solver='liblinear', random_state=seed, max_iter=int(1e+5))

# trainset으로 모델 훈련
full_lr.fit(dia_trn, y_trn)

# 각 feature의 p-value 확인
x_trn_with_const= sm.add_constant(dia_trn)
logit_model = sm.Logit(y_trn, x_trn_with_const)
result = logit_model.fit()

# 설명 변수들의 이름 가져오기
variable_names = new_data.columns[:-1].tolist()

#결과 출력
print("모든 변수를 사용한 경우:")
print(result.summary(xname=['Intercept']+variable_names))


#Q 6-3. Confusion matrix 생성
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Union

# 학습된 모델의 클래스 레이블 확인
classes = full_lr.classes_

# 양성 클래스 확인
positive_class = classes[1]

print("양성 클래스:", positive_class)

# 성능 평가 함수
def perf_eval_clf(y_true: Union[np.array,list], y_pred: Union[np.array,list])->list:
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    TPR = round(tp / (tp + fn), 4)
    PRE = round(tp / (tp + fp), 4)
    TNR = round(tn / (fp + tn), 4)
    ACC = round((tp + tn) / (tn + fp + fn + tp), 4)
    BCR = round(np.sqrt(TPR * TNR), 4)
    F1 = round(2 * TPR * PRE / (TPR + PRE), 4)
    return [TPR, PRE, TNR, ACC, BCR, F1]

# 학습 데이터셋에 대한 예측
y_train_pred = full_lr.predict(dia_trn)

# 테스트 데이터셋에 대한 예측
y_test_pred = full_lr.predict(dia_test)

# 학습 데이터셋의 confusion matrix 및 성능 평가
print("학습 데이터셋:")
cm_train = confusion_matrix(y_true=y_trn, y_pred=y_train_pred)
perf_train = perf_eval_clf(y_trn, y_train_pred)
print("Confusion Matrix:")
print(cm_train)
print("성능 평가 결과:", perf_train)

# 테스트 데이터셋의 confusion matrix 및 성능 평가
print("\n테스트 데이터셋:")
cm_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
perf_test = perf_eval_clf(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm_test)
print("성능 평가 결과", perf_test)

#Q 6-4.
def calculate_auroc(y_true, y_score):
    sorted_index=np.argsort(y_score)[::-1]
    y_true=np.asarray(y_true)[sorted_index]
    n_pos=np.sum(y_true) #양성 클래스의 개수
    n_neg=len(y_true)-n_pos #음성 클래스의 개수
    tpr_list, fpr_list=[0], [0] #tpr, fpr 저장할 리스트
    tp, fp=0, 0
    for i in range(len(y_true)): #정렬된 요소에 대해 반복 수행
        if y_true[i] == 1:
            tp+=1
        else:
            fp+=1
        tpr_list.append(tp/n_pos)
        fpr_list.append(fp/n_neg)
    return np.trapz(tpr_list, fpr_list)  #TPR 및 FPR 사이의 면적을 계산하여 AUROC 값 반환

# 훈련 데이터와 테스트 데이터에 대한 예측 점수 계산
y_train_score = full_lr.predict_proba(dia_trn)[:, 1]
y_test_score = full_lr.predict_proba(dia_test)[:, 1]

# 훈련 데이터와 테스트 데이터에 대한 AUROC 값 계산
auroc_train = round(calculate_auroc(y_trn, y_train_score), 4)
auroc_test = round(calculate_auroc(y_test, y_test_score), 4)

# AUROC 값 출력
print("훈련 데이터의 AUROC 값:", auroc_train)
print("테스트 데이터의 AUROC 값:", auroc_test)

#ROC Curve 그리기
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# 훈련 데이터와 테스트 데이터에 대한 FPR, TPR, 임계값 계산
fpr_train, tpr_train, thresholds_train = roc_curve(y_trn, y_train_score)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_score)

# ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label='Training ROC Curve')
plt.plot(fpr_test, tpr_test, label='Test ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Random Classifier 직선
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

#7
# 학습 데이터셋에 대한 예측
y_train_pred_preg_removed = full_lr_preg_removed.predict(dia_trn_preg_removed)

# 테스트 데이터셋에 대한 예측
y_test_pred_preg_removed = full_lr_preg_removed.predict(dia_test_preg_removed)

# 학습 데이터셋의 confusion matrix 및 성능 평가
print("Q 7-3 학습 데이터셋:")
cm_train_preg_removed = confusion_matrix(y_true=y_trn, y_pred=y_train_pred_preg_removed)
perf_train_preg_removed = perf_eval_clf(y_trn, y_train_pred_preg_removed)
print("Confusion Matrix:")
print(cm_train_preg_removed)
print("성능 평가 결과:", perf_train_preg_removed)

# 테스트 데이터셋의 confusion matrix 및 성능 평가
print("\nQ 7-3. 테스트 데이터셋:")
cm_test_preg_removed = confusion_matrix(y_true=y_test, y_pred=y_test_pred_preg_removed)
perf_test_preg_removed = perf_eval_clf(y_test, y_test_pred_preg_removed)
print("Confusion Matrix:")
print(cm_test_preg_removed)
print("성능 평가 결과", perf_test_preg_removed)

# AUROC 값 도출
# 훈련 데이터와 테스트 데이터에 대한 예측 점수 계산
y_train_score = full_lr_preg_removed.predict_proba(dia_trn_preg_removed)[:, 1]
y_test_score = full_lr_preg_removed.predict_proba(dia_test_preg_removed)[:, 1]

# 훈련 데이터와 테스트 데이터에 대한 AUROC 값 계산
auroc_train = round(calculate_auroc(y_trn, y_train_score), 4)
auroc_test = round(calculate_auroc(y_test, y_test_score), 4)

# AUROC 값 출력
print("Pregnancies 변수 제외한 경우:")
print("훈련 데이터의 AUROC 값:", auroc_train)
print("테스트 데이터의 AUROC 값:", auroc_test)

# ROC Curve 그리기
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# 훈련 데이터와 테스트 데이터에 대한 FPR, TPR, 임계값 계산
fpr_train, tpr_train, thresholds_train = roc_curve(y_trn, y_train_score)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_score)

# ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label='Training ROC Curve')
plt.plot(fpr_test, tpr_test, label='Test ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Random Classifier 직선
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()


#8

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import confusion_matrix  # confusion_matrix 함수 추가
import time
import numpy as np

# 전체 데이터셋 스플릿
dia_trn, dia_test, y_trn, y_test = train_test_split(dia_input_scaled, dia_target, test_size=test_size, random_state=seed)


#8-1. Forward #트레이닝 auroc 걸린 시간
full_config = {
    'penalty':None,
    'fit_intercept':True,
    'max_iter':int(1e+5),
    'solver':'saga',
    'random_state':seed,
    'n_jobs':-1
}
full_model = LogisticRegression(**full_config)

# 특성의 이름 가져오기
feature_names = dia.columns[:-1] #종속 변수 outcome 제외 변수 가져오기

# 순차적 특성 선택기 설정
forward_config = {
    'estimator': full_model,
    'n_features_to_select': 'auto',
    'tol': 1e-4,
    'direction': 'forward',
    'scoring': 'f1',
    'cv': 5,
    'n_jobs': -1
}

start_time_forward = time.time()  # forward selection 시작 시간 기록
forward_selection = SFS(**forward_config).fit(dia_trn, y_trn)
end_time_forward = time.time()  # forward selection 종료 시간 기록
elapsed_time_forward = end_time_forward - start_time_forward  # forward selection 걸린 시간 계산

selected_indices = [i for i, val in enumerate(forward_selection.get_support()) if val == True]
selected_features = [feature_names[i] for i in selected_indices]

print("Selected Features:", selected_features)
print("Time taken for forward selection:", round(elapsed_time_forward,4), "seconds")


def calculate_auroc(y_true, y_score):
    sorted_index = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[sorted_index]
    n_pos = np.sum(y_true)  # 양성 클래스의 개수
    n_neg = len(y_true) - n_pos  # 음성 클래스의 개수
    tpr_list, fpr_list = [0], [0]  # tpr, fpr 저장할 리스트
    tp, fp = 0, 0
    for i in range(len(y_true)):  # 정렬된 요소에 대해 반복 수행
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    return np.trapz(tpr_list, fpr_list)  # TPR 및 FPR 사이의 면적을 계산하여 AUROC 값 반환


# Training Logistic Regression with selected features
x_trn_forward = dia_trn[:, selected_indices]
x_tst_forward = dia_test[:, selected_indices]

forward_logit = LogisticRegression(**full_config).fit(x_trn_forward, y_trn)
forward_logit2 = LogisticRegression(**full_config).fit(x_tst_forward, y_test)

y_pred_trn = forward_logit.predict(x_trn_forward)
y_pred_tst = forward_logit2.predict(x_tst_forward)

auroc_tr = round(calculate_auroc(y_trn, y_pred_trn),4)  # AUROC 계산
print("Training AUROC:", auroc_tr)


from typing import Union
# 테스트 데이터셋의 성능 평가
def perf_eval_clf(y_true: Union[np.array,list], y_pred: Union[np.array,list])->list:
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    TPR = round(tp / (tp + fn), 4)
    PRE = round(tp / (tp + fp), 4)
    TNR = round(tn / (fp + tn), 4)
    ACC = round((tp + tn) / (tn + fp + fn + tp), 4)
    BCR = round(np.sqrt(TPR * TNR), 4)
    F1 = round(2 * TPR * PRE / (TPR + PRE), 4)
    return [TPR, PRE, TNR, ACC, BCR, F1]


#테스트 데이터 셋
print("\n테스트 데이터셋:")
auroc_test = round(calculate_auroc(y_test, y_pred_tst),4)  # AUROC 계산
print("Test AUROC:", auroc_test)

perf_test = perf_eval_clf(y_test, y_pred_tst)
print("성능 평가 결과", perf_test)


#8-2. Backward
# 특성의 이름 가져오기
feature_names = dia.columns[:-1] #종속 변수 outcome 제외 변수 가져오기

full_config = {
    'penalty':None,
    'fit_intercept':True,
    'max_iter':int(1e+5),
    'solver':'saga',
    'random_state':seed,
    'n_jobs':-1
}
full_model = LogisticRegression(**full_config)

print("\nBackward")
backward_config = {
'estimator':full_model,
'n_features_to_select':'auto',
'tol':int(1e-3),
'direction':'backward',
'scoring':'f1',
'cv':5,
'n_jobs':-1
}

start_time_backward = time.time()
backward_selection = SFS(**backward_config).fit(dia_trn, y_trn)
end_time_backward = time.time()
elapsed_time_backward = end_time_backward - start_time_backward

selected_indices_backward = [i for i, val in enumerate(backward_selection.get_support()) if val == True]
selected_features_backward = [feature_names[i] for i in selected_indices_backward]

print("\nBackward Selected Features:", selected_features_backward)
print("Time taken for backward selection:", round(elapsed_time_backward, 4), "seconds")

x_trn_backward = dia_trn[:, selected_indices_backward]
x_tst_backward = dia_test[:, selected_indices_backward]

backward_logit = LogisticRegression(**full_config).fit(x_trn_backward, y_trn)
backward_logit2 = LogisticRegression(**full_config).fit(x_tst_backward, y_test)

y_pred_trn = backward_logit.predict(x_trn_backward)
y_pred_tst = backward_logit2.predict(x_tst_backward)

auroc_tr = round(calculate_auroc(y_trn, y_pred_trn),4)  # AUROC 계산
print("Training AUROC:", auroc_tr)

#테스트 데이터 셋
print("\n테스트 데이터셋:")
auroc_test = round(calculate_auroc(y_test, y_pred_tst),4)  # AUROC 계산
print("Test AUROC:", auroc_test)

perf_test = perf_eval_clf(y_test, y_pred_tst)
print("성능 평가 결과", perf_test)


#8-3. Stepwise
print("\nStepwise")
stepwise_config = {
'estimator':full_model,
'n_features_to_select':'auto',
'tol':None,
'direction':'forward',
'scoring':'f1',
'cv':5,
'n_jobs':-1
}
start_time_stepwise = time.time()  # stepwise selection 시작 시간 기록
stepwise_selection = SFS(**stepwise_config).fit(dia_trn, y_trn)
end_time_stepwise = time.time()  # stepwise selection 종료 시간 기록
elapsed_time_stepwise = end_time_stepwise - start_time_stepwise  # stepwise selection걸린 시간 계산

selected_indices_stepwise = [i for i, val in enumerate(stepwise_selection.get_support()) if val == True]
selected_features_stepwise= [feature_names[i] for i in selected_indices]

print("Selected Features:", selected_features)
print("Time taken for backward selection:", round(elapsed_time_stepwise,4), "seconds")

# Training Logistic Regression with selected features
x_trn_stepwise = dia_trn[:, selected_indices_stepwise]
x_tst_stepwise = dia_test[:, selected_indices_stepwise]

stepwise_logit = LogisticRegression(**full_config).fit(x_trn_stepwise, y_trn)
stepwise_logit2 = LogisticRegression(**full_config).fit(x_tst_stepwise, y_test)

y_pred_trn = stepwise_logit.predict(x_trn_stepwise)
y_pred_tst = stepwise_logit2.predict(x_tst_stepwise)

auroc_tr = round(calculate_auroc(y_trn, y_pred_trn),4)  # AUROC 계산
print("Training AUROC:", auroc_tr)

#테스트 데이터 셋
print("\n테스트 데이터셋:")
auroc_test = round(calculate_auroc(y_test, y_pred_tst),4)  # AUROC 계산
print("Test AUROC:", auroc_test)

perf_test = perf_eval_clf(y_test, y_pred_tst)
print("성능 평가 결과", perf_test)

