#데이터 불러오기
# Dataset 1: Seoul Bike
import pandas as pd
import numpy as np
bike = pd.read_csv("SeoulBikeData.csv")
print(bike.info())

# Indices for the activated input variables
nBike = len(bike.index) #instane 개수
nVar = len(bike.columns) #변수 개수

#명목형 변수 가공

#1. 계절
dummy_sp = np.array([0]*nBike)
dummy_sm= np.array([0]*nBike)
dummy_a= np.array([0]*nBike)
dummy_w= np.array([0]*nBike)

sp_idx= bike['Seasons']=='Spring' #3~5월
sm_idx= bike['Seasons']=='Summer' #6~8월
a_idx= bike['Seasons']=='Autumn' #9~11월
w_idx= bike['Seasons']=='Winter' #12~2월

dummy_sp[sp_idx] = 1
dummy_sm[sm_idx] = 1
dummy_a[a_idx] = 1
dummy_w[w_idx] = 1

#2. 휴일
dummy_hd = np.array([0]*nBike)
dummy_nhd= np.array([0]*nBike)

hd_idx= bike['Holiday']=='Holiday'
nhd_idx= bike['Holiday']=='No Holiday'

dummy_hd[hd_idx] = 1
dummy_nhd[nhd_idx] = 1

#3.근무일
dummy_f = np.array([0]*nBike)
dummy_nf= np.array([0]*nBike)

f_idx= bike['Functioning Day']=='Yes'
nf_idx= bike['Functioning Day']=='No'
dummy_f[f_idx] = 1
dummy_nf[nf_idx] = 1

#Combine the dataset
season= pd.DataFrame({'Spring':dummy_sp,'Summer':dummy_sm,'Autumn':dummy_a,'Winter':dummy_w})
holiday= pd.DataFrame({'Holiday':dummy_hd,'No Holiday':dummy_nhd})
functioning_day= pd.DataFrame({'Functioning day':dummy_f,'Not Functioning day':dummy_nf})

# Input Variable Remove
total_variable_idx = [i for i in range(nVar)] #[0,1,2,…,13]
removal_variable_idx = [0,11,12,13] #0번째 인덱스: 날짜 /  나머지 인덱스:명목형 변수
selected_variable_idx = list(set(total_variable_idx)- set(removal_variable_idx))

# Prepare the data for MLR
bike_mlr_data = pd.concat((bike.iloc[:,selected_variable_idx], season, holiday, functioning_day), axis=1)

# 데이터프레임 정보 출력
print(bike_mlr_data.info())

#Q3. 데이터 통계량 계산

import scipy.stats as stats
skewness = bike_mlr_data.apply(stats.skew)
kurtosis = bike_mlr_data.apply(stats.kurtosis)

# 통계량 데이터프레임 생성
statistics_df = pd.DataFrame({
    'Mean': bike_mlr_data.mean(),
    'Std Dev': bike_mlr_data.std(),
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

# 통계량 출력
print(statistics_df)


import matplotlib.pyplot as plt
# Numerical 변수에 대해 각 열에 대한 상자 그림 그리기
for column in bike_mlr_data.columns[:9]:
    plt.figure(figsize=(6, 4))
    bike_mlr_data[column].plot(kind='box')
    plt.title(f'Box plot of {column}')
    plt.show()

# 정규분포 여부 판별
from scipy.stats import shapiro

test_stats=[]
p_vals=[]
normality=[]

for var in bike_mlr_data.columns:
    test_stat, p_val = shapiro(bike_mlr_data[var])
    test_stats.append(test_stat)
    p_vals.append(p_val)
    # p-value가 0.05보다 크면 정규분포를 따른다고 판단
    normality.append(p_val > 0.05)

shapiro_wilks_test = pd.DataFrame({'test_statistic': test_stats, 'p_value': p_vals, 'normality': normality}, index=bike_mlr_data.columns)
print(shapiro_wilks_test)

#Q4. 이상치 제거
def remove_outliers(df):
    new_df = pd.DataFrame()  # 비어 있는 새로운 데이터프레임 생성

    for col in df.columns:
        lower_bound = df[col].quantile(0.01)  # 하위 1% 값
        upper_bound = df[col].quantile(0.99)  # 상위 1% 값
        # 이상치 제거 후 새로운 데이터프레임에 추가
        filtered_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        new_df[col] = filtered_data[col]

    return new_df

# remove_outliers 함수를 호출하여 이상치를 제거한 새로운 데이터프레임을 생성
new_data = remove_outliers(bike_mlr_data)
new_data=new_data.dropna() #이상치 제거한 데이터프레임의 결측치 제거

# 이상치 제거 전 후의 데이터프레임 출력
print(f'Before remove outliers:\n{bike_mlr_data}')
print(f'\nAfter remove outliers:\n{new_data}')


#Q5. 상관성 분석
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.set(style='whitegrid') #그래프의 배경을 흰색, 격자 라인이 있는 스타일로 지정
sns.pairplot(new_data) #각 열의 조합에 대한 산점도 그리기
plt.show()


#heatmap
corr= new_data.corr()
plt.figure(figsize=(20, 16), dpi=200)
sns.heatmap(corr, fmt='.2f', annot=True, annot_kws={"size": 10})
plt.title('Correlation of Variable')
plt.show()


#Q6. MLR 모델 학습

#시간에 따른 bias 방지 위해 행 기준 섞음
from sklearn.utils import shuffle
seed=1234
shuffled_data=shuffle(new_data, random_state=seed)


from sklearn.model_selection import train_test_split
seed=1234
test_size=0.3
bike_trn_data, bike_valid_data=train_test_split(shuffled_data,test_size=test_size, random_state=seed)

#input, output variables in numpy array
x_trn=bike_trn_data.iloc[:,1:].to_numpy()
y_trn=bike_trn_data.iloc[:,0].to_numpy()

#model setting
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()

#Train the model
mlr.fit(x_trn, y_trn)

#Get R-Squared value
r_squared = mlr.score(x_trn, y_trn)
print("R-squared value:", r_squared)

#Q. 6-1 Adjusted R^2를 이용한 데이터 선형성 판단
import statsmodels.api as sm

X=bike_trn_data.iloc[:,1:]
X=sm.add_constant(X) #상수항 추가하여 절편 모델링
y=bike_trn_data.iloc[:,0]

result=sm.OLS(y, X)
result=result.fit()
print(result.summary())

#Q. 6-2
import seaborn as sns
import matplotlib.pyplot as plt

# 모델 예측
y_pred = result.predict(X)

# 설명변수와 종속변수의 선형성을 나타내는 plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], '--r') # Prediction line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

# 예측값이 0보다 작은 경우 0으로 처리
y_pred = np.where(y_pred < 0, 0, y_pred)

# 설명변수와 종속변수의 선형성을 나타내는 plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], '--r') # Prediction line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

# 잔차 계산
residuals = y - y_pred

# Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Residual Plot with adjusted x-axis ticks
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.xticks(np.arange(0, len(residuals), 1000))  # Adjust x-axis ticks
plt.show()

# Residual Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# QQ Plot
import scipy.stats as stats

plt.subplots(figsize=(12,6))
plt.title("QQ plot")
stats.probplot(residuals, dist=stats.norm, plot=plt)
plt.show()

#Q7
# 회귀분석 결과(summary)에서 p-value가 0.01보다 작은 변수들을 찾기
significant_vars = result.pvalues[result.pvalues < 0.01].index

# 유의수준에서 통계적으로 유의미한 변수 출력
print("Statistically significant variables at the significance level of 0.01:", significant_vars)

# 해당 변수들의 회귀계수 확인하여 양의 상관관계 또는 음의 상관관계 확인
posi_coef = []
nega_coef = []
other_coef = []
for var in significant_vars:
    coef = result.params[var]  # 회귀계수
    if coef > 0:
        posi_coef.append(var)
    elif coef < 0:
        nega_coef.append(var)
    else:
        other_coef.append(var)
print(f'Positive correlation: {posi_coef}')
print(f'Negative correlation: {nega_coef}')
print(f'No correlation: {other_coef}')

#Q8
from typing import Union
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

def root_mean_squared_error(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
def perf_eval_reg(y_true: Union[np.array,list], y_pred: Union[np.array,list])->list:
    rmse =root_mean_squared_error(y_true,y_pred)
    mae = MAE(y_true,y_pred)
    mape = MAPE(y_true,y_pred)
    return [rmse,mae,mape]


def display_result(result_dict: dict, dataset_names: list, metric_names: list):
    result_df = pd.DataFrame(result_dict).T
    result_df.index = dataset_names
    result_df.columns = metric_names
    print(result_df.round(4)) #유효 숫자 4자리 통일

result_dict={} # 초기화

result_dict = perf_eval_reg(y, y_pred)

dataset_names = ['Seoul bike rent']
metric_names = ['RMSE', 'MAE', 'MAPE']
display_result(result_dict, dataset_names, metric_names)

#Q9.
# Rainfall이 0인 비율 계산
rainfall_zero_ratio = (bike_mlr_data['Rainfall(mm)'] == 0).mean()

# Snowfall이 0인 비율 계산
snowfall_zero_ratio = (bike_mlr_data['Snowfall(cm)'] == 0).mean()

print("Rainfall이 0인 비율:", rainfall_zero_ratio)
print("Snowfall이 0인 비율:", snowfall_zero_ratio)
#다중공산성 계산
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 설명 변수에 상수항 추가
X_with_const = sm.add_constant(bike_trn_data.iloc[:, 1:])

# 변수 제거 전 VIF 계산
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

print(vif_data)


# 특정 변수들을 제외한 나머지 변수들 선택
selected_variables = [col for col in bike_trn_data.columns if col not in ['Wind speed', 'Visibility', 'Dew point temperature', 'Rainfall(mm)', 'Snowfall(cm)', 'Functioning day','Not Functioning day','Holiday', 'No Holiday', 'Spring', 'Summer', 'Autumn', 'Winter']]

# 선택된 변수들로 데이터프레임 구성
X_selected = bike_trn_data[selected_variables]

# 상수항 추가
X_selected_with_const = sm.add_constant(X_selected)

# VIF 계산
vif_data_selected = pd.DataFrame()
vif_data_selected["Variable"] = X_selected_with_const.columns
vif_data_selected["VIF"] = [variance_inflation_factor(X_selected_with_const.values, i) for i in range(X_selected_with_const.shape[1])]

print(vif_data_selected)

#Q10.
# 입력변수 재선정
bike_mlr_data2 = new_data[['Rented Bike Count', 'Hour', 'Temperature', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)']]
print(bike_mlr_data2.head())

seed=1234
test_size=0.3
bike_trn_data2, bike_valid_data2=train_test_split(bike_mlr_data2, test_size=test_size, random_state=seed)

x_trn2=bike_trn_data2.iloc[:,1:].to_numpy()
y_trn2=bike_trn_data2.iloc[:,0].to_numpy()

from sklearn.linear_model import LinearRegression
new_mlr=LinearRegression()

new_mlr.fit(x_trn2,y_trn2)

import statsmodels.api as sm

X=bike_trn_data2.iloc[:,1:]
X=sm.add_constant(X) #상수항 추가하여 절편 모델링
y=bike_trn_data2.iloc[:,0]

result=sm.OLS(y, X)
result=result.fit()
print(result.summary())

#새로운 데이터셋에 대한 평가지표 출력
y_pred=result.predict(X)
result_dict={} # 초기화
result_dict = perf_eval_reg(y, y_pred)

dataset_names = ['Seoul bike rent']
metric_names = ['RMSE', 'MAE', 'MAPE']
display_result(result_dict, dataset_names, metric_names)

#Extra
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


# 목표 변수 및 입력 변수 선택(이상치 제거한 전체 데이터셋으로 진행)
X = new_data.drop(columns=['Rented Bike Count'])
y = new_data['Rented Bike Count']

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 라쏘 회귀 모델 훈련
alpha = 1.0  # 라쏘 회귀의 규제 강도 (하이퍼파라미터)
max_iter = 10000  # 최대 반복 횟수
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_scaled, y_train)

# 테스트 세트 예측
y_pred = lasso_model.predict(X_test_scaled)

# 평가 지표 계산
result_dict = perf_eval_reg(y_test, y_pred)
dataset_names = ['Seoul bike rent']
metric_names = ['RMSE', 'MAE', 'MAPE']
display_result(result_dict, dataset_names, metric_names)