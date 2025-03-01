#9, 10

import pandas as pd
dia = pd.read_csv("Diabetes.csv")
# 종속 변수를 제외한 열의 수 계산
max_features = len(dia.columns) - 1  # 종속 변수 1개를 제외함

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


#데이터 표준화 전처리
from sklearn.preprocessing import scale

dia_input_scaled= scale(new_data.iloc[:,:8])
dia_target = new_data.iloc[:,-1] #종속 변수

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
seed=12345
test_size=0.3


from sklearn.metrics import roc_auc_score
from typing import Union
import numpy as np

class GeneticAlgorithm(object):
    def __init__(self, population_size: int, n_feat: int, n_parents: int, n_gen: int,
                 init_rate: float, mutation_rate: float, crossover_rate: float,
                 model: object, seed: int) -> None:
        self.population_size = population_size
        self.n_feat = n_feat
        self.n_parents = n_parents
        self.n_gen = n_gen
        self.init_rate = init_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.model = model
        self.seed = seed

    def initialization_of_population(self, size: int, n_feat: int, init_prob: float) -> list:
        population = []
        for i in range(size):
            chromosome = np.bool_(np.ones(n_feat))
            chromosome[:int(init_prob * n_feat)] = False
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def fit(self, X_train, y_train):
        np.random.seed(self.seed)
        best_chromo = []
        best_score = []
        population_nextgen = self.initialization_of_population(self.population_size, self.n_feat, self.init_rate)
        for i in range(self.n_gen):
            scores, pop_after_fit = self.fitness_score(self.model, population_nextgen, X_train, y_train)
            best_chromo.append(pop_after_fit[0])
            best_score.append(scores[0])
            print('Best score(Training) in generation', i + 1, ':', scores[:1])
            pop_after_sel = self.selection(pop_after_fit, self.n_parents)
            pop_after_cross = self.crossover(pop_after_sel, self.crossover_rate)
            population_nextgen = self.mutation(pop_after_cross, self.mutation_rate, self.n_feat)
        return best_chromo, best_score

    def fitness_score(self, model, population: list, X_train: np.array,
                      Y_train: np.array):
        scores = []
        for chromosome in population:
            model.fit(X_train[:, chromosome], Y_train)
            predictions_proba = model.predict_proba(X_train[:, chromosome])[:, 1]  # Positive class probability
            auc = roc_auc_score(Y_train, predictions_proba)
            scores.append(auc)
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)  # ascending orders
        return list(scores[inds][::-1]), list(population[inds, :][::-1])

    def selection(self, pop_after_fit, n_parents):
        population_nextgen = []
        for i in range(n_parents):
            population_nextgen.append(pop_after_fit[i])
        return population_nextgen

    def crossover(self, pop_after_sel: list, crossover_rate: float):
        pop_nextgen = pop_after_sel
        for i in range(0, len(pop_after_sel), 2):
            new_par = []
            child_1, child_2 = pop_nextgen[i], pop_nextgen[i + 1]
            select_idx = np.random.random_sample(len(child_1)) > crossover_rate
        for j in range(len(child_1)):
            new_par.append(child_1[j] if select_idx[j] else child_2[j])
            pop_nextgen.append(new_par)
        return pop_nextgen

    def mutation(self, pop_after_cross: list, mutation_rate: float, n_feat: int):
        mutation_range = int(mutation_rate * n_feat)
        pop_next_gen = []
        for n in range(0, len(pop_after_cross)):
            chromo = pop_after_cross[n]
            rand_posi = []
            for i in range(0, mutation_range):
                pos = np.random.randint(0, n_feat - 1)
                rand_posi.append(pos)
            for j in rand_posi:
                chromo[j] = not chromo[j]
            pop_next_gen.append(chromo)
        return pop_next_gen


full_config = {
'penalty': None,
'fit_intercept':True,
'max_iter':int(1e+5),
'solver':'saga',
'random_state':seed,
'n_jobs':-1
}
full_model = LogisticRegression(**full_config)

genetic_config={
    'population_size': 20,
    'n_feat': max_features,
    'n_parents': 2,
    'n_gen': 5,
    'init_rate': 0.3,
    'mutation_rate': 0.1,
    'crossover_rate': 0.3,
    'model': full_model,
    'seed': seed
}

import numpy as np
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time


# Define function to evaluate model performance
def evaluate_performance(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, balanced_accuracy, f1

# Define function to calculate feature reduction
def calculate_reduction(initial_features, selected_features):
    initial_count = np.sum(initial_features)
    selected_count = np.sum(selected_features)
    reduction = (initial_count - selected_count) / initial_count * 100
    return reduction

# Define function to run genetic algorithm
def run_genetic_algorithm(genetic_config, X_train, X_val, y_train, y_val):
    start_time_selection = time.time()  # 변수 선택 시작 시간 기록

    genetic_algo = GeneticAlgorithm(**genetic_config)
    selected_features, _ = genetic_algo.fit(X_train, y_train)
    end_time_selection = time.time()  # 변수 선택 종료 시간 기록
    selection_time = end_time_selection - start_time_selection  # 변수 선택에 걸린 시간 계산

    # Select features based on the best chromosome
    best_chromosome = selected_features[0]
    selected_X_train = X_train[:, best_chromosome]
    selected_X_val = X_val[:, best_chromosome]

    # Train logistic regression model
    model = LogisticRegression(**full_config)
    model.fit(selected_X_train, y_train)

    # Evaluate performance on validation set
    accuracy, balanced_accuracy, f1 = evaluate_performance(model, selected_X_val, y_val)

    # Calculate reduction in features
    reduction = calculate_reduction(np.ones(X_train.shape[1]), best_chromosome)

    return accuracy, balanced_accuracy, f1, reduction, best_chromosome, selection_time

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(dia_input_scaled, dia_target, test_size=test_size, random_state=seed)

# Run genetic algorithm and evaluate performance
accuracy, balanced_accuracy, f1, reduction, best_chromosome, selection_time = run_genetic_algorithm(genetic_config, X_train, X_val, y_train, y_val)

# Calculate AUROC
model = LogisticRegression(**full_config)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = round(roc_auc_score(y_val, y_pred_proba),4)

# Print performance on validation set
print("Performance on Validation Set:")
print("Accuracy:", round(accuracy,4))
print("Balanced Accuracy:", round(balanced_accuracy,4))
print("F1 Score:", round(f1,4))
print("Variable Reduction Rate:", round(reduction,4))
print("AUROC:", auc)
print("Execution Time:", round(selection_time,4))

# Define hyperparameter candidates
population_sizes = [10, 30, 50]
mutation_rates = [0.05, 0.1, 0.2]
crossover_rates = [0.1, 0.3, 0.5]

# Initialize counter for combinations
combination_count = 0

# Iterate over hyperparameter combinations
for pop_size in population_sizes:
    for mutation_rate in mutation_rates:
        for crossover_rate in crossover_rates:
            # Update combination count
            combination_count += 1

            # Configure genetic algorithm
            genetic_config = {
                'population_size': pop_size,
                'n_feat': dia_input_scaled.shape[1],
                'n_parents': 2,
                'n_gen': 5,
                'init_rate': 0.3,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'model': LogisticRegression(max_iter=int(1e5), solver='saga', random_state=seed, n_jobs=-1),
                'seed': seed
            }

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(dia_input_scaled, dia_target, test_size=test_size, random_state=seed)

            # Run genetic algorithm and evaluate performance
            accuracy, balanced_accuracy, f1,  reduction, best_chromosome, selection_time = run_genetic_algorithm(genetic_config, X_train, X_val, y_train, y_val)

            # Print results
            print(f"Combination {combination_count}:")
            print(f"Population Size: {pop_size}")
            print(f"Mutation Rate: {mutation_rate}")
            print(f"Crossover Rate: {crossover_rate}")
            print(f"Selected Variables Count: {sum(genetic_config['model'].fit(X_train[:, best_chromosome], y_train).coef_[0] != 0)}")
            print(f"Selected Variables: {new_data.columns[:-1][best_chromosome]}")
            print(f"Execution Time: {round(selection_time, 4)} seconds")
            print()