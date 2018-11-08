import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('./housing.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

regressor = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

non_medv = [[column] for column in df.columns if column != 'MEDV']

def get_permutations(X, n=1):
    if n <= 1:
        return X 

    elements = [[x] for x in np.unique(X)] 
    if n > len(elements):
        return get_permutations(X, n - 1)
    permutations = [x + element for x in X for element in elements if element[0] not in x]
    unique_permutations = np.unique([sorted(p) for p in permutations], axis=0).tolist()

    return get_permutations(unique_permutations, n - 1) 

all_combos = []

for i in range(len(non_medv)):
    all_combos += get_permutations(non_medv, i + 1)

best_score = 0
best_selection_transformation = None
for combo in all_combos:
    print("fitting for combo %s" % (combo))
    X = df[combo].values
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)
    y = df['MEDV'].values

    regressor = LinearRegression()

    regressor.fit(X, y)
    linear_score = r2_score(y, regressor.predict(X))
    if linear_score > best_score:
        best_score = linear_score
        best_selection_transformation = "%s linear (%f)" % (combo, best_score)
    print("linear score: %s" % linear_score)
    
    regressor.fit(X_quad, y)
    quad_score = r2_score(y, regressor.predict(X_quad))
    if quad_score > best_score:
        best_score = quad_score
        best_selection_transformation = "%s quadratic (%f)" % (combo, best_score)
    print("quadratic score: %s" % quad_score)

    regressor.fit(X_cubic, y)
    cubic_score = r2_score(y, regressor.predict(X_cubic))
    if cubic_score > best_score:
        best_score = cubic_score
        best_selection_transformation = "%s cubic (%f)" % (combo, best_score)
    print("cubic score: %s" % cubic_score)

print(best_selection_transformation)
