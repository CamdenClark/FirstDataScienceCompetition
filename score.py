import pandas as pd
import numpy as np
symbols = pd.read_csv("symbol.csv")

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    """
    return [[x0, y0, z0] for x0 in arrays[0][0] for y0 in arrays[1][0] for z0 in arrays[2][0]]

numscores = [pd.read_csv("num1scores.csv"), pd.read_csv("num2scores.csv"), pd.read_csv("num3scores.csv")]
numscores = [scores.drop(['Unnamed: 0'], axis=1) for scores in numscores]
finscores = [scores.apply(lambda x: np.where(x.as_matrix() >= .2), axis=1) for scores in numscores]

def score_all(equation, prod):
    for combination in prod:
        for combo in combination:
            if score(equation, combination):
                return 1
    return 0

def score(equation, combination):
    if equation['symbol1'] == '-':
        return combination[0] - combination[1] == combination[2]
    if equation['symbol1'] == '+':
        return combination[0] + combination[1] == combination[2]
    if equation['symbol1'] == '=':
        if equation['symbol2'] == '-':
            return combination[1] - combination[2] == combination[0]
        if equation['symbol2'] == '+':
            return combination[1] + combination[2] == combination[0]
        else:
            return False
    else:
        return False
testing = np.array(finscores).transpose([1, 0, 2])
final = [score_all(symb, cartesian(testing[index])) for index, symb in symbols.iterrows()]
outdf = pd.DataFrame(final, columns=['label']).to_csv("to_score.csv", index_label="index")
