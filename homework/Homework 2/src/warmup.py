'''
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: Thomas Kelly
@author: Ona Igbinedion
@author: Raul Rodriguez
'''
import numpy as np
import pandas as pd
import pomegranate as pm


# Takes the distribution from a predict_proba and the state names from the dataframe, and then creates a dictionary like this
# {symbol 1 : [(val1, probability),(val2,probability),(val3...)]
# For example : {S :[(0,.48),(1,.26),(2,.26)]}
def symbolMatch(dist, col):
    result = {}
    for e in range(0, len(col)):
        #x = dist[e]
        # print(x.items())

        if dist[e] == 1:
            result[col[e]] = dist[e]
        else:
            result[col[e]] = dist[e].items()
    return result


if __name__ == '__main__':
    """
    PROBLEM 2.1
    Using the Pomegranate Interface, determine the answers to the
    queries specified in the instructions.

    ANSWER GOES BELOW:
    """
    df = pd.read_csv("../dat/adbot-data.csv")
    symdict = {}
    cols = df.columns
    for s in range(0, len(cols)):
        symdict[cols[s]] = s
    bn = pm.BayesianNetwork.from_samples(
        X=df, state_names=cols, algorithm="exact")
    p = bn.predict_proba({})
    test = symbolMatch(p, cols)
    print("Answering P(S)")
    print("S = 0", test["S"][0][1])
    print("S = 1", test["S"][1][1])
    print("S = 2", test["S"][2][1])
    """
    Written answer
    P(S) 
    S = 0 : 0.46959
    S = 1 : 0.26994
    S = 2 : 0.26047
    
    """
    p = bn.predict_proba({"G": 1})
    test = symbolMatch(p, cols)
    print("Answering P(S|G=1)")
    print("S = 0", test["S"][0][1])
    print("S = 1", test["S"][1][1])
    print("S = 2", test["S"][2][1])
    """
    Written Answer
    P(S|G=1)
    S = 0 : 0.54558
    S = 1 : 0.24338
    S = 2 : 0.21105
    """
    p = bn.predict_proba({"T": 1, "H": 1})
    test = symbolMatch(p, cols)
    print("Answering P(S|T=1,H=1)")
    print("S = 0", test["S"][0][1])
    print("S = 1", test["S"][1][1])
    print("S = 2", test["S"][2][1])
    """
    Written Answer
    P(S|T=1,H=1)
    S = 0 : 0.40347
    S = 1 : 0.30736
    S = 2 : 0.28917
    """
