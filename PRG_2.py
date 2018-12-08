# (QUESTION: 02)
# For a given set of training data examples stored in a .CSV file, implement and
# demonstrate the Candidate-Elimination algorithmto output a description of the set
# of all hypotheses consistent with the training examples.

import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('PRG_2.csv'))
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]

    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])

    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final S:", s_final, sep="\n")
print("Final G:", g_final, sep="\n")
data.head()