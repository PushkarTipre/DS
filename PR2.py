import numpy as np

def McCulloch_pitts(input,weights,threshold):
    activation=np.dot(input,weights)
    output = 1 if activation >= threshold else 0
    return output

def ANDNOT(x1,x2):
    input = [x1,x2]
    weights = [1,-1]
    threshold=1
    return McCulloch_pitts(input,weights,threshold)

inp1=int(input("Enter 1st Input : ")) #Any input (1 or 0)
inp2=int(input("Enter 2nd Input : ")) #Any input (1 or 0)

print(ANDNOT(inp1,inp2))

#NOT -1 T0
#AND 1 1 T2
#OR 1 1 T1
#NOR -1 -1 T-1
