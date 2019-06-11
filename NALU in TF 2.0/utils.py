import numpy as np

def generate_data(min_val, max_val, observations, op):
    data = np.random.uniform(min_val, max_val, size=(observations, 2))
    if op == '+':
        target = data[:, 0] + data[:, 1]
    elif op == '-':
        target = data[:, 0] - data[:, 1]
    elif op == '*':
        target = data[:, 0] * data[:, 1]
    elif op == '/':
        target = data[:, 0] / data[:, 1]
    elif op == '^2':
        data = np.random.uniform(min_val, max_val, size=(observations, 1))
        target = data ** 2
    elif op == 'sqrt':
        data = np.random.uniform(min_val, max_val, size=(observations, 1))
        target = np.sqrt(data)
    
    return data, target
