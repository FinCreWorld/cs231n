from unittest import result
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def activition_by_weight(weight_scales, model_generator, solver_generator):
    model = model_generator(0)
    cols = model.num_layers + 1
    rows = len(weight_scales)
    figure = plt.figure(figsize=(cols * 5, rows * 5))
    for row, weight_scale in tqdm(enumerate(weight_scales)):
        model = model_generator(weight_scale)
        solver = solver_generator(model)
        solver.train()
        for col in range(cols):
            figure.add_subplot(rows, cols, cols * row + col + 1)
            # print(f"count nan: ", np.sum(np.isnan(model.activation[col])))
            # print(f"count total: ", model.activation[col].flatten().shape)
            np.nan_to_num(model.activation[col], False)
            plt.hist(model.activation[col].flatten(), 100, histtype='bar')
            # plt.hist(model.activation[col][:, 0], 100, histtype='bar')
            plt.title(f"weight_scale={weight_scale:.1e}")
    plt.show()


def train_acc_by_lr(learning_rates, model_generator, solver_generator):
    results = []
    for learning_rate in tqdm(learning_rates):
        model = model_generator()
        solver = solver_generator(model, learning_rate)
        solver.train()
        results.append(np.sum(solver.train_acc_history[-5:])/5)
    plt.figure(figsize=(8, 8))
    plt.plot(np.log10(learning_rates), np.array(results))
    plt.show()
    return results

def val_acc_by_lr(learning_rates, model_generator, solver_generator):
    results = []
    for learning_rate in tqdm(learning_rates):
        model = model_generator()
        solver = solver_generator(model, learning_rate)
        solver.train()
        results.append(np.sum(solver.val_acc_history[-5:])/5)
    plt.figure(figsize=(8, 8))
    plt.plot(np.log10(learning_rates), np.array(results))
    plt.show()
    return results



