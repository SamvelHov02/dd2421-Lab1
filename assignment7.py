import monkdata as m
import dtree as d
import random
import numpy as np
import matplotlib.pyplot as plt

def partition(data, fraction):
    """Randomly splits the dataset into training and validation sets."""
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune_tree(tree, validation_data):
    """Performs pruning by selecting the best pruned tree using validation data."""
    best_tree = tree
    best_score = d.check(tree, validation_data)
    
    while True:
        pruned_trees = d.allPruned(best_tree)
        improved = False
        
        for pruned_tree in pruned_trees:
            score = d.check(pruned_tree, validation_data)
            if score >= best_score:  
                best_tree = pruned_tree
                best_score = score
                improved = True
                
        if not improved:
            break  
    
    return best_tree

def evaluate_pruning(dataset, test_set, fractions, runs=20):
    """Evaluates pruning effect on test error for different partition fractions."""
    results = {}
    
    for fraction in fractions:
        test_errors = []
        
        for _ in range(runs):
            train_set, val_set = partition(dataset, fraction)
            full_tree = d.buildTree(train_set, m.attributes)
            pruned_tree = prune_tree(full_tree, val_set)
            test_error = 1 - d.check(pruned_tree, test_set)
            test_errors.append(test_error)
        
        results[fraction] = (np.mean(test_errors), np.std(test_errors))
    
    return results

def plot_pruning_results(results, dataset_name):
    """Plots test classification error with spread as a function of partition fraction."""
    fractions = sorted(results.keys())
    mean_errors = [results[f][0] for f in fractions]
    std_errors = [results[f][1] for f in fractions]
    
    plt.errorbar(fractions, mean_errors, yerr=std_errors, marker='o', linestyle='-', label=f'{dataset_name} Test Error')
    plt.xlabel('Training Data Fraction')
    plt.ylabel('Test Classification Error')
    plt.title('Effect of Pruning on Test Error')
    plt.legend()
    plt.grid()

# Evaluate pruning effect on monk1 and monk3 
fraction_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
plt.figure(figsize=(8, 5))

data_sets = [(m.monk1, m.monk1test, "MONK-1"), (m.monk3, m.monk3test, "MONK-3")]
for train_data, test_data, name in data_sets:
    pruning_results = evaluate_pruning(train_data, test_data, fraction_values)
    plot_pruning_results(pruning_results, name)

plt.show()
