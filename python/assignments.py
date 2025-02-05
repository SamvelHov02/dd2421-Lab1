import monkdata as m 
import dtree as d
import drawtree_qt5 as qt
import random
import numpy as np
import matplotlib.pyplot as plt

def assignment1():  
    print(d.entropy(m.monk1))
    print(d.entropy(m.monk2))
    print(d.entropy(m.monk3))


def assignment3():
    datasets = [m.monk1, m.monk2, m.monk3]
    for i, set in enumerate(datasets):
        for j, att in enumerate(m.attributes):
            print(f"Information Gain: MONK-{i + 1} for attribute a{j + 1} is {d.averageGain(set, att)}")
     
#assignment3()

def assignment4():
    datasets = [m.monk1, m.monk2, m.monk3]
    for i, set in enumerate(datasets):
        print(f"Best attribute for MONK-{i + 1} is {d.bestAttribute(set, m.attributes)}")
    

def monk1BuildTree():
    '''attributeName = d.bestAttribute(m.monk1, m.attributes)
    for attribute in m.attributes:
        if attribute.name == attributeName:
            for val in attribute.values:
                d.select(m.monk1, attribute, val)'''
                
    tree = d.buildTree(m.monk1,m.attributes, 2)
    print(tree)
    

def pre_assignment5():
    dataset = m.monk1
    a = d.bestAttribute(dataset, m.attributes)
    
    subsets = [d.select(dataset, a, v) for v in a.values]
    print(a)
    print(len(subsets))
    

def assignment5():
    datasets = [m.monk1, m.monk2, m.monk3]
    testsets = [m.monk1test, m.monk2test, m.monk3test]
    
    for i in range(len(datasets)):
        t = d.buildTree(datasets[i],m.attributes)
        
        train_err = d.check(t,datasets[i])
        test_err  = d.check(t,testsets[i])
        
        print(f"Training error on MONK-{i + 1} is {1 - train_err}")
        print(f"Test error on MONK-{i + 1} is {1 - test_err}")
        
        
def pre_assignment6():
    pass


# pre_assignment5()

qt.drawTree(d.buildTree(dataset=m.monk1,attributes=m.attributes))



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
        
        for pruned_tree in pruned_trees:
            score = d.check(pruned_tree, validation_data)
            if score >= best_score:  
                best_tree = pruned_tree
                best_score = score
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
