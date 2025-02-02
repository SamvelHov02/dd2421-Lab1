import monkdata as m 
import dtree as d
import drawtree_qt5 as qt
import random

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
    

def assignment5():
    datasets = [m.monk1, m.monk2, m.monk3]
    testsets = [m.monk1test, m.monk2test, m.monk3test]
    
    for i in range(len(datasets)):
        t = d.buildTree(datasets[i],m.attributes)
        
        train_err = d.check(t,datasets[i])
        test_err  = d.check(t,testsets[i])
        
        print(f"Training error on MONK-{i + 1} is {1 - train_err}")
        print(f"Test error on MONK-{i + 1} is {1 - test_err}")

assignment5()


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)