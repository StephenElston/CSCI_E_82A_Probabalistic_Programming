import numpy as np
import numpy.random as nr

def sim_bernoulli(p, n = 25):
    """
    Function to compute the vectors with probabilities for each 
    condition (input value) of the dependent variable using the Bernoulli
    distribution. 
    
    The arguments are:
    p - a vector of probabilites of success for each case.
    n - The numer of realizations. 
    """
    temp = np.zeros(shape = (len(p), n))
    for i in range(len(p)): 
        temp[i,:] = nr.binomial(1, p[i], n)
    return(temp)

def selec_dist_1(sims, var, lg):
    """
    Function to integrate the conditional probabilities for
    each of the cases of the parent variable. 
    
    The arguments are:
    sims - the array of simulated realizations with one row for each state of the
           parent variable. 
    var - the vector of values of parent variable used to select the value from the 
          sims array.
    lg - vector of states of possible states of the parent variable. These must be
         in the same order as for the sims array. 
    """
    out = sims[0,:] # Copy of values for first parent state
    var = np.array(var).ravel()
    for i in range(1, sims.shape[0]): # loop over other parent states
        out = [x if u == lg[i] else y for x,y,u in zip(sims[i,:], out, var)]
    return([int(x) for x in out])

def set_class(x):
    """
    Function to flatten the array produced by the numpy.random.multinoulli function. 
    The function tests which binary value of the array of output states is true
    and substitutes an integer for that state. This function only works for up to three
    output states. 
    
    Argument:
    x - The array produced by the numpy.random.multinoulli function. 
    """
    out = []
    for i,j in enumerate(x):
        if j[0] == 1: out.append(0)
        elif j[1] == 1: out.append(1)
        else: out.append(2)   
    return(out)   


def sim_multinoulli(p, n = 25):
    """
    Function to compute the vectors with probabilities for each 
    condition (input value) of the dependent variable using the multinoulli
    distribution. 
    
    The arguments are:
    p - an array of probabilites of success for each possible combination
        of states of the parent variables. Each row in the array are the 
        probabilities for each state of the multinoulli distribution for 
        that combination of parent values.
    n - The numer of realizations. 
    """
    temp = np.zeros(shape = (p.shape[0], n))
    for i in range(p.shape[0]): 
        ps = p[i,:]
        mutlis = nr.multinomial(1, ps, n) 
        temp[i,:] = set_class(mutlis)
    return(temp)

def selec_dist_2(sims, var1, var2, lg1, lg2):
    """
    Function to integrate the conditional probabilities for
    each of the cases of two parent variables. 
    
    The arguments are:
    sims - the array of simulated realizations with one row for each state of the
           union of the parent variables. 
    var1 - the vector of values of first parent variable used to select the value from the 
          sims array.
    var2 - the vector of values of second parent variable used to select the value from the 
          sims array.
    lg1 - vector of states of possible states of the first parent variable. These must be
         in the same order as for the sims array. 
    lg2 - vector of states of possible states of the second parent variable. These must be
         in the same order as for the sims array. 
    """
    out = sims[0,:] # Copy values for first combination of states for parent variables
    ## make sure the parent variables are 1-d numpy arrays.
    var1 = np.array(var1).ravel() 
    var2 = np.array(var2).ravel()
    for i in range(1, sims.shape[0]): # Loop over all comnination of states of the parent variables
        out = [x if u == lg1[i] and v == lg2[i] else y for x,y,u,v in zip(sims[i,:], out, var1, var2)]
    return([int(x) for x in out])

