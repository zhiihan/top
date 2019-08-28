import sys,os

os.environ['OMP_NUM_THREADS']=str(4) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']=str(4) # set number of MKL threads to run in parallel

#os.environ['OMP_NUM_THREADS']=str(int(sys.argv[1])) # set number of OpenMP threads to run in parallel
#os.environ['MKL_NUM_THREADS']=str(int(sys.argv[2])) # set number of MKL threads to run in parallel

#This is a script that implements the full tensor product hamiltonian in Quspin. 
#This may take a long time for large N

import numpy as np # generic math functions
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
import pandas as pd
import math

import holoviews as hv
hv.extension('bokeh')

def cond(i):
    if i == '0':
        return -1
    if i == '1':
        return 1
    else:
        print(i)
        raise ValueError
        
def make_delta_mat(N,Delta,binary):
    """makes the Delta matrix"""   
    Delta_mat = np.zeros((N,N))
    for i,num in enumerate(binary):
        Delta_mat[i%N,(i+1)%N] = cond(num)
        Delta_mat[(i+1)%N,i%N] = -cond(num)
    #Delta_mat[N-1, 0] = cond(binary[-1])
    #Delta_mat[0, N-1] = -cond(binary[-1])
    return Delta_mat*Delta

def make_t_mat(N, t):
    """makes the t matrix"""  
    t_mat = np.zeros((N,N))
    for i in range(0,N):
        t_mat[i%N,(i+1)%N] = -t
        t_mat[(i+1)%N,i%N] = -t
    #t_mat[0, N-1] = -t
    #t_mat[N-1, 0] = -t
    return t_mat
    
def make_z_string(N):
    """generates a random string using recursion"""
    
    if N == 0:
        return ['']
    else:
        return ['1' + i for i in make_z_string(N-1)] + ['0' + i for i in make_z_string(N-1)]

def make_BDG(N, t, Delta, binary):
    """builds a BDG Hamiltonian and returns as numpy array"""
    
    t_mat = make_t_mat(N, t)
    Delta_mat = make_delta_mat(N, Delta, binary)    
    
    return np.block([
        [t_mat, Delta_mat],
        [-Delta_mat, -t_mat]
    ])

def make_J_term(J, binary):
    """Calculates the value of the -J term, with the minus sign. """
    
    v = np.zeros(len(binary))
    u = np.zeros(len(binary))
       
    for i,j in enumerate(binary):     
        v[i] = cond(j)
        u[i-1] = cond(j)
    return -J*np.dot(v,u)/len(binary)

def compute_M(binary):
    """Computes M^2."""
    return (1/len(binary)*sum([cond(i) for i in binary]))**2

def compute_Ms(binary):
    """Computes Ms^2."""
    return (1/len(binary)*sum([((-1)**j)*cond(num) for j,num in enumerate(binary)]))**2

def compute_data(N, Delta, t, mu, J):
    """Returns a Pandas dataframe object."""
    
    #generate all binary strings and t blocks
    nums = make_z_string(N) 

    #iterate over all strings
    list_dicts = []
    for l, binary in enumerate(nums): 

        #get the delta block and make the BDG Hamiltonian
        H_BDG = make_BDG(N, t, Delta, binary) 

        #diagonalize and save data
        E, V = la.eigsh(H_BDG,which='SA',return_eigenvectors=True,k=1) #lanczos algorithm

        J_term = make_J_term(J, binary)
                
        #now its ground per particle
        dataset = {"ground_J=0": E[0]/math.pi,
                   "J_term": J_term,
                   "M^2" :compute_M(binary),
                   "N" :N,
                   "J" :J,
                   "Delta": Delta,
                   "Ms^2": compute_Ms(binary),
                   "binary": str(binary),
                   "ground": E[0]/math.pi + J_term,
                   "t": t
                  }

        list_dicts.append(dataset)
    return list_dicts

# tediously compute the data

mu = 0
list_dicts = []
linspace = 50
t = 1
for N in range(6, 15):
    # Tediously compute the data
    for J in np.linspace(0, 2, num=linspace):
        for Delta in np.linspace(0, 2,num=linspace):
            list_dicts += compute_data(N, Delta, t, mu, J)
            print('Successful', {'N':N, 'Delta':Delta, 'J':J})
    
    print("Done computing, minimizing...")
    df = pd.DataFrame(list_dicts)
    # Find minimum
    ids = []
    for J in np.linspace(0, 2, num=linspace):
        for Delta in np.linspace(0, 2,num=linspace):
            try:
                ids.append(df[(abs(df.J - J) < 0.00005) & (abs(df.Delta - Delta) < 0.00005)].ground.idxmin())
                print('Successful', {'N':N, 'Delta':Delta, 'J':J})
            except ValueError:
                print(J, Delta)
    df_minvals = df.iloc[ids]
    pd.DataFrame.to_csv(df_minvals,'bdg_eigenvals_full_minvals'+str(N)+'.csv', index=False)
    #df_minvals = pd.read_csv('bdg_eigenvals_full_minvals'+str(N)+'.csv', converters={'index': str})