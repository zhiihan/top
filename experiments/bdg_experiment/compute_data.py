import sys,os

os.environ['OMP_NUM_THREADS']=str(4) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']=str(4) # set number of MKL threads to run in parallel

#os.environ['OMP_NUM_THREADS']=str(int(sys.argv[1])) # set number of OpenMP threads to run in parallel
#os.environ['MKL_NUM_THREADS']=str(int(sys.argv[2])) # set number of MKL threads to run in parallel

#This is a script that implements the full tensor product hamiltonian in Quspin. 
#This may take a long time for large N

import quspin
import numpy as np # generic math functions
import matplotlib.pyplot as plt
import time
import scipy
import pandas as pd

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
        
def make_periodic_matrix(N,t=1,Delta=1,binary=None):
    """makes a periodic matrix"""   
    
    #makes the Delta Matrix
    if binary: 
        Delta_mat = np.zeros((N,N))
        for i,num in zip(range(0,N-1),binary):
            Delta_mat[i,i+1] = cond(num)
            Delta_mat[i+1,i] = -cond(num)
        Delta_mat[N-1, 0] = cond(binary[-1])
        Delta_mat[0, N-1] = -cond(binary[-1])
        return Delta_mat*Delta
    
    #makes the t Matrix
    else:
        t_mat = np.zeros((N,N))
        for i in range(0,N-1):
            t_mat[i,i+1] = -t
            t_mat[i+1,i] = -t
        t_mat[0, N-1] = -t
        t_mat[N-1, 0] = -t
        return t_mat
    
def make_z_string(N):
    """generates a random string using recursion"""
    
    if N == 0:
        return ['']
    else:
        return ['1' + i for i in make_z_string(N-1)] + ['0' + i for i in make_z_string(N-1)]

def make_BDG(t_mat, Delta_mat):
    """builds a BDG Hamiltonian and returns as numpy array"""
    
    return np.block([
        [t_mat, Delta_mat],
        [-Delta_mat, -t_mat]
    ])

def make_J_term(binary,J=1):
    """Calculates the value of the -J term, with the minus sign. """
    v = np.zeros(len(binary))
    u = np.zeros(len(binary))
       
    for i,j in enumerate(binary):     
        v[i] = cond(j)
        u[i-1] = cond(j)
    return -J*np.dot(v,u)

def compute_M(binary):
    """Computes M^2."""
    return (1/len(binary)*sum([cond(i) for i in binary]))**2

def compute_data(N, Delta = 1, t = 1, mu = 0, J=1):
    """Returns a Pandas dataframe object."""
    
     #initial conditions
    data_N = np.zeros((2**N, 6)) 

    #generate all binary strings and t blocks
    nums = make_z_string(N) 
    t_mat = make_periodic_matrix(N,t=t,Delta=Delta,binary=None)

    #iterate over all strings
    for l, binary in enumerate(nums): 

        #get the delta block and make the BDG Hamiltonian
        Delta_mat = make_periodic_matrix(N,t,Delta=Delta,binary=binary) 
        H_BDG = make_BDG(t_mat, Delta_mat) 

        #diagonalize and save data
        E, V = scipy.sparse.linalg.eigsh(H_BDG,which='SA',return_eigenvectors=True,k=1) #lanczos algorithm

        data_N[l,0] = E 
        data_N[l,1] = make_J_term(binary, J=J)
        data_N[l,2] = compute_M(binary)
        data_N[l,3] = N
        data_N[l,4] = J
        data_N[l,5] = Delta

    #store data in Pandas dataframe
    df = pd.DataFrame(data_N, columns = ['ground_J=0', '-J_term', 'M^2', 'N', 'J', 'Delta'], index = nums) 
    df['ground'] = df['ground_J=0'] + df['-J_term']
    df['index'] = df.index
    #sort and save
    df = df.sort_values('ground',ascending=True)
    #pd.DataFrame.to_csv(df,'bdg_eigenvals'+str(N)+'.csv')
    return df

# tediously compute the data

for N in range(12,14):
    df = pd.DataFrame()
    for Delta in np.linspace(0,2,num=250):
        for J in np.linspace(0,1,num=250):
            df = df.append(compute_data(N, J=J,Delta=Delta))
            print(N,Delta,J)
    pd.DataFrame.to_csv(df,'bdg_eigenvals_full'+str(N)+'.csv')
    df = pd.read_csv('bdg_eigenvals_full'+str(N)+'.csv', converters={'index': str})
    df = df.drop(columns='Unnamed: 0')
    ids = []

    for Delta in np.linspace(0,2,num=250):
        for J in np.linspace(0,1,num=250):
            try:
                ids.append(df[(abs(df.J - J) < 0.00005) & (abs(df.Delta - Delta) < 0.00005)].ground.idxmin())
            except ValueError:
                print(J, Delta)
    df_minvals = df.iloc[ids]
    pd.DataFrame.to_csv(df_minvals,'bdg_eigenvals_full_minvals'+str(N)+'.csv')