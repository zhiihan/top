import sys,os

os.environ['OMP_NUM_THREADS']=str(16) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']=str(16) # set number of MKL threads to run in parallel

#os.environ['OMP_NUM_THREADS']=str(int(sys.argv[1])) # set number of OpenMP threads to run in parallel
#os.environ['MKL_NUM_THREADS']=str(int(sys.argv[2])) # set number of MKL threads to run in parallel

#This is a script that implements the full tensor product hamiltonian in Quspin. 
#This may take a long time for large N

import quspin
import numpy as np # generic math functions
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
import pandas as pd

def make_Hamiltonian(N,J,h,t,mu,Delta):
    """Returns a Quantum Hamiltonian."""
    # Spin Hamiltonian 
    J_sum = [[-J, i, (i+1)%N] for i in range(N)]
    h_sum = [[-h, i] for i in range(N)]

    # Fermion Hamiltonian
    t_sum_pm = [[-t, i,(i+1)%N] for i in range(N)]
    t_sum_mp = [[-t, (i+1)%N,i] for i in range(N)]
    mu_sum = [[-mu,i] for i in range(N)]
    Delta_sum_zmm = [[Delta, i,i,(i+1)%N] for i in range(N)]
    Delta_sum_zpp = [[Delta, i,(i+1)%N,i] for i in range(N)]

    static = [
        ["zz|",J_sum],
        ["x|",h_sum],
        ["|+-",t_sum_pm],
        ["|+-",t_sum_mp],
        ["z|--",Delta_sum_zmm],
        ["z|++",Delta_sum_zpp],
        ['|n',mu_sum]
    ]
    dynamic = []
    
    spin_basis = quspin.basis.spin_basis_1d(N)    
    fermion_basis = quspin.basis.spinless_fermion_basis_1d(N)
    tensor_basis = quspin.basis.tensor_basis(spin_basis,fermion_basis) #spin | fermion

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

    H = quspin.operators.hamiltonian(static,dynamic,basis=tensor_basis,**no_checks)
    return H #returns as a Quspin Hamiltonian

def make_Magnetisation(N):
    """Returns a Quspin Quantum Operator. (Hamiltonian)"""
    z_sum = [[(1/N), i] for i in range(N)]

    static = [
                ['z|', z_sum], #pauli z
             ]

    dynamic = []
    
    spin_basis = quspin.basis.spin_basis_1d(N)
    fermion_basis = quspin.basis.spinless_fermion_basis_1d(N)
    tensor_basis = quspin.basis.tensor_basis(spin_basis,fermion_basis)

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

    M = quspin.operators.hamiltonian(static,dynamic,basis=tensor_basis,**no_checks).tocsr()
    return M 

def make_Magnetisation_staggered(N):
    """Returns a Quspin Quantum Operator. (Hamiltonian)"""
    z_sum = [[((-1)**i)/N, i] for i in range(N)]

    static = [
                ['z|', z_sum], #pauli z
             ]

    dynamic = []
    
    spin_basis = quspin.basis.spin_basis_1d(N)
    fermion_basis = quspin.basis.spinless_fermion_basis_1d(N)
    tensor_basis = quspin.basis.tensor_basis(spin_basis,fermion_basis)

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

    M = quspin.operators.hamiltonian(static,dynamic,basis=tensor_basis,**no_checks).tocsr()
    return M 

def make_Fermion_pair(N):
    """Returns a Quantum Operator."""
    mm_sum = [[1, i,(i+1)%N] for i in range(N)]

    static = [
                ['|--', mm_sum], #pauli z
             ]
    
    dynamic = []
    

    spin_basis = quspin.basis.spin_basis_1d(N)
    fermion_basis = quspin.basis.spinless_fermion_basis_1d(N)
    tensor_basis = quspin.basis.tensor_basis(spin_basis,fermion_basis)

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

    O = quspin.operators.hamiltonian(static,dynamic,basis=tensor_basis,**no_checks).tocsr()
    
    #returns Quantum_linearOperator object
    return O

def compute_data(N, Delta, t, mu, J, h):
    """Returns a Python dictionary."""
    
    #print('Beginning trial N = ',N,'h = ',h,'J = ',J, 'Delta =', Delta)
    
    #Add Divide by N

    H = make_Hamiltonian(N,J,h,t,mu,Delta)

    E, V = eigsh(H.aslinearoperator(),which='SA',return_eigenvectors=True,k=1) #Multi-OpenMP/MKL 
    #E, V = scipy.sparse.linalg.eigsh(H.tocsr(),which='SA',return_eigenvectors=True,k=1) #Not Multithreaded
    #E, V = H.eigsh(which='SA',return_eigenvectors=True,k=2)    
    
    M = make_Magnetisation(N)
    O = make_Fermion_pair(N)
    Ms = make_Magnetisation_staggered(N)
    V = V[:,0]
    V = V[:, np.newaxis]
    Ms2_expval = np.vdot(Ms @ V,Ms @ V) #complex dotproduct
    M2_expval = np.vdot(M @ V,M @ V) #complex dotproduct
    O2_expval = np.vdot(O @ V,O @ V) #complex dotproduct
    
    dataset = {'E'+str(i)+"_N": energy/N for i, energy in enumerate(E)}    
    
    dataset.update({"E_N":E[0]/N,
            'M^2': np.real(M2_expval),
            'O^2':np.real(O2_expval),
            'Ms^2':np.real(Ms2_expval),
            'J':J,
            'Delta':Delta,
            'h':h, 
            'N':N,
            't':t,
            'identity': np.real(M2_expval - np.vdot(V, M @ V)**2)     
            })
    
    #store the data as a dictionary, since running this function once is one observation
    return dataset 