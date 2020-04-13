#This file contains the functions called from the main.py file for a full order frequency sweep
#Functions -FullSweep (frequency sweep no pod)
#          -FullSweepMulti (frequency sweep in parallel no pod)
#Importing
import os
import sys
import time
import multiprocessing_on_dill as multiprocessing

import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from MPTFunctions import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters



#Function definition for a full order frequency sweep
def FullSweep(Object,Order,alpha,inorout,mur,sig,Array,BigProblem):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(5)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    R=np.zeros([3,3])
    I=np.zeros([3,3])
    TensorArray=np.zeros([NumberofFrequencies,9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies,3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies,3])
    Eigenvalues = np.zeros([NumberofFrequencies,3], dtype=complex)



#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])
    
    
    #Run in three directions and save in an array for later
    for i in range(3):
        Theta0Sol[:,i] = Theta0(fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
    print(' solved theta0 problems    ')
    
    #Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(i+1):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh)))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh))
    
    #Copy the tensor 
    N0+=np.transpose(N0-np.eye(3)@N0)


#########################################################################
#Theta1
#This section solves the Theta1 problem to calculate the solution vectors
#of the snapshots

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]
    
    #Setup the grid functions and arrays which will be used to store the solution vectors
    Theta1i = GridFunction(fes2)
    Theta1j = GridFunction(fes2)
    Theta1Sol = np.zeros([ndof2,3],dtype=complex)
    
    #Sweep through all points
    for i,Omega in enumerate(Array):
        nu = Omega*Mu0*(alpha**2)
        
        #Solve for each direction
        Theta1Sol[:,0] = Theta1(fes,fes2,Theta0Sol[:,0],xivec[0],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofFrequencies,Solver)
        Theta1Sol[:,1] = Theta1(fes,fes2,Theta0Sol[:,1],xivec[1],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofFrequencies,Solver)
        Theta1Sol[:,2] = Theta1(fes,fes2,Theta0Sol[:,2],xivec[2],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofFrequencies,Solver)
        
        #Calculate the tensors
        R,I = MPTCalculator(mesh,fes,fes2,Theta1Sol[:,0],Theta1Sol[:,1],Theta1Sol[:,2],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,"No Print",NumberofFrequencies)
        TensorArray[i,:] = (N0+R+1j*I).flatten()
        RealEigenvalues[i,:] = np.sort(np.linalg.eigvals(N0+R))
        ImaginaryEigenvalues[i,:] = np.sort(np.linalg.eigvals(I))
    print(' solved theta1 problems     ')
    print(' frequency sweep complete')
    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues

    return TensorArray, EigenValues, N0, numelements



#Function definition for a full order frequency sweep in parallel
def FullSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,CPUs,BigProblem):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(5)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    TensorArray=np.zeros([NumberofFrequencies,9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies,3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies,3])
    Eigenvalues = np.zeros([NumberofFrequencies,3], dtype=complex)



#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])
    
    #Setup the inputs for the functions to run
    Runlist = []
    for i in range(3):
        if CPUs<3:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
        else:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,"No Print",Solver)
        Runlist.append(NewInput)
    #Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(Theta0, Runlist)
    print(' solved theta0 problems    ')
    
    #Unpack the outputs
    for i,Direction in enumerate(Output):
        Theta0Sol[:,i] = Direction


#Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh)))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh))



#########################################################################
#Theta1
#This section solves the Theta1 problem and saves the solution vectors

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]
    
    #Setup the grid functions and arrays which will be used to store the solution vectors
    Theta1i = GridFunction(fes2)
    Theta1j = GridFunction(fes2)
    
    Theta1E1Sol = np.zeros([ndof2,NumberofFrequencies],dtype=complex)
    Theta1E2Sol = np.zeros([ndof2,NumberofFrequencies],dtype=complex)
    Theta1E3Sol = np.zeros([ndof2,NumberofFrequencies],dtype=complex)
    
    #Setup the inputs for the functions to run
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 1)
    for j,Omega in enumerate(Array):
        nu = Omega*Mu0*(alpha**2)
        for i in range(3):
            NewInput = (fes,fes2,Theta0Sol[:,i],xivec[i],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,counter,len(Array),Solver)
            Runlist.append(NewInput)
    
    #Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(Theta1, Runlist)
    print(' solved theta1 problems    ')
    #Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        position = int(floor(i/3))
        direction = i-3*position
        if direction==0:
            Theta1E1Sol[:,position] = OutputNumber
        if direction==1:
            Theta1E2Sol[:,position] = OutputNumber
        if direction==2:
            Theta1E3Sol[:,position] = OutputNumber
    


#########################################################################
#Calculate the tensors
#Calculate the tensors and eigenvalues which will be used in the sweep
    
    #Create the inputs for the calculation of the tensors
    Runlist = []
    counter = manager.Value('i', 0)
    for i,Omega in enumerate(Array):
        nu = Omega*Mu0*(alpha**2)
        NewInput = (mesh,fes,fes2,Theta1E1Sol[:,i],Theta1E2Sol[:,i],Theta1E3Sol[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,counter,NumberofFrequencies)
        Runlist.append(NewInput)
    
    #Run in parallel
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(MPTCalculator, Runlist)
    print(' calculated tensors             ')
    print(' frequency sweep complete')    
    #Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        TensorArray[i,:]=(N0+OutputNumber[0]+1j*OutputNumber[1]).flatten()
        RealEigenvalues[i,:]=np.sort(np.linalg.eigvals(N0+OutputNumber[0]))
        ImaginaryEigenvalues[i,:]=np.sort(np.linalg.eigvals(OutputNumber[1]))
    
    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues
    
    return TensorArray, EigenValues, N0, numelements