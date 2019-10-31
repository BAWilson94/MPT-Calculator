#This file contains the functions called from the main.py file when Pod=True
#Functions -PODSweep (frequency sweep using POD)
#          -PODSweepMulti (frequency sweep in parallel using POD)
#Importing
import os
import sys
import time
import multiprocessing

import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from MPTFunctions import *
from PODFunctions import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters




#Function definition for a frequency sweep which uses the PODP method
#Outputs -An array of Tensors as a numpy array
#        -An array of Eigen values as a numpy array
def PODSweep(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod):
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
    NumberofSnapshots = len(PODArray)
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
        Theta0Sol[:,i] = Theta0(fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver).vec.FV().NumPy()
    print(' solved theta0 problems   ',)

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
    
    Theta1E1Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    Theta1E2Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    Theta1E3Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    
    #Setup the inputs for the functions to run
    print(' taking snapshots')
    for i,Omega in enumerate(PODArray):
        nu = Omega*Mu0*(alpha**2)
        Theta1E1Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,0],xivec[0],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver).vec.FV().NumPy()
        Theta1E2Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,1],xivec[1],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver).vec.FV().NumPy()
        Theta1E3Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,2],xivec[2],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver).vec.FV().NumPy()
    print(' snapshots taken          ',)



#########################################################################
#Calculate snapshot tensors
#Calculate the tensors and eigenvalues for the snapshots if necassary
    if PlotPod==True:
        print(' calculating snapshot tensors')
        #Create a place to store the tensors and eigenvalues
        PODTensors = np.zeros([NumberofSnapshots,9],dtype=complex)
        PODEigenValues = np.zeros([NumberofSnapshots,3],dtype=complex)
        #Calculate tensors for each of the frequencies in the array
        for i,Omega in enumerate(PODArray):
            nu = Omega*Mu0*(alpha**2)
            R,I = MPTCalculator(mesh,fes,fes2,Theta1E1Sol[:,i],Theta1E2Sol[:,i],Theta1E3Sol[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,i+1,NumberofSnapshots)
            PODTensors[i,:] = (N0+R+1j*I).flatten()
            RealEig = np.sort(np.linalg.eigvals(N0+R))
            ImaginaryEig = np.sort(np.linalg.eigvals(I))
            PODEigenValues[i,:] = RealEig+1j*ImaginaryEig
        print(' calculated snapshot tensors        ',)



#########################################################################
#POD
#This function uses the snapshots from the previous section to calulate
#solution vectors for the points in the desired frequency sweep
    W1,W2,W3 = PODP(mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol)



#########################################################################
#Calculate the tensors
#Calculate the tensors and eigenvalues which will be used in the sweep
    
    #Calculate tensors for each of the frequencies in the array
    for i,Omega in enumerate(Array):
        nu = Omega*Mu0*(alpha**2)
        R,I = MPTCalculator(mesh,fes,fes2,W1[:,i],W2[:,i],W3[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,i+1,NumberofFrequencies)
        TensorArray[i,:] = (N0+R+1j*I).flatten()
        RealEigenvalues[i,:] = np.sort(np.linalg.eigvals(N0+R))
        ImaginaryEigenvalues[i,:] = np.sort(np.linalg.eigvals(I))
    print(' tensors calculated        ',)
    print(' frequency sweep complete     ')
    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues
    
    if PlotPod==True:
        return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements
    else:
        return TensorArray, EigenValues, N0, numelements




#Function definition for a frequency sweep which uses the PODP method in parallel
#Outputs -An array of Tensors as a numpy array
#        -An array of Eigen values as a numpy array
def PODSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs):
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
    NumberofSnapshots = len(PODArray)
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
        Theta0Sol[:,i] = Direction.vec.FV().NumPy()


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
    
    Theta1E1Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    Theta1E2Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    Theta1E3Sol = np.zeros([ndof2,NumberofSnapshots],dtype=complex)
    
    #Setup the inputs for the functions to run
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 1)
    print(' taking snapshots')
    for Omega in PODArray:
        nu = Omega*Mu0*(alpha**2)
        for i in range(3):
            NewInput = (fes,fes2,Theta0Sol[:,i],xivec[i],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,counter,NumberofSnapshots,Solver)
            Runlist.append(NewInput)
    
    #Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(Theta1, Runlist)
    print(' snapshots taken          ')
    
    #Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        position = int(floor(i/3))
        direction = i-3*position
        if direction==0:
            Theta1E1Sol[:,position] = OutputNumber.vec.FV().NumPy()
        if direction==1:
            Theta1E2Sol[:,position] = OutputNumber.vec.FV().NumPy()
        if direction==2:
            Theta1E3Sol[:,position] = OutputNumber.vec.FV().NumPy()
    


#########################################################################
#Calculate snapshot tensors
#Calculate the tensors and eigenvalues for the snapshots if necassary
    if PlotPod==True:
        print(' calculating snapshot tensors')
        #Create a place to store the tensors and eigenvalues
        PODTensors = np.zeros([NumberofSnapshots,9],dtype=complex)
        PODEigenValues = np.zeros([NumberofSnapshots,3],dtype=complex)
        #Create the inputs for the calculation of the tensors
        Runlist = []
        counter = manager.Value('i', 0)
        for i,Omega in enumerate(PODArray):
            nu = Omega*Mu0*(alpha**2)
            NewInput = (mesh,fes,fes2,Theta1E1Sol[:,i],Theta1E2Sol[:,i],Theta1E3Sol[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,counter,NumberofSnapshots)
            Runlist.append(NewInput)
    
        #Run in parallel
        with multiprocessing.Pool(CPUs) as pool:
            Output = pool.starmap(MPTCalculator, Runlist)
        print(' calculated snapshot tensors             ')
        
        #Unpack the outputs
        for i, OutputNumber in enumerate(Output):
            PODTensors[i,:]=(N0+OutputNumber[0]+1j*OutputNumber[1]).flatten()
            RealEig=np.sort(np.linalg.eigvals(N0+OutputNumber[0]))
            ImaginaryEig=np.sort(np.linalg.eigvals(OutputNumber[1]))
            PODEigenValues[i,:] = RealEig+1j*ImaginaryEig


#########################################################################
#POD
#This function uses the snapshots from the previous section to calulate
#solution vectors for the points in the desired frequency sweep
    W1,W2,W3 = PODPMulti(mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol,CPUs)



#########################################################################
#Calculate the tensors
#Calculate the tensors and eigenvalues which will be used in the sweep
    
    #Create the inputs for the calculation of the tensors
    Runlist = []
    counter = manager.Value('i', 0)
    for i,Omega in enumerate(Array):
        nu = Omega*Mu0*(alpha**2)
        NewInput = (mesh,fes,fes2,W1[:,i],W2[:,i],W3[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,counter,NumberofFrequencies)
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
    
    if PlotPod==True:
        return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements
    else:
        return TensorArray, EigenValues, N0, numelements
