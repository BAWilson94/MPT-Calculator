#This file contains functions used for solving transmission problems and calculating tensors
#Functions -Theta0
#          -Theta1
#          -MPTCalculator
#Importing
import numpy as np
from ngsolve import *

    
#Function definition to solve the Theta0 problem
#Output -The solution to the theta0 problem as a (NGSolve) gridfunction
def Theta0(fes,Order,alpha,mu,inout,e,Tolerance,Maxsteps,epsi,simnumber,Solver):
    #print the progress
    try:
        print(' solving theta0 %d/3' % (simnumber), end='\r')
    except:
        print(' solving the theta0 problem', end='\r')
        
    Theta=GridFunction(fes) 
    Theta.Set((0,0,0), BND)
    
    #Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    #Create the bilinear form
    f = LinearForm(fes)
    f += SymbolicLFI(inout*(2*(1-mu**(-1)))*InnerProduct(e,curl(v)))
    a = BilinearForm(fes, symmetric=True, condense=True)
    a += SymbolicBFI((mu**(-1))*(curl(u)*curl(v)))
    a += SymbolicBFI(epsi*(u*v))
    if Solver=="bddc":
        c = Preconditioner(a,"bddc")#Apply the bddc preconditioner
    a.Assemble()
    f.Assemble()
    if Solver=="local":
        c = Preconditioner(a,"local")#Apply the local preconditioner
    c.Update()
    
    #Solve the problem
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * Theta.vec
    inverse= CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
    Theta.vec.data += inverse * res
    Theta.vec.data += a.inner_solve * f.vec
    Theta.vec.data += a.harmonic_extension * Theta.vec
    
    return Theta



#Function definition to solve the Theta1 problem
#Output -The solution to the theta1 problem as a (NGSolve) gridfunction
def Theta1(fes,fes2,Theta0Sol,xi,Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,simnumber,outof,Solver):
    #print the counter
    try:#This is used for the simulations run in parallel
        simnumber.value+=1
        print(' solving theta1 %d/%d    ' % (floor((simnumber.value)/3),outof), end='\r')
    except:
        try:#This is for the simulations run consecutively and the single frequency case
            print(' solving theta1 %d/%d    ' % (simnumber,outof), end='\r')
        except:# This is for the single frequency case with 3 CPUs
            print(' solving the theta1 problem  ', end='\r')
        
    Theta0=GridFunction(fes)
    Theta0.vec.FV().NumPy()[:]=Theta0Sol
    Theta=GridFunction(fes2)
    Theta.Set((0,0,0), BND)

    #Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    #Create the bilinear form
    f = LinearForm(fes2)
    f += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta0,v))
    f += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xi,v))
    a = BilinearForm(fes2, symmetric=True, condense=True)
    a += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
    a += SymbolicBFI((1j) * inout * nu*sigma * InnerProduct(u,v))
    a += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
    if Solver=="bddc":
        c = Preconditioner(a,"bddc")#Apply the bddc preconditioner
    a.Assemble()
    f.Assemble()
    if Solver=="local":
        c = Preconditioner(a,"local")#Apply the local preconditioner
    c.Update()

    #Solve
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * Theta.vec
    inverse= CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
    Theta.vec.data += inverse * res
    Theta.vec.data += a.inner_solve * f.vec
    Theta.vec.data += a.harmonic_extension * Theta.vec

    return Theta


#Function definition to calculate MPTs from solution vectors
#Outputs -R as a numpy array
#        -I as a numpy array (This contains real values not imaginary ones)
def MPTCalculator(mesh,fes,fes2,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,Theta0Sol,xivec,alpha,mu,sigma,inout,nu,tennumber,outof):
    #Print the progress of the sweep
    try:#This is used for the simulations run in parallel
        tennumber.value+=1
        print(' calculating tensor %d/%d    ' % (tennumber.value,outof), end='\r')
    except:#This is for the POD run consecutively
        try:
            print(' calculating tensor %d/%d    ' % (tennumber,outof), end='\r')
        except:#This is for the full sweep run consecutively (no print)
            pass
    
    R=np.zeros([3,3])
    I=np.zeros([3,3])
    Theta0i=GridFunction(fes)
    Theta0j=GridFunction(fes)
    Theta1i=GridFunction(fes2)
    Theta1j=GridFunction(fes2)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:]=Theta0Sol[:,i]
        xii=xivec[i]
        if i==0:
            Theta1i.vec.FV().NumPy()[:]=Theta1E1Sol
        if i==1:
            Theta1i.vec.FV().NumPy()[:]=Theta1E2Sol
        if i==2:
            Theta1i.vec.FV().NumPy()[:]=Theta1E3Sol
        for j in range(i+1):
            Theta0j.vec.FV().NumPy()[:]=Theta0Sol[:,j]
            xij=xivec[j]
            if j==0:
                Theta1j.vec.FV().NumPy()[:]=Theta1E1Sol
            if j==1:
                Theta1j.vec.FV().NumPy()[:]=Theta1E2Sol
            if j==2:
                Theta1j.vec.FV().NumPy()[:]=Theta1E3Sol

            #Real and Imaginary parts
            R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*InnerProduct(curl(Theta1j),Conj(curl(Theta1i))),mesh)).real
            I[i,j]=((alpha**3)/4)*Integrate(inout*nu*sigma*InnerProduct(Theta1j+Theta0j+xij,Conj(Theta1i+Theta0i+xii)),mesh).real
    R+=np.transpose(R-np.diag(np.diag(R))).real
    I+=np.transpose(I-np.diag(np.diag(I))).real
    return R, I
