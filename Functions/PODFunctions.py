#This file contains functions used for creating and solving the reduced order model
#Functions -ROMRightHandSide (This is called from the PODP function)
#          -ROMMatrix (This is called from the PODP function)
#          -PODPMulti
#          -PODP
#Importing
import numpy as np
import multiprocessing
from ngsolve import *


#Function definition to set up the reduced right hand side of the ROM
def ROMRightHandSide(mesh,fes,fes2,Theta0Sol,Theta1Sol,xi,alpha,sigma,mu,inout):
    Mu0=4*np.pi*10**(-7)
    Theta0=GridFunction(fes)
    Theta1=GridFunction(fes2)
    Theta0.vec.FV().NumPy()[:]=Theta0Sol[:]
    Theta1.vec.FV().NumPy()[:]=Theta1Sol[:]
    RightHandSide=-1j*alpha**2*Mu0*Integrate(inout*sigma*InnerProduct(Theta1,Theta0+xi),mesh)
    return RightHandSide



#Function definition to set up the reduced stiffness matrix of the ROM
def ROMMatrix(mesh,fes,fes2,Theta1iSol,Theta1jSol,alpha,sigma,mu,inout,epsi,intnumber,outof):
    #print the integral number
    intnumber.value+=1
    print(' solving integral %d/%d    ' % (floor((intnumber.value)/3),outof), end='\r')
    
    Mu0=4*np.pi*10**(-7)
    Theta1i=GridFunction(fes2)
    Theta1j=GridFunction(fes2)
    Theta1i.vec.FV().NumPy()[:]=Theta1iSol[:]
    Theta1j.vec.FV().NumPy()[:]=Theta1jSol[:]
    MatrixConstant=Integrate(mu**(-1)*InnerProduct(curl(Theta1i),curl(Theta1j)),mesh)+1j*epsi*Integrate((1-inout)*InnerProduct(Theta1i,Theta1j),mesh)
    MatrixVariable=1j*alpha**2*Mu0*Integrate(inout*sigma*InnerProduct(Theta1i,Theta1j),mesh)
    return MatrixConstant, MatrixVariable



#Function definition to use snapshots to produce a full frequency sweep in parallel
#Outputs -Solution vectors for a full frequency sweep (3, 1 for each direction) as numpy arrays
def PODPMulti(mesh,fes,fes2,Theta0SolVec,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,FrequencyArray,ConstructedFrequencyArray,PODtol,CPUs):
    #Print an update on progress
    print(' performing SVD',end='\r')
    #Set up some useful constants
    NumberofFrequencies=len(FrequencyArray)
    NumberofConstructedFrequencies=len(ConstructedFrequencyArray)
    ndof=len(Theta1E1Sol)
    Mu0=4*np.pi*10**(-7)
    
    #Perform SVD on the solution vector matrices
    u1, s1, vh1 = np.linalg.svd(Theta1E1Sol, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(Theta1E2Sol, full_matrices=False)
    u3, s3, vh3 = np.linalg.svd(Theta1E3Sol, full_matrices=False)
    #Print an update on progress
    print(' SVD complete      ')
    
    #scale the value of the modes
    s1norm=s1/s1[0]
    s2norm=s2/s2[0]
    s3norm=s3/s3[0]
    
    #Decide where to truncate
    cutoff=NumberofFrequencies
    for i in range(NumberofFrequencies):
        if s1norm[i]<PODtol:
            if s2norm[i]<PODtol:
                if s3norm[i]<PODtol:
                    cutoff=i
                    break
    
    #Truncate the SVD matrices
    u1Truncated=u1[:,:cutoff]
    s1Truncated=s1[:cutoff]
    vh1Truncated=vh1[:cutoff,:]
    
    u2Truncated=u2[:,:cutoff]
    s2Truncated=s2[:cutoff]
    vh2Truncated=vh2[:cutoff,:]
    
    u3Truncated=u3[:,:cutoff]
    s3Truncated=s3[:cutoff]
    vh3Truncated=vh3[:cutoff,:]
    
    #Turn s into a matrix
    s1mat=np.diag(s1)
    s1Truncatedmat=np.diag(s1Truncated)
    s2Truncatedmat=np.diag(s2Truncated)
    s3Truncatedmat=np.diag(s3Truncated)
    
    #Create where the final solution vectors will be saved
    W1=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    W2=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    W3=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    
    #Create the ROM
    A1Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A1Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R1Variable=np.zeros([cutoff,1],dtype=complex)
    
    A2Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A2Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R2Variable=np.zeros([cutoff,1],dtype=complex)
    
    A3Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A3Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R3Variable=np.zeros([cutoff,1],dtype=complex)
    
    #Create the inputs for the ROM problems
    #Print an update on progress
    print(' creating reduced order model')
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 1)
    total=cutoff*(cutoff+1)/2
    MatrixRunlist=[]
    ForcingRunlist=[]
    for k in range(3):
        if k == 0:
            U=u1Truncated
        elif k ==1:
            U=u2Truncated
        else:
            U=u3Truncated
        for i in range(cutoff):
            #Forcing function
            NewInputForcing=(mesh,fes,fes2,Theta0SolVec[:,k],np.conjugate(U[:,i]),xivec[k],alpha,sigma,mu,inout)
            ForcingRunlist.append(NewInputForcing)
            
            for j in range(i+1):
                #Stifness Matrix
                NewInputMatrix=(mesh,fes,fes2,np.conjugate(U[:,i]),U[:,j],alpha,sigma,mu,inout,epsi,counter,total)
                MatrixRunlist.append(NewInputMatrix)
    
    #Run the integrals for the rhs in parallel
    with multiprocessing.Pool(CPUs) as pool:
        ForcingOutput=pool.starmap(ROMRightHandSide, ForcingRunlist)
    
    #Run the integrals for the matrix in parallel
    with multiprocessing.Pool(CPUs) as pool:
        MatrixOutput=pool.starmap(ROMMatrix, MatrixRunlist)
    #Print an update on progress
    print(' reduced order model created')
    
    #Unpack the reduced order rhs
    for i, Output in enumerate(ForcingOutput):
        direction=int(floor(i/cutoff))
        position=int(i-cutoff*direction)
        if direction==0:
            R1Variable[position,0]=Output
        elif direction==1:
            R2Variable[position,0]=Output
        else:
            R3Variable[position,0]=Output
    
    #Unpack the reduced order matrix
    ival=0
    jval=0
    for Number,Output in enumerate(MatrixOutput):
        direction=int(floor((3*Number)/len(MatrixOutput)))
        if ival==jval:
            if direction==0:
                A1Constant[ival,jval]=Output[0]
                A1Variable[ival,jval]=Output[1]
            elif direction==1:
                A2Constant[ival,jval]=Output[0]
                A2Variable[ival,jval]=Output[1]
            else:
                A3Constant[ival,jval]=Output[0]
                A3Variable[ival,jval]=Output[1]
            if ival==cutoff-1:
                ival=0
                jval=0
            else:
                jval=0
                ival+=1
        else:
            if direction==0:
                A1Constant[ival,jval]=Output[0]
                A1Variable[ival,jval]=Output[1]
                A1Constant[jval,ival]=np.conj(Output[0])
                A1Variable[jval,ival]=-np.conj(Output[1])
            elif direction==1:
                A2Constant[ival,jval]=Output[0]
                A2Variable[ival,jval]=Output[1]
                A2Constant[jval,ival]=np.conj(Output[0])
                A2Variable[jval,ival]=-np.conj(Output[1])
            else:
                A3Constant[ival,jval]=Output[0]
                A3Variable[ival,jval]=Output[1]
                A3Constant[jval,ival]=np.conj(Output[0])
                A3Variable[jval,ival]=-np.conj(Output[1])
            jval+=1
    
    #Solve the reduced order problems and save the new solution vectors
    for i,Omega in enumerate(ConstructedFrequencyArray):
        g1=np.linalg.solve(A1Constant+A1Variable*Omega,R1Variable*Omega)
        g2=np.linalg.solve(A2Constant+A2Variable*Omega,R2Variable*Omega)
        g3=np.linalg.solve(A3Constant+A3Variable*Omega,R3Variable*Omega)
        W1[:,i]=np.dot(u1Truncated,g1).flatten()
        W2[:,i]=np.dot(u2Truncated,g2).flatten()
        W3[:,i]=np.dot(u3Truncated,g3).flatten()
    #Print an update on progress
    print(' reduced order models solved')
    
    return W1, W2, W3



#Function definition to use snapshots to produce a full frequency sweep
#Outputs -Solution vectors for a full frequency sweep (3, 1 for each direction) as numpy arrays
def PODP(mesh,fes,fes2,Theta0SolVec,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,FrequencyArray,ConstructedFrequencyArray,PODtol):
    #Print an update on progress
    print(' performing SVD',end='\r')
    #Set up some useful constants
    NumberofFrequencies=len(FrequencyArray)
    NumberofConstructedFrequencies=len(ConstructedFrequencyArray)
    ndof=len(Theta1E1Sol)
    Mu0=4*np.pi*10**(-7)
    
    #Perform SVD on the solution vector matrices
    u1, s1, vh1 = np.linalg.svd(Theta1E1Sol, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(Theta1E2Sol, full_matrices=False)
    u3, s3, vh3 = np.linalg.svd(Theta1E3Sol, full_matrices=False)
    #Print an update on progress
    print(' SVD complete      ')
    
    #scale the value of the modes
    s1norm=s1/s1[0]
    s2norm=s2/s2[0]
    s3norm=s3/s3[0]
    
    #Decide where to truncate
    cutoff=NumberofFrequencies
    for i in range(NumberofFrequencies):
        if s1norm[i]<PODtol:
            if s2norm[i]<PODtol:
                if s3norm[i]<PODtol:
                    cutoff=i
                    break
    
    #Truncate the SVD matrices
    u1Truncated=u1[:,:cutoff]
    s1Truncated=s1[:cutoff]
    vh1Truncated=vh1[:cutoff,:]
    
    u2Truncated=u2[:,:cutoff]
    s2Truncated=s2[:cutoff]
    vh2Truncated=vh2[:cutoff,:]
    
    u3Truncated=u3[:,:cutoff]
    s3Truncated=s3[:cutoff]
    vh3Truncated=vh3[:cutoff,:]
    
    #Turn s into a matrix
    s1mat=np.diag(s1)
    s1Truncatedmat=np.diag(s1Truncated)
    s2Truncatedmat=np.diag(s2Truncated)
    s3Truncatedmat=np.diag(s3Truncated)
    
    #Create where the final solution vectors will be saved
    W1=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    W2=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    W3=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)

########################################################################
#Create the ROM
    
    #Set up the stiffness matrices and right hand side
    A1Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A1Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R1Variable=np.zeros([cutoff,1],dtype=complex)
    
    A2Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A2Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R2Variable=np.zeros([cutoff,1],dtype=complex)
    
    A3Constant=np.zeros([cutoff,cutoff],dtype=complex)
    A3Variable=np.zeros([cutoff,cutoff],dtype=complex)
    R3Variable=np.zeros([cutoff,1],dtype=complex)
    
    #Print an update on progress
    print(' creating reduced order model')
    
    #Calculate the elements of the ROM (only half due to reflection)
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 1)
    total=cutoff*(cutoff+1)/2
    for i in range(cutoff):

        R1Variable[i,0] = ROMRightHandSide(mesh,fes,fes2,Theta0SolVec[:,0],np.conjugate(u1[:,i]),xivec[0],alpha,sigma,mu,inout)
        R2Variable[i,0] = ROMRightHandSide(mesh,fes,fes2,Theta0SolVec[:,1],np.conjugate(u2[:,i]),xivec[1],alpha,sigma,mu,inout)
        R3Variable[i,0] = ROMRightHandSide(mesh,fes,fes2,Theta0SolVec[:,2],np.conjugate(u3[:,i]),xivec[2],alpha,sigma,mu,inout)
        
        for j in range(i+1):

            A1Constant[i,j], A1Variable[i,j] = ROMMatrix(mesh,fes,fes2,np.conjugate(u1Truncated[:,i]),u1Truncated[:,j],alpha,sigma,mu,inout,epsi,counter,total)
            A2Constant[i,j], A2Variable[i,j] = ROMMatrix(mesh,fes,fes2,np.conjugate(u2Truncated[:,i]),u2Truncated[:,j],alpha,sigma,mu,inout,epsi,counter,total)
            A3Constant[i,j], A3Variable[i,j] = ROMMatrix(mesh,fes,fes2,np.conjugate(u3Truncated[:,i]),u3Truncated[:,j],alpha,sigma,mu,inout,epsi,counter,total)
            
            A1Constant[j,i] = np.conj(A1Constant[i,j])
            A2Constant[j,i] = np.conj(A1Constant[i,j])
            A3Constant[j,i] = np.conj(A1Constant[i,j])
            
            A1Variable[j,i] = -np.conj(A1Variable[i,j])
            A2Variable[j,i] = -np.conj(A2Variable[i,j])
            A3Variable[j,i] = -np.conj(A3Variable[i,j])
            
    #Print an update on progress
    print(' reduced order model created')
    
    #Solve the reduced order problems and save the new solution vectors
    for i,Omega in enumerate(ConstructedFrequencyArray):
        g1=np.linalg.solve(A1Constant+A1Variable*Omega,R1Variable*Omega)
        g2=np.linalg.solve(A2Constant+A2Variable*Omega,R2Variable*Omega)
        g3=np.linalg.solve(A3Constant+A3Variable*Omega,R3Variable*Omega)
        W1[:,i]=np.dot(u1Truncated,g1).flatten()
        W2[:,i]=np.dot(u2Truncated,g2).flatten()
        W3[:,i]=np.dot(u3Truncated,g3).flatten()
    #Print an update on progress
    print(' reduced order models solved')
    
    return W1, W2, W3
