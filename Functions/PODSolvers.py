#This file contains the functions called from the main.py file when Pod=True
#Functions -PODSweep (frequency sweep using POD)
#          -PODSweepMulti (frequency sweep in parallel using POD)
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
from PODFunctions import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters




#Function definition for a frequency sweep which uses the PODP method
#Outputs -An array of Tensors as a numpy array
#        -An array of Eigen values as a numpy array
#def PODSweep(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod):
def PODSweep(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem):
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
    PODN0Errors = np.zeros([3,1])
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
    print(' solved theta0 problems   ',)

    #Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(i+1):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh)))
                if PODErrorBars==True:
                    PODN0Errors[i,0] = (Integrate(inout*InnerProduct(Theta0i,Theta0j),mesh))**(1/2)
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
    fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Work out alphaLB if bounds are required
    if PODErrorBars==True:
        Omega = Array[0]
        u,v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += (mu**(-1))*curl(u)*curl(v)*dx
        amax += (1-inout)*epsi*u*v*dx
        amax += inout*sigma*(alpha**2)*Mu0*Omega*u*v*dx

        m = BilinearForm(fes3)
        m += u*v*dx

        apre = BilinearForm(fes3)
        apre += curl(u)*curl(v)*dx + u*v*dx
        #pre = Preconditioner(apre, "direct", inverse="sparsecholesky")
        pre = Preconditioner(amax, "bddc")
    
        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose() # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat   # multiply matrices
            math1[0,0] += 1     # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")
            
            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)

        alphaLB = evals[0]
    else:
        alphaLB = False
    
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
        Theta1E1Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,0],xivec[0],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver)
        Theta1E2Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,1],xivec[1],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver)
        Theta1E3Sol[:,i] = Theta1(fes,fes2,Theta0Sol[:,2],xivec[2],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,NumberofSnapshots,Solver)
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

    POD_Time = time.time()

#########################################################################
#POD
#This function uses the snapshots from the previous section to calulate
#solution vectors for the points in the desired frequency sweep
    #if PODErrorBars==True:
    #    RealTensors,ImagTensors,ErrorTensors = PODP(mesh,fes0,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol,PODN0Errors,alphaLB,PODErrorBars)
    #else:
    #    RealTensors,ImagTensors = PODP(mesh,fes0,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol,PODN0Errors,alphaLB,PODErrorBars)
    
    
    Theta0SolVec,FrequencyArray,ConstructedFrequencyArray,PODtol,N0Errors = Theta0Sol,PODArray,Array,PODTol,PODN0Errors


    #Print an update on progress
    print(' performing SVD',end='\r')
    #Set up some useful constants
    NumberofFrequencies=len(FrequencyArray)
    NumberofConstructedFrequencies=len(ConstructedFrequencyArray)
    ndof=len(Theta1E1Sol)
    ndof2 = fes2.ndof
    ndof0 = fes0.ndof
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
    #W1=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    #W2=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
    #W3=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)

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
    print(' creating reduced order model',end='\r')
    with TaskManager():
        Mu0=4*np.pi*10**(-7)
        nu=Mu0*(alpha**2)

        Theta_0=GridFunction(fes)

        u = fes0.TrialFunction()
        v = fes0.TestFunction()

        if PODErrorBars==True:
            m = BilinearForm(fes0)
            m += SymbolicBFI(InnerProduct(u,v))
            f = LinearForm(fes0)
            m.Assemble()
            f.Assemble()
            rowsm,colsm,valsm = m.mat.COO()
            M = sp.csc_matrix((valsm,(rowsm,colsm)))

        u = fes2.TrialFunction()
        v = fes2.TestFunction()

        a0 = BilinearForm(fes2)
        a0 += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
        a0 += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
        a1 = BilinearForm(fes2)
        a1 += SymbolicBFI((1j) * inout * nu*sigma * InnerProduct(u,v))

        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,0]
        r1 = LinearForm(fes2)
        r1 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r1 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[0],v))
        r1.Assemble()

        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,1]
        r2 = LinearForm(fes2)
        r2 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r2 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[1],v))
        r2.Assemble()

        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,2]
        r3 = LinearForm(fes2)
        r3 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r3 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[2],v))
        r3.Assemble()

        a0.Assemble()
        a1.Assemble()

        rows0,cols0,vals0 = a0.mat.COO()
        rows1,cols1,vals1 = a1.mat.COO()

    A0 = sp.csr_matrix((vals0,(rows0,cols0)))
    A1 = sp.csr_matrix((vals1,(rows1,cols1)))

    R1 = sp.csr_matrix(r1.vec.FV().NumPy())
    R2 = sp.csr_matrix(r2.vec.FV().NumPy())
    R3 = sp.csr_matrix(r3.vec.FV().NumPy())

    H1=sp.csr_matrix(u1Truncated)
    H2=sp.csr_matrix(u2Truncated)
    H3=sp.csr_matrix(u3Truncated)

    A0H1 = A0@H1
    A1H1 = A1@H1
    A0H2 = A0@H2
    A1H2 = A1@H2
    A0H3 = A0@H3
    A1H3 = A1@H3

    HA0H1 = (np.conjugate(np.transpose(H1))@A0H1).todense()
    HA1H1 = (np.conjugate(np.transpose(H1))@A1H1).todense()
    HR1 = (np.conjugate(np.transpose(H1))@np.transpose(R1)).todense()

    HA0H2 = (np.conjugate(np.transpose(H2))@A0H2).todense()
    HA1H2 = (np.conjugate(np.transpose(H2))@A1H2).todense()
    HR2 = (np.conjugate(np.transpose(H2))@np.transpose(R2)).todense()

    HA0H3 = (np.conjugate(np.transpose(H3))@A0H3).todense()
    HA1H3 = (np.conjugate(np.transpose(H3))@A1H3).todense()
    HR3 = (np.conjugate(np.transpose(H3))@np.transpose(R3)).todense()

    print(' created reduced order model    ')
    if PODErrorBars==True:
        print(' calculating error bars for reduced order model')

        Rerror1 = np.zeros([ndof2,cutoff*2+1],dtype=complex)
        Rerror2 = np.zeros([ndof2,cutoff*2+1],dtype=complex)
        Rerror3 = np.zeros([ndof2,cutoff*2+1],dtype=complex)

        RerrorReduced1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)

        Rerror1[:,0] = R1.todense()
        Rerror2[:,0] = R2.todense()
        Rerror3[:,0] = R3.todense()

        Rerror1[:,1:cutoff+1] = A0H1.todense()
        Rerror2[:,1:cutoff+1] = A0H2.todense()
        Rerror3[:,1:cutoff+1] = A0H3.todense()

        Rerror1[:,cutoff+1:] = A1H1.todense()
        Rerror2[:,cutoff+1:] = A1H2.todense()
        Rerror3[:,cutoff+1:] = A1H3.todense()

        MR1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        MR2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        MR3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)


        with TaskManager():
            ProH = GridFunction(fes2)
            ProL = GridFunction(fes0)

            for i in range(2*cutoff+1):
                ProH.vec.FV().NumPy()[:]=Rerror1[:,i]
                ProL.Set(ProH)
                RerrorReduced1[:,i] = ProL.vec.FV().NumPy()[:]
    
                ProH.vec.FV().NumPy()[:]=Rerror2[:,i]
                ProL.Set(ProH)
                RerrorReduced2[:,i] = ProL.vec.FV().NumPy()[:]
    
                ProH.vec.FV().NumPy()[:]=Rerror3[:,i]
                ProL.Set(ProH)
                RerrorReduced3[:,i] = ProL.vec.FV().NumPy()[:]
    
        try:
            lu = spl.spilu(M,drop_tol=10**-12)
            flagSolver = 0
        except:
            print('Could not solve with  spilu')
            flagSolver = 1

        for i in range(2*cutoff+1):
            if i==0:
                if flagSolver==0:
                    MR1 = sp.csr_matrix(lu.solve(RerrorReduced1[:,i]))
                    MR2 = sp.csr_matrix(lu.solve(RerrorReduced2[:,i]))
                    MR3 = sp.csr_matrix(lu.solve(RerrorReduced3[:,i]))
                else:
                    solMR1,_ = spl.cg(M, RerrorReduced1[:,i], tol=1e-09)
                    solMR2,_ = spl.cg(M, RerrorReduced2[:,i], tol=1e-09)
                    solMR3,_ = spl.cg(M, RerrorReduced3[:,i], tol=1e-09)

                    MR1 = sp.csr_matrix(solMR1)
                    MR2 = sp.csr_matrix(solMR2)
                    MR3 = sp.csr_matrix(solMR3)
            else:    
                if flagSolver==0:
                    MR1 = sp.vstack((MR1,sp.csr_matrix(lu.solve(RerrorReduced1[:,i]))))
                    MR2 = sp.vstack((MR2,sp.csr_matrix(lu.solve(RerrorReduced2[:,i]))))
                    MR3 = sp.vstack((MR3,sp.csr_matrix(lu.solve(RerrorReduced3[:,i]))))
                else:
                    solMR1,_ = spl.cg(M, RerrorReduced1[:,i], tol=1e-09)
                    solMR2,_ = spl.cg(M, RerrorReduced2[:,i], tol=1e-09)
                    solMR3,_ = spl.cg(M, RerrorReduced3[:,i], tol=1e-09)

                    MR1 = sp.vstack((MR1,sp.csr_matrix(solMR1)))
                    MR2 = sp.vstack((MR2,sp.csr_matrix(solMR2)))
                    MR3 = sp.vstack((MR3,sp.csr_matrix(solMR3)))


        G1 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR1)
        G2 = np.transpose(np.conjugate(RerrorReduced2))@np.transpose(MR2)
        G3 = np.transpose(np.conjugate(RerrorReduced3))@np.transpose(MR3)
        G12 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR2)
        G13 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR3)
        G23 = np.transpose(np.conjugate(RerrorReduced2))@np.transpose(MR3)

        rom1 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom2 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom3 = np.zeros([1+2*cutoff,1],dtype=complex)

        TensorErrors=np.zeros([NumberofConstructedFrequencies,3])
        ErrorTensors=np.zeros([NumberofConstructedFrequencies,6])
        ErrorTensor =np.zeros([3,3])


########################################################################
#Produce the sweep on the lower dimensional space

    for i,omega in enumerate(ConstructedFrequencyArray):

        #This part is for obtaining the solutions in the lower dimensional space
        #print(' solving reduced order system %d/%d    ' % (i+1,NumberofConstructedFrequencies), end='\r')

        g1=np.linalg.solve(HA0H1+HA1H1*omega,HR1*omega)
        g2=np.linalg.solve(HA0H2+HA1H2*omega,HR2*omega)
        g3=np.linalg.solve(HA0H3+HA1H3*omega,HR3*omega)
        
        
        #This part projects the problem to the higher dimensional space
        W1=np.dot(u1Truncated,g1).flatten()
        W2=np.dot(u2Truncated,g2).flatten()
        W3=np.dot(u3Truncated,g3).flatten()
        
        nu = omega*Mu0*(alpha**2)
        R,I = MPTCalculator(mesh,fes,fes2,W1,W2,W3,Theta0Sol,xivec,alpha,mu,sigma,inout,nu,i,NumberofConstructedFrequencies)
        TensorArray[i,:] = (N0+R+1j*I).flatten()
        RealEigenvalues[i,:] = np.sort(np.linalg.eigvals(N0+R))
        ImaginaryEigenvalues[i,:] = np.sort(np.linalg.eigvals(I))

    
        if PODErrorBars==True:
            rom1[0,0] = omega
            rom2[0,0] = omega
            rom3[0,0] = omega

            rom1[1:1+cutoff,0] = -g1.flatten()
            rom2[1:1+cutoff,0] = -g2.flatten()
            rom3[1:1+cutoff,0] = -g3.flatten()

            rom1[1+cutoff:,0] = -(g1*omega).flatten()
            rom2[1+cutoff:,0] = -(g2*omega).flatten()
            rom3[1+cutoff:,0] = -(g3*omega).flatten()

            error1 = np.conjugate(np.transpose(rom1))@G1@rom1
            error2 = np.conjugate(np.transpose(rom2))@G2@rom2
            error3 = np.conjugate(np.transpose(rom3))@G3@rom3
            error12 = np.conjugate(np.transpose(rom1))@G12@rom2
            error13 = np.conjugate(np.transpose(rom1))@G13@rom3
            error23 = np.conjugate(np.transpose(rom2))@G23@rom3
    
            error1 = abs(error1)**(1/2)
            error2 = abs(error2)**(1/2)
            error3 = abs(error3)**(1/2)
            error12 = error12.real
            error13 = error13.real
            error23 = error23.real
    
            Errors=[error1,error2,error3,error12,error13,error23]
            
            for j in range(6):
                if j<3:
                    ErrorTensors[i,j] = ((alpha**3)/4)*(Errors[j]**2)/alphaLB
                else:
                    ErrorTensors[i,j] = -2*Errors[j]
                    if j==3:
                        ErrorTensors[i,j] += (Errors[0]**2)+(Errors[1]**2)
                        ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                    if j==4:
                        ErrorTensors[i,j] += (Errors[0]**2)+(Errors[2]**2)
                        ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                    if j==5:
                        ErrorTensors[i,j] += (Errors[1]**2)+(Errors[2]**2)
                        ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])


    print(' reduced order systems solved        ')
    print(' frequency sweep complete')    

    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues
    
    if PlotPod==True:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements
    else:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors
        else:
            return TensorArray, EigenValues, N0, numelements




#Function definition for a frequency sweep which uses the PODP method in parallel
#Outputs -An array of Tensors as a numpy array
#        -An array of Eigen values as a numpy array
#def PODSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs):
def PODSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem):
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
    PODN0Errors = np.zeros([3,1])
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
                if PODErrorBars==True:
                    PODN0Errors[i,0] = (Integrate(inout*InnerProduct(Theta0i,Theta0j),mesh))**(1/2)
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh))



#########################################################################
#Theta1
#This section solves the Theta1 problem to calculate the solution vectors
#of the snapshots

    #Setup the finite element spaces
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    
    #Work out alphaLB if bounds are required
    if PODErrorBars==True:
        Omega = Array[0]
        u,v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += (mu**(-1))*curl(u)*curl(v)*dx
        amax += (1-inout)*epsi*u*v*dx
        amax += inout*sigma*(alpha**2)*Mu0*Omega*u*v*dx

        m = BilinearForm(fes3)
        m += u*v*dx

        apre = BilinearForm(fes3)
        apre += curl(u)*curl(v)*dx + u*v*dx
        #pre = Preconditioner(apre, "direct", inverse="sparsecholesky")
        pre = Preconditioner(amax, "bddc")
    
        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose() # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat   # multiply matrices
            math1[0,0] += 1     # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")
            
            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)

        alphaLB = evals[0]
    else:
        alphaLB = False
    
    
    
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
            Theta1E1Sol[:,position] = OutputNumber
        if direction==1:
            Theta1E2Sol[:,position] = OutputNumber
        if direction==2:
            Theta1E3Sol[:,position] = OutputNumber
    


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
    #if PODErrorBars==True:
    #    RealTensors,ImagTensors,ErrorTensors = PODP(mesh,fes0,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol,PODN0Errors,alphaLB,PODErrorBars)
    #else:
    #    RealTensors,ImagTensors = PODP(mesh,fes0,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,epsi,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,PODArray,Array,PODTol,PODN0Errors,alphaLB,PODErrorBars)
    
    
    Theta0SolVec,FrequencyArray,ConstructedFrequencyArray,PODtol,N0Errors = Theta0Sol,PODArray,Array,PODTol,PODN0Errors
    
    
    #Print an update on progress
    print(' performing SVD',end='\r')
    #Set up some useful constants
    NumberofFrequencies=len(FrequencyArray)
    NumberofConstructedFrequencies=len(ConstructedFrequencyArray)
    ndof=len(Theta1E1Sol)
    ndof2 = fes2.ndof
    ndof0 = fes0.ndof
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
    print(' creating reduced order model',end='\r')
    with TaskManager():
        Mu0=4*np.pi*10**(-7)
        nu=Mu0*(alpha**2)

        Theta_0=GridFunction(fes)
        Theta_1=GridFunction(fes2)
    
        u = fes0.TrialFunction()
        v = fes0.TestFunction()
    
        if PODErrorBars==True:
            m = BilinearForm(fes0)
            m += SymbolicBFI(InnerProduct(u,v))
            f = LinearForm(fes0)
            m.Assemble()
            f.Assemble()
            rowsm,colsm,valsm = m.mat.COO()
            M = sp.csc_matrix((valsm,(rowsm,colsm)))
    
        u = fes2.TrialFunction()
        v = fes2.TestFunction()
    
        a0 = BilinearForm(fes2)
        a0 += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
        a0 += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
        a1 = BilinearForm(fes2)
        a1 += SymbolicBFI((1j) * inout * nu*sigma * InnerProduct(u,v))
    
        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,0]
        r1 = LinearForm(fes2)
        r1 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r1 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[0],v))
        r1.Assemble()
        ROM_Vec = r1.vec.CreateVector()
    
        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,1]
        r2 = LinearForm(fes2)
        r2 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r2 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[1],v))
        r2.Assemble()
    
        Theta_0.vec.FV().NumPy()[:]=Theta0SolVec[:,2]
        r3 = LinearForm(fes2)
        r3 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta_0,v))
        r3 += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xivec[2],v))
        r3.Assemble()

        a0.Assemble()
        a1.Assemble()
    
        rows0,cols0,vals0 = a0.mat.COO()
        rows1,cols1,vals1 = a1.mat.COO()
    
    try:
        A0 = sp.csr_matrix((vals0,(rows0,cols0)))
        A1 = sp.csr_matrix((vals1,(rows1,cols1)))

        R1 = sp.csr_matrix(r1.vec.FV().NumPy())
        R2 = sp.csr_matrix(r2.vec.FV().NumPy())
        R3 = sp.csr_matrix(r3.vec.FV().NumPy())

        H1=sp.csr_matrix(u1Truncated)
        H2=sp.csr_matrix(u2Truncated)
        H3=sp.csr_matrix(u3Truncated)

        A0H1 = A0@H1
        A1H1 = A1@H1
        A0H2 = A0@H2
        A1H2 = A1@H2
        A0H3 = A0@H3
        A1H3 = A1@H3
        
        HA0H1 = (np.conjugate(np.transpose(H1))@A0H1).todense()
        HA1H1 = (np.conjugate(np.transpose(H1))@A1H1).todense()
        HR1 = (np.conjugate(np.transpose(H1))@np.transpose(R1)).todense()

        HA0H2 = (np.conjugate(np.transpose(H2))@A0H2).todense()
        HA1H2 = (np.conjugate(np.transpose(H2))@A1H2).todense()
        HR2 = (np.conjugate(np.transpose(H2))@np.transpose(R2)).todense()

        HA0H3 = (np.conjugate(np.transpose(H3))@A0H3).todense()
        HA1H3 = (np.conjugate(np.transpose(H3))@A1H3).todense()
        HR3 = (np.conjugate(np.transpose(H3))@np.transpose(R3)).todense()

    except:
        time1 = time.time()
        A0H1 = np.zeros([ndof,cutoff],dtype=complex)
        A1H1 = np.zeros([ndof,cutoff],dtype=complex)
        A0H2 = np.zeros([ndof,cutoff],dtype=complex)
        A1H2 = np.zeros([ndof,cutoff],dtype=complex)
        A0H3 = np.zeros([ndof,cutoff],dtype=complex)
        A1H3 = np.zeros([ndof,cutoff],dtype=complex)
        
        for i in range(cutoff):
            Theta_1.vec.FV().NumPy()[:] = u1Truncated[:,i]
            ROM_Vec.data = a0.mat * Theta_1.vec
            A0H1[:,i] = ROM_Vec.FV().NumPy()
            ROM_Vec.data = a1.mat * Theta_1.vec
            A1H1[:,i] = ROM_Vec.FV().NumPy()
        for i in range(cutoff):
            Theta_1.vec.FV().NumPy()[:] = u2Truncated[:,i]
            ROM_Vec.data = a0.mat * Theta_1.vec
            A0H2[:,i] = ROM_Vec.FV().NumPy()
            ROM_Vec.data = a1.mat * Theta_1.vec
            A1H2[:,i] = ROM_Vec.FV().NumPy()
        for i in range(cutoff):
            Theta_1.vec.FV().NumPy()[:] = u3Truncated[:,i]
            ROM_Vec.data = a0.mat * Theta_1.vec
            A0H3[:,i] = ROM_Vec.FV().NumPy()
            ROM_Vec.data = a1.mat * Theta_1.vec
            A1H3[:,i] = ROM_Vec.FV().NumPy()
        
        R1 = r1.vec.FV().NumPy()
        R2 = r2.vec.FV().NumPy()
        R3 = r3.vec.FV().NumPy()
        
        HA0H1 = np.conjugate(np.transpose(u1Truncated))@A0H1
        HA1H1 = np.conjugate(np.transpose(u1Truncated))@A1H1
        HR1 =   np.conjugate(np.transpose(u1Truncated))@np.transpose(R1)

        HA0H2 = np.conjugate(np.transpose(u2Truncated))@A0H2
        HA1H2 = np.conjugate(np.transpose(u2Truncated))@A1H2
        HR2 =   np.conjugate(np.transpose(u2Truncated))@np.transpose(R2)

        HA0H3 = np.conjugate(np.transpose(u3Truncated))@A0H3
        HA1H3 = np.conjugate(np.transpose(u3Truncated))@A1H3
        HR3 =   np.conjugate(np.transpose(u3Truncated))@np.transpose(R3)
    

    print(' created reduced order model    ')
    if PODErrorBars==True:
        print(' calculating error bars for reduced order model')

        Rerror1 = np.zeros([ndof2,cutoff*2+1],dtype=complex)
        Rerror2 = np.zeros([ndof2,cutoff*2+1],dtype=complex)
        Rerror3 = np.zeros([ndof2,cutoff*2+1],dtype=complex)

        RerrorReduced1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)

        Rerror1[:,0] = R1.todense()
        Rerror2[:,0] = R2.todense()
        Rerror3[:,0] = R3.todense()

        Rerror1[:,1:cutoff+1] = A0H1.todense()
        Rerror2[:,1:cutoff+1] = A0H2.todense()
        Rerror3[:,1:cutoff+1] = A0H3.todense()

        Rerror1[:,cutoff+1:] = A1H1.todense()
        Rerror2[:,cutoff+1:] = A1H2.todense()
        Rerror3[:,cutoff+1:] = A1H3.todense()

        MR1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        MR2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        MR3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)


        with TaskManager():
            ProH = GridFunction(fes2)
            ProL = GridFunction(fes0)
    
            for i in range(2*cutoff+1):
                ProH.vec.FV().NumPy()[:]=Rerror1[:,i]
                ProL.Set(ProH)
                RerrorReduced1[:,i] = ProL.vec.FV().NumPy()[:]
        
                ProH.vec.FV().NumPy()[:]=Rerror2[:,i]
                ProL.Set(ProH)
                RerrorReduced2[:,i] = ProL.vec.FV().NumPy()[:]
        
                ProH.vec.FV().NumPy()[:]=Rerror3[:,i]
                ProL.Set(ProH)
                RerrorReduced3[:,i] = ProL.vec.FV().NumPy()[:]
        
        try:
            lu = spl.spilu(M,drop_tol=10**-12)
            flagSolver = 0
        except:
            print('Could not solve with  spilu')
            flagSolver = 1

        for i in range(2*cutoff+1):
            if i==0:
                if flagSolver==0:
                    MR1 = sp.csr_matrix(lu.solve(RerrorReduced1[:,i]))
                    MR2 = sp.csr_matrix(lu.solve(RerrorReduced2[:,i]))
                    MR3 = sp.csr_matrix(lu.solve(RerrorReduced3[:,i]))
                else:
                    solMR1,_ = spl.cg(M, RerrorReduced1[:,i], tol=1e-09)
                    solMR2,_ = spl.cg(M, RerrorReduced2[:,i], tol=1e-09)
                    solMR3,_ = spl.cg(M, RerrorReduced3[:,i], tol=1e-09)

                    MR1 = sp.csr_matrix(solMR1)
                    MR2 = sp.csr_matrix(solMR2)
                    MR3 = sp.csr_matrix(solMR3)
            else:    
                if flagSolver==0:
                    MR1 = sp.vstack((MR1,sp.csr_matrix(lu.solve(RerrorReduced1[:,i]))))
                    MR2 = sp.vstack((MR2,sp.csr_matrix(lu.solve(RerrorReduced2[:,i]))))
                    MR3 = sp.vstack((MR3,sp.csr_matrix(lu.solve(RerrorReduced3[:,i]))))
                else:
                    solMR1,_ = spl.cg(M, RerrorReduced1[:,i], tol=1e-09)
                    solMR2,_ = spl.cg(M, RerrorReduced2[:,i], tol=1e-09)
                    solMR3,_ = spl.cg(M, RerrorReduced3[:,i], tol=1e-09)

                    MR1 = sp.vstack((MR1,sp.csr_matrix(solMR1)))
                    MR2 = sp.vstack((MR2,sp.csr_matrix(solMR2)))
                    MR3 = sp.vstack((MR3,sp.csr_matrix(solMR3)))


        G1 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR1)
        G2 = np.transpose(np.conjugate(RerrorReduced2))@np.transpose(MR2)
        G3 = np.transpose(np.conjugate(RerrorReduced3))@np.transpose(MR3)
        G12 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR2)
        G13 = np.transpose(np.conjugate(RerrorReduced1))@np.transpose(MR3)
        G23 = np.transpose(np.conjugate(RerrorReduced2))@np.transpose(MR3)
        
        rom1 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom2 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom3 = np.zeros([1+2*cutoff,1],dtype=complex)

        TensorErrors=np.zeros([NumberofConstructedFrequencies,3])
        ErrorTensors=np.zeros([NumberofConstructedFrequencies,6])
        ErrorTensor =np.zeros([3,3])


########################################################################
#Produce the sweep on the lower dimensional space
    try:
        #Create where the final solution vectors will be saved
        W1=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
        W2=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
        W3=np.zeros([ndof,NumberofConstructedFrequencies],dtype=complex)
        for i,omega in enumerate(ConstructedFrequencyArray):
    
            #This part is for obtaining the solutions in the lower dimensional space
            #print(' solving reduced order system %d/%d    ' % (i+1,NumberofConstructedFrequencies), end='\r')
    
            g1=np.linalg.solve(HA0H1+HA1H1*omega,HR1*omega)
            g2=np.linalg.solve(HA0H2+HA1H2*omega,HR2*omega)
            g3=np.linalg.solve(HA0H3+HA1H3*omega,HR3*omega)
    
            #This part projects the problem to the higher dimensional space
            W1[:,i]=np.dot(u1Truncated,g1).flatten()
            W2[:,i]=np.dot(u2Truncated,g2).flatten()
            W3[:,i]=np.dot(u3Truncated,g3).flatten()
    
        
            if PODErrorBars==True:
                rom1[0,0] = omega
                rom2[0,0] = omega
                rom3[0,0] = omega

                rom1[1:1+cutoff,0] = -g1.flatten()
                rom2[1:1+cutoff,0] = -g2.flatten()
                rom3[1:1+cutoff,0] = -g3.flatten()

                rom1[1+cutoff:,0] = -(g1*omega).flatten()
                rom2[1+cutoff:,0] = -(g2*omega).flatten()
                rom3[1+cutoff:,0] = -(g3*omega).flatten()

                error1 = np.conjugate(np.transpose(rom1))@G1@rom1
                error2 = np.conjugate(np.transpose(rom2))@G2@rom2
                error3 = np.conjugate(np.transpose(rom3))@G3@rom3
                error12 = np.conjugate(np.transpose(rom1))@G12@rom2
                error13 = np.conjugate(np.transpose(rom1))@G13@rom3
                error23 = np.conjugate(np.transpose(rom2))@G23@rom3
        
                error1 = abs(error1)**(1/2)
                error2 = abs(error2)**(1/2)
                error3 = abs(error3)**(1/2)
                error12 = error12.real
                error13 = error13.real
                error23 = error23.real
                
                Errors=[error1,error2,error3,error12,error13,error23]
                
                for j in range(6):
                    if j<3:
                        ErrorTensors[i,j] = ((alpha**3)/4)*(Errors[j]**2)/alphaLB
                    else:
                        ErrorTensors[i,j] = -2*Errors[j]
                        if j==3:
                            ErrorTensors[i,j] += (Errors[0]**2)+(Errors[1]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                        if j==4:
                            ErrorTensors[i,j] += (Errors[0]**2)+(Errors[2]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                        if j==5:
                            ErrorTensors[i,j] += (Errors[1]**2)+(Errors[2]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])


        print(' reduced order systems solved        ')

        #########################################################################
        #Calculate the tensors
        #Calculate the tensors and eigenvalues which will be used in the sweep
    
        #Create the inputs for the calculation of the tensors
        Runlist = []
        counter = manager.Value('i', 0)
        for i,Omega in enumerate(Array):
            nu = Omega*Mu0*(alpha**2)
            NewInput = (mesh,fes,fes2,W1[:,i],W2[:,i],W3[:,i],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,counter,NumberofConstructedFrequencies)
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
    
    except:
        for i,omega in enumerate(ConstructedFrequencyArray):
    
            g1=np.linalg.solve(HA0H1+HA1H1*omega,HR1*omega)
            g2=np.linalg.solve(HA0H2+HA1H2*omega,HR2*omega)
            g3=np.linalg.solve(HA0H3+HA1H3*omega,HR3*omega)
    
            #This part projects the problem to the higher dimensional space
            W1=np.dot(u1Truncated,g1).flatten()
            W2=np.dot(u2Truncated,g2).flatten()
            W3=np.dot(u3Truncated,g3).flatten()
            
            
            nu = omega*Mu0*(alpha**2)
            R,I = MPTCalculator(mesh,fes,fes2,W1,W2,W3,Theta0Sol,xivec,alpha,mu,sigma,inout,nu,i,NumberofConstructedFrequencies)
            TensorArray[i,:] = (N0+R+1j*I).flatten()
            RealEigenvalues[i,:] = np.sort(np.linalg.eigvals(N0+R))
            ImaginaryEigenvalues[i,:] = np.sort(np.linalg.eigvals(I))
            
            if PODErrorBars==True:
                rom1[0,0] = omega
                rom2[0,0] = omega
                rom3[0,0] = omega

                rom1[1:1+cutoff,0] = -g1.flatten()
                rom2[1:1+cutoff,0] = -g2.flatten()
                rom3[1:1+cutoff,0] = -g3.flatten()

                rom1[1+cutoff:,0] = -(g1*omega).flatten()
                rom2[1+cutoff:,0] = -(g2*omega).flatten()
                rom3[1+cutoff:,0] = -(g3*omega).flatten()

                error1 = np.conjugate(np.transpose(rom1))@G1@rom1
                error2 = np.conjugate(np.transpose(rom2))@G2@rom2
                error3 = np.conjugate(np.transpose(rom3))@G3@rom3
                error12 = np.conjugate(np.transpose(rom1))@G12@rom2
                error13 = np.conjugate(np.transpose(rom1))@G13@rom3
                error23 = np.conjugate(np.transpose(rom2))@G23@rom3
        
                error1 = abs(error1)**(1/2)
                error2 = abs(error2)**(1/2)
                error3 = abs(error3)**(1/2)
                error12 = error12.real
                error13 = error13.real
                error23 = error23.real
        
                Errors=[error1,error2,error3,error12,error13,error23]
                
                for j in range(6):
                    if j<3:
                        ErrorTensors[i,j] = ((alpha**3)/4)*(Errors[j]**2)/alphaLB
                    else:
                        ErrorTensors[i,j] = -2*Errors[j]
                        if j==3:
                            ErrorTensors[i,j] += (Errors[0]**2)+(Errors[1]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                        if j==4:
                            ErrorTensors[i,j] += (Errors[0]**2)+(Errors[2]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
                        if j==5:
                            ErrorTensors[i,j] += (Errors[1]**2)+(Errors[2]**2)
                            ErrorTensors[i,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[i,j])
            
        
    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues

    #Volume = Integrate(inout,mesh)
    #print(Volume)
    
    if PlotPod==True:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements
    else:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors
        else:
            return TensorArray, EigenValues, N0, numelements
