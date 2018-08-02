#! /usr/bin/env python3

import matplotlib                                                               
matplotlib.use('Agg')                                                           
import matplotlib.pyplot as plt                                                 
from mpl_toolkits.mplot3d import Axes3D                                         
from matplotlib.backends.backend_pdf import PdfPages                            
import matplotlib.patches as patches                                            
from matplotlib.colors import ListedColormap                                    
import matplotlib.gridspec as gridspec                                          
                                                                                
from mpl_toolkits.axes_grid1 import make_axes_locatable                         
import mpl_toolkits.axes_grid.axes_size as Size                                 
from mpl_toolkits.axes_grid import Divider                                      
                                                                                
from matplotlib.pyplot import cm                                                
matplotlib.rcParams.update({'font.size': 9.0})   

import scipy
from scipy.stats import moment
from scipy.stats import gamma
from scipy.optimize import curve_fit
from copy import deepcopy
import numpy as np
import os,sys
import re
import math
import json
import pickle
#import glob,multiprocessing

try:
    import gmpy2
    from gmpy2 import mpfr
except:
    raise Warning("gmpy2 library not found, will not be able to calculate isotherms")

try:
    import preos
except:
    raise Warning("preos (Peng Robinson EOS) not found, will not be able to calculate isotherms")


ANGSTROM            =1e-10              # m                                     
AVOGADRO_CONSTANT   =6.0221419947e23    # mol^-1                                
PLANCK_CONSTANT     =6.6260687652e-34   # J.s                                   
BOLTZMANN_CONSTANT  =1.380650324e-23    # J K^-1 
MOLAR_GAS_CONSTANT  =8.314464919        # J mol^-1 K^-1


def get_flist(fname):
    f = open(fname, "r")
    flist=f.read()
    flist=flist.strip().split('\n')
    return flist

def get_constants(constants_fname):

    f = open(constants_fname, "r")
    constants={}
    
    for line in f.readlines():
        if(re.match(r'\s*#', line)):
            pass
        else:
            parsed = line.strip().split()
            if(len(parsed)==2):
                try:
                    constants[parsed[0]]=float(parsed[1])
                except:
                    constants[parsed[0]]=parsed[1]
            elif(len(parsed)>2):
                constants[parsed[0]]=[float(parsed[i]) for i in range(1,len(parsed))]

    return constants

def maxwell_function(x, a, b, c):

    y_data = a*np.power(x-b,2)*np.exp(-np.power(x-b,2)/c)
    # makes the function (could also multiply by heaviside)
    y_data[x<b]=0
    return y_data
    #if(x<c):
    #    return 0
    #else:
    #    return a*np.power(x-c,2)*exp(-np.power(x-c,2)/b)

def GEV_function(x, mu, sigma, zeta):
    """
    Generalized extreme value function
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """ 
    # Note zeta must be > 0
    # Support is x in [mu -sigma/zeta, +inf) when zeta > 0

    t=np.power(1+zeta*(x-mu)/sigma,-1/zeta)
    y_data=1/sigma*np.power(t,zeta+1)*np.exp(-t)
    y_data[x<=mu-sigma/zeta]=0.0
    return y_data
    

def scipy_fit_dist(x, y, dist_name="norm"):
    """
    Fit raw data (y) to a scipy distribution
    For now only consider distributions whose support is (0,inf)
    """
    #num_samples=len(y) # num sample points
    #size=max(y)-min(y) # range over which the data should be fit 
    #print(size)
    #x=scipy.arange(size+1)
    #print(x)
    dist=getattr(scipy.stats,dist_name)
    param=dist.fit(y)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    # mean, variance, skew, kurtosis
    moments = dist.stats(*param[:-2], loc=param[-2], scale=param[-1], moments="mvsk")
    # IMPORTANT: need to adjust skew, kurtosis to get the actual central moment 
    moments    = [float(moments[i]) for i in range(len(moments))]
    moments[2] = moments[2]*np.power(moments[1],3/2)
    moments[3] = (moments[3]+3)*np.power(moments[1],2)


    # IMPRTANT: this would give the NON-CENTRAL moment of order n    
    #moments = [dist(*param[:-2], loc=param[-2], scale=param[-1]).moment(i) for i in range(0,4)]
    return param, pdf_fitted, [0]+moments


def np_readfile(fname):
    pass


class EOS(object):

    def __init__(self, constants):

        self.constants = constants
        self.NumOfComps=int(self.constants['numcomps'])
        self.comps=[i for i in range(self.NumOfComps)]
        
        # setup critical constants array
        self.extract_critical_constants()

        ## Different pressures to obtain fugacity coeffs
        #if('Pstart' in self.constants.keys() and\
        #   'Pend' in self.constants.keys() and\
        #   'Pdel' in self.constants.keys()):
        #    self.P = np.arange(self.constants['Pstart'],self.constants['Pend'],self.constants['Pdel'])
        #elif('pressure' in self.constants.keys()):
        #    self.P = self.constants['pressure']
        #else:
        #    raise ValueError("No pressure provided in your constants dictionary")
        #self.NumOfPs = len(self.P)

        ## Different temperatures to obtain fugacity coeffs
        #if('T_extrap' in self.constants.keys()):
        #    self.T = self.constants['T_extrap']
        #elif('temperature' in self.constants.keys()):
        #    self.T = self.constants['temperature']
        #else:
        #    raise ValueError("No temperature provided in your constants dictionary")
        #self.NumOfTs = len(self.T)


        #self.FugacityCoeffs=np.zeros((self.NumOfTs,self.NumOfPs,self.NumOfComps))
        #self.FluidState=np.chararray((self.NumOfTs,self.NumOfPs,self.NumOfComps),itemsize=20)
        ##print(np.shape(self.FluidState))

        # set up arrays to store variables for PREOS at a given T, P 
        self.a=np.zeros(self.NumOfComps)
        self.b=np.zeros(self.NumOfComps)
        self.A=np.zeros(self.NumOfComps)
        self.B=np.zeros(self.NumOfComps)
        self.aij=np.zeros((self.NumOfComps,self.NumOfComps))
        self.Aij=np.zeros((self.NumOfComps,self.NumOfComps))
        self.BinaryInteractionParameter=np.zeros((self.NumOfComps,self.NumOfComps))


    def extract_critical_constants(self):

        # critical constants data structure
        # constant = [[Tc_1 Pc_1 w_1 x_1],
        #             [Tc_2 Pc_2 w_2 x_2],
        #             [etc...]] 
        self.criticalconst=np.zeros((self.NumOfComps,4))
        for i in range(self.NumOfComps):
            self.criticalconst[i,0]=self.constants["Tc"+str(i+1)]
            self.criticalconst[i,1]=self.constants["Pc"+str(i+1)]
            self.criticalconst[i,2]=self.constants["w"+str(i+1)]
            self.criticalconst[i,3]=self.constants["x"+str(i+1)]
        print("Critical constants and mol fractions:")
        print(self.criticalconst)

    def calculate_fugacity_coeff_at_state_pt(self,T,P):
        # NOTE Not yet tested for multicomponent

        ### get individual params
        # NOTE P [=] Pa and T [=] K
        # NOTE still need binary interaction parameters for mixtures
        for i in range(self.NumOfComps):

            Tc=self.criticalconst[i,0]
            Pc=self.criticalconst[i,1]
            w= self.criticalconst[i,2] 
            Tr = T/Tc

            kappa=0.37464+1.54226*w-0.26992*(w)**2;                             
            alpha=(1.0+kappa*(1.0-np.sqrt(Tr)))**2;
            self.a[i]=0.45724*alpha*(MOLAR_GAS_CONSTANT*Tc)**2/Pc;                   
            self.b[i]=0.07780*MOLAR_GAS_CONSTANT*Tc/Pc;                              
            self.A[i]=self.a[i]*P/(MOLAR_GAS_CONSTANT*T)**2;             
            self.B[i]=self.b[i]*P/(MOLAR_GAS_CONSTANT*T);
            
            #print(self.a[i])
            #print(self.b[i])
            #print(self.A[i])
            #print(self.B[i])
            #print(self.P[P])
            #print(self.T[T])
        ###


        ### compute mixing
        for i in range(self.NumOfComps): 
            for j in range(self.NumOfComps): 
                self.aij[i,j]=(1.0-self.BinaryInteractionParameter[i,j])*np.sqrt(self.a[i]*self.a[j]); 
                self.Aij[i,j]=(1.0-self.BinaryInteractionParameter[i,j])*np.sqrt(self.A[i]*self.A[j]); 
        Amix=0.0; 
        Bmix=0.0; 
        for i in range(self.NumOfComps): 
            Bmix+=self.criticalconst[i,3]*self.b[i];                
            for j in range(self.NumOfComps): 
                Amix+=self.criticalconst[i,3]*self.criticalconst[j,3]*self.aij[i,j];
        Amix*=P/(MOLAR_GAS_CONSTANT*T)**2;                     
        Bmix*=P/(MOLAR_GAS_CONSTANT*T);                        
        ###


        ### Cubic equation (Note order reversed from RASPA)
        coefficients=np.zeros(4)
        coefficients[0]=1.0;                                                    
        coefficients[1]=Bmix-1.0;                                               
        coefficients[2]=Amix-3.0*Bmix**2-2.0*Bmix;                            
        coefficients[3]=-(Amix*Bmix-Bmix**2-Bmix**3); 
        compressibility=np.roots(coefficients)
        #print("State condition (T=%.4f,P=%.4f):"%(T, P))
        #print(coefficients)
        #print(compressibility)
        # bool array for which solutions are real
        real_solns = np.isreal(compressibility)
        ###


        ### for each compressibility solution, calculate fugacity coeff
        fugacitycoeff=np.zeros((self.NumOfComps,3))
        for i in range(self.NumOfComps):
            for s in range(0,3):
                temp=0
                for k in range(self.NumOfComps):
                    # NOTE this is 2*mol_frac*Aij
                    temp+=2.0*self.criticalconst[k,3]*self.Aij[i,k]
                    #print("PREOS:\n a=%e \n b=%e \n A=%e \n B=%e \n aij=%e \n Aij=%e \n Amix=%e \n Bmix= %e \n temp=%e \n"%\
                    #      (self.a[i],self.b[i],self.A[i],self.B[i],self.aij[i,k],self.Aij[i,k],Amix,Bmix,temp))

                fugacitycoeff[i,s]=np.exp((self.B[i]/Bmix)*(compressibility[s]-1.0)-np.log(compressibility[s]-Bmix)\
                    -(Amix/(2.0*np.sqrt(2.0)*Bmix))*(temp/Amix-self.B[i]/Bmix)*\
                    np.log((compressibility[s]+(1.0+np.sqrt(2.0))*Bmix)/(compressibility[s]+(1.0-np.sqrt(2))*Bmix)));


            #print(fugacitycoeff)
        ###


        num_solns =np.sum(real_solns)
        #print(real_solns)

        # get fugacity coefficient of each component
        final_fugacity_coeff=np.zeros(self.NumOfComps)
        final_fluid_state="Unknown"
        for i in range(self.NumOfComps):
            if(num_solns==1):
                # single soln is the real compressibility root
                ind=np.where(real_solns==True)
                final_fugacity_coeff[i]=fugacitycoeff[i,ind]
                if(T>self.criticalconst[i,0] and P>self.criticalconst[i,1]):
                    final_fluid_state="SUPER_CRITICAL_FLUID"
                elif(T<self.criticalconst[i,0] and P<self.criticalconst[i,1]):
                    final_fluid_state="VAPOR"
                elif(T<self.criticalconst[i,0] and P>self.criticalconst[i,1]):
                    final_fluid_state="LIQUID"
            else:
                # if largest and smallest compressibilities are real, both are possible solns
                ind=[0,2]

                if(compressibility[2]>0):
                    if(fugacitycoeff[i,0]<fugacitycoeff[i,2]):
                        final_fugacity_coeff[i]=fugacitycoeff[i,0]
                        final_fluid_state="VAPOR_STABLE"
                    elif(fugacitycoeff[i,0]>fugacitycoeff[i,2]):
                        final_fugacity_coeff[i]=fugacitycoeff[i,2]
                        final_fluid_state="LIQUID_STABLE"
                    else:
                        # will need to set a comparison tolerance if we ever want to achieve this state
                        final_fugacity_coeff[i]=fugacitycoeff[i,0]
                        final_fluid_state="VAPOR_LIQUID_STABLE"

                else:
                    final_fugacity_coeff[i]=fugacitycoeff[i,0]
                    if(T>self.criticalconst[i,0] and P>self.criticalconst[i,1]):
                        final_fluid_state="SUPER_CRITICAL_FLUID"
                    elif(T<self.criticalconst[i,0] and P<self.criticalconst[i,1]):
                        final_fluid_state="VAPOR"
                    elif(T<self.criticalconst[i,0] and P>self.criticalconst[i,1]):
                        final_fluid_state="LIQUID"

        #print("Fugacity coefficient of each component:\n"+str(final_fugacity_coeff))
        #print("Final fluid state:\n"+str(final_fluid_state))
        return final_fugacity_coeff, final_fluid_state    





class DOS(object):

    def __init__(self, flist, constants):


        ##### Read/Setup constants and simulation details #####
        
        # read
        self.flist=flist
        self.constants = constants
        print("User-specified system properties:")
        print(self.constants)

        # obtain all relevant T, P points for adsorption based on user input
        self.extract_state_points()

        # obtain rosenbluth weights, if applicable
        self.extract_rosenbluth_weights()

        # obtain ideal chain partition functions, if applicable
        self.extract_Qideal_chain()

        # extract number of beads in flexible chain, if applicable
        self.extract_numbeads()

        # obtain all FHMC acceptance data
        self.extract_acc_data(flist)

        # get maximum moments to test in Taylor expansion
        self.extract_max_moments()
  


        ##### Transition Matrix solutions ######
 
        # Coinstruct C matrix from acceptance data 
        self.construct_C_matrix()

        # construct PI matrix from C matrix
        self.compute_PI_matrix()

        # compute lnQ from PI matrix
        self.compute_lnQ_array()

        # Debug 
        #print("C matrix:")
        #print(C_mat_sum)
        #print("PI matrix:")
        #print(PI_mat)
        #print("ln(Q) array:")
        #print(lnQ_array)

        # write sampling sizes for each macrostate
        self.write_sampling_sizes()


        ##### Temp Reweighting an Isotherm Analysis #####

        # setup the data structures necessary for T reweighting
        self.setup_T_reweight_structs()

        # import the equation of state data
        self.link_DOS_to_EOS()

        # repeat the analysis if we want to test the effect of including higher
        # order moments to the Taylor expansion
        print(self.all_max_moments)
        for m in self.all_max_moments:
            self.T_reweight_lnQ_array(m)
            self.write_lnQ_array()

            # compute isotherm
            self.get_reservoir_thermo()
            self.compute_isotherm()
            self.print_isotherm()


    def extract_max_moments(self):
        # User specifies different max Taylor expansion terms to try 
        # for the T extrapolation            

        if("all_max_moments" in self.constants.keys()):
            if(type(self.constants['all_max_moments']) is list):
                self.all_max_moments=[int(val) for val in self.constants['all_max_moments']]
            else:
                self.all_max_moments=[int(self.constants['all_max_moments'])]
        else:
            self.all_max_moments=[3]

    def extract_rosenbluth_weights(self):
        # User specifies the ideal gas rosenbluth weight of the molecule
        # at a series of temperatures

        if("IGRW_file" in self.constants.keys()):
            data=np.loadtxt(self.constants['IGRW_file'])
            self.rosen={}
            for T in range(np.shape(data)[0]):
                self.rosen["%.1f"%data[T,0]]=data[T,1]
        else:
            self.rosen=None

    def extract_Qideal_chain(self):
        # User specifies the ideal chain partition function of the molecule
        # at a series of temperatures
        if("Qidchain_file" in self.constants.keys()):
            try:
                data=np.loadtxt(self.constants['Qidchain_file'])
            except:
                raise ValueError("Qideal_chain file %s requested but not found"%\
                                 self.constants['Qidchain_file'])
            self.Qidchain={}
            for T in range(np.shape(data)[0]):
                self.Qidchain["%.1f"%data[T,0]]=data[T,1]
        else:
            self.Qidchain=None


    def extract_numbeads(self):
        if("numbeads" in self.constants.keys()):
            self.numbeads=self.constants['numbeads']
        else:
            self.numbeads=1

    def write_sampling_sizes(self):
        # write to file how many samples we have for each value of collective variable

        f=open("sampling_sizes.txt","w")
        for N in range(self.Nmax):
            f.write("%d %d\n"%(N,np.shape(self.acc_data[N])[0]))
        f.close()
         
    def extract_state_points(self):
        """
        From user input, determine all T, P state points that we would like
        to extrapolate the free energies to
        """

        # Different pressures to obtain fugacity coeffs
        if('Pstart' in self.constants.keys() and\
           'Pend' in self.constants.keys() and\
           'Pdel' in self.constants.keys()):
            self.allP = np.arange(self.constants['Pstart'],self.constants['Pend'],self.constants['Pdel'])
        elif('pressure' in self.constants.keys()):
            self.allP = self.constants['pressure']
        else:
            raise ValueError("No pressure provided in your constants dictionary")
        if(self.allP[0]==0.0):
            self.allP[0]==1e-20
        self.NumOfPs = len(self.allP)

        # Different temperatures to obtain fugacity coeffs
        if('Tstart' in self.constants.keys() and\
           'Tend' in self.constants.keys() and\
           'Tdel' in self.constants.keys()):
            self.T_extrap = list(np.arange(self.constants['Tstart'],self.constants['Tend'],self.constants['Tdel']))
            self.constants['T_extrap']=self.T_extrap
        elif('T_extrap' in self.constants.keys()):
            self.T_extrap = self.constants['T_extrap']
        elif('temperature' in self.constants.keys()):
            # No extrapolation, just the temperature of the simulation
            # self.allT = self.constants['temperature']
            pass
        else:
            raise ValueError("No temperature provided in your constants dictionary")
        self.NumOfTs = len(self.T_extrap)


    def extract_acc_data(self, flist):   
        """
        From the supplied file list, extract acceptance data for up to Nmax
        """

        self.acc_data = [None for i in range(len(flist))]

        if('Nmax' in self.constants.keys()):
            self.Nmax=int(self.constants['Nmax'])
        else:
            self.Nmax=len(self.acc_data)


        for fname in flist:
            N = int(re.findall(r'N0-(.*?).all_energies',fname)[0])
            if(N < self.Nmax):
                print("Loading file: %s"%fname)
                # column1 = widom insertion acc
                # column2 = widom deletion acc
                # column3 = potential energy
                self.acc_data[N] = np.loadtxt(fname)

        self.acc_data=self.acc_data[:self.Nmax]


    def construct_C_matrix(self):
        """
        Construct the C matrix from Transition Matrix approach to FHMC
        using the acceptance data
        """

        self.C_mat=[[None for i in range(self.Nmax)] for i in range(self.Nmax)]
        self.C_mat_mod=[[None for i in range(self.Nmax)] for i in range(self.Nmax)]
        self.C_mat_sum=np.zeros((self.Nmax,self.Nmax))

        for N in range(self.Nmax):
            if(N==0):
                col1=1
                col2=0
            elif(N==self.Nmax-1):
                col1=self.Nmax-1
                col2=N-1
            else:
                col1=N+1
                col2=N-1

            #print(N, col1, col2)

            # forward acc
            self.C_mat[N][col1]       = self.acc_data[N][:,0]
            self.C_mat_mod[N][col1]   = self.acc_data[N][:,0]
            self.C_mat[N][col1][self.C_mat[N][col1] > 1.0] = 1.0
            self.C_mat_sum[N,col1]    = np.sum(self.C_mat[N][col1])

            # rev acc
            self.C_mat[N][col2]       = self.acc_data[N][:,1]
            self.C_mat_mod[N][col2]   = self.acc_data[N][:,1]
            self.C_mat_mod[N][col2][self.C_mat_mod[N][col2] > 1.0] = 1.0
            self.C_mat_sum[N,col2]    = np.sum(self.C_mat_mod[N][col2])

            # forward reject
            self.C_mat[N][N]=np.append(self.C_mat[N][N],1-self.acc_data[N][:,0])
            self.C_mat_sum[N,N]+=np.sum(1-self.C_mat_mod[N][col1])
        
            # reverse reject
            self.C_mat[N][N]=np.append(self.C_mat[N][N],1-self.acc_data[N][:,1])
            self.C_mat_sum[N,N]+=np.sum(1-self.C_mat_mod[N][col2])



    def reweight_C_matrix(self):
        # good to have the option to reweight the C matrix if we want acc
        # at a different ref. chem pot
        pass

    def compute_PI_matrix(self):
        """
        Construct the PI matrix (macrostate transition probability) from the 
        C matrix
        """

        self.PI_mat=np.zeros((self.Nmax,self.Nmax))

        for N in range(self.Nmax):
            if(N==0):
                self.PI_mat[N,N+1]=self.C_mat_sum[N,N+1]/\
                    (self.C_mat_sum[N,N]  +self.C_mat_sum[N,N+1])
            elif(N==self.Nmax-1):
                self.PI_mat[N,N-1]=self.C_mat_sum[N,N-1]/\
                    (self.C_mat_sum[N,N-1]+self.C_mat_sum[N,N])
            else:
                self.PI_mat[N,N+1]=self.C_mat_sum[N,N+1]/\
                    (self.C_mat_sum[N,N-1]+self.C_mat_sum[N,N]+self.C_mat_sum[N,N+1])
                self.PI_mat[N,N-1]=self.C_mat_sum[N,N-1]/\
                    (self.C_mat_sum[N,N-1]+self.C_mat_sum[N,N]+self.C_mat_sum[N,N+1])


    def compute_lnQ_array(self):
        """
        Compute the canonical partition function at each macrostate based on the
        macrostate transition probabilities
        """

        # store all T in case we reweight to get lnQ at different T
        self.allT         =[self.constants['temperature']]
        self.allB         =[1/self.allT[0]]
        self.allB_PartPerJ=[1/(constants['temperature']*BOLTZMANN_CONSTANT)]

        # each column is lnQ for a different temperature
        self.lnQ_array=np.zeros((self.Nmax,1))
        self.lnQ_array[0]=0
        self.lnQ_array_ref0=np.zeros((self.Nmax,1))
        self.lnQ_array_ref0[0]=0

        # Dimensionless thermal de broglie wavelength
        self.constants['RTDBW_prefactor']=np.power(
            PLANCK_CONSTANT/(np.sqrt(2*np.pi*self.constants['mass1']*BOLTZMANN_CONSTANT/(1000*AVOGADRO_CONSTANT)))/ANGSTROM,
            3)
        self.constants['RTDBW']=self.constants['RTDBW_prefactor']*np.power(1/np.sqrt(self.constants['temperature']),3)
        self.allRTDBW=[self.constants['RTDBW']]

        #print(self.constants['RTDBW_ref'])
        #print(self.constants['RTDBW'])
        #print(np.log(self.constants['RTDBW']**-1))

        for N in range(self.Nmax-1):
            # Thus lnQ_array represents the TOTAL partition function 
            self.lnQ_array[N+1,0]=\
                np.log(self.constants['RTDBW']**-1)+\
                self.lnQ_array[N,0]+\
                np.log(self.PI_mat[N,N+1]/self.PI_mat[N+1,N])+\
                0#np.log(self.constants['IGRW']**-1)+\

            # THus lnQ_array_ref0 represents the CONGIGURATIONAL factor of the partition function
            self.lnQ_array_ref0[N+1,0]=+self.lnQ_array_ref0[N,0]+\
                np.log(self.PI_mat[N,N+1]/self.PI_mat[N+1,N])

    def find_all_rosen(self):
        """
        Align the values of IGRW(T) with the values of T in the T extrapolation
        """
        self.allRosen=[]
        if(self.rosen is not None):
            for T in self.allT:
                key="%.1f"%T
                try:
                    self.allRosen.append(self.rosen[key])
                except:
                    raise ValueError("IGRW for T=%.1f not provided in %s"%(T,self.constants["IGRW_file"]))
        else:
            for T in self.allT:
                self.allRosen.append(1.0)

    def find_all_Qideal_chain(self):
        """
        Align the values of Q_idchain(T) with the values of T in the T extrapolation
        """

        self.allQidchain=[]
        if(self.Qidchain is not None):
            for T in self.allT:
                key="%.1f"%T
                try:
                    self.allQidchain.append(self.Qidchain[key])
                except:
                    raise ValueError("Qidchain for T=%.1f not provided in %s"%(T,self.constants["Qidchain_file"]))
        else:
            for T in self.allT:
                self.allQidchain.append(1.0)

    def setup_T_reweight_structs(self):
        """
        Setup the data structures needed to do T extrapolation
        """
    
        for numT in range(len(self.T_extrap)):
            self.lnQ_array=np.hstack((self.lnQ_array,np.zeros((self.Nmax,1))))     
            self.lnQ_array_ref0=np.hstack((self.lnQ_array_ref0,np.zeros((self.Nmax,1))))     
        #print(self.lnQ_array)
        #print(self.lnQ_array_ref0)
        
        # New temperatures and reference chem pots (Note first entry always at simulation conditions
        self.allT+=self.T_extrap
        # Beta units 1/K
        self.allB+=[1/(thisT) for thisT in self.T_extrap]
        # Beta units 1/(J/part)
        self.allB_PartPerJ+=[1/(thisT*BOLTZMANN_CONSTANT) for thisT in self.T_extrap]
        # lambda^3
        self.allRTDBW+=[self.constants['RTDBW_prefactor']*np.power(1/np.sqrt(thisT),3) for thisT in self.T_extrap]
        # extract rosenbluth IG weight if specified for each T
        self.find_all_rosen()
        # extract ideal chain partition functions if specified for each T
        self.find_all_Qideal_chain()

        print("All RTDBW:")
        print(self.allRTDBW)

        print("All rosen")
        print(self.allRosen)

        print("All Qidchain")
        print(self.allQidchain)

    def T_reweight_lnQ_array(self,max_moment=3):
        """
        Use the energy fluctuations at each N macrostate to extrapolate
        lnQ using a Taylor series expansion

        max moment is the mth derivative of lnQ (mth central moment) to include in the Taylor series expansion
        """

        self.max_moment=max_moment

        # Numerical/analytic moments of the E distribution for each N macrostate
        self.E_moment=[[0.0 for j in range(max_moment+1)] for i in range(self.Nmax)]
        self.E_moment_analytic=[[0.0 for j in range(max_moment+1)] for i in range(self.Nmax)]

        # DOS for fictitious system w/u_ref^0 = 0 (aka RTDBW=1) 
        self.lnQ_array_ref0_newT = [None for thisT in self.T_extrap]

        # DOS for actual system with u_ref^0 = 1/beta*ln(RTDBW^3)
        self.lnQ_array_newT = [None for thisT in self.T_extrap]

        # array to write the fluctuation statistics 
        self.taylor_expansion         =np.zeros((len(self.T_extrap),self.Nmax,max_moment+1)) 
        self.taylor_expansion_analytic=np.zeros((len(self.T_extrap),self.Nmax,max_moment+1)) 
        self.fluctuation_terms        =np.zeros((                   self.Nmax,max_moment+1)) 



        # Obtain numerical moments of dist
        for N in range(self.Nmax):
            print("Computing fluctuations: N=%d"%N)

            # Fit dist to gamma function and obtain analytic moments
            # TODO bit hand-wavey on how much to pad 0's on either side for now
            min_y=min(self.acc_data[N][:,2])
            max_y=max(self.acc_data[N][:,2])
            range_y=max_y-min_y # range of values of E
            upshift=min_y-1 # how much we need to shift E by to have min == 1
            y=self.acc_data[N][:,2]-upshift # make min value in y = 1
            x=scipy.arange(range_y+0.1*range_y) # make range of random var from min(y)-1 = 0 to max(y)+1
            moments=[0 for i in range(max_moment+1)]
            pdf_params=[0]
            #pdf_params, pdf_fitted, moments = scipy_fit_dist(x, y, dist_name="gamma")
            #moments[1]+=upshift # recenter the dist

            # note that first deriv of lnQ is average of E
            # PASS FOR NOW: analytic gamma fitted avg
            self.E_moment_analytic[N][1]=moments[1]
            # numerical avg
            self.E_moment[N][1]=np.average(self.acc_data[N][:,2])

            # obtain higher moments
            for m in range(2,max_moment+1):
                # analytic gamma fitted moment
                if(m<5):
                    self.E_moment_analytic[N][m]=moments[m]
                # numerical moment
                self.E_moment[N][m]=moment(self.acc_data[N][:,2],moment=m)

            print("Analytic params:"+str(pdf_params))
            print("Analytic:"+str(self.E_moment_analytic[N]))
            print("Numeric:"+str(self.E_moment[N]))

        # Calculate reweighted DOS at new temperatures
        for i in range(1, len(self.allT)):
            thisT=self.allT[i]
            thisB=self.allB[i]
            thisRTDBW=self.allRTDBW[i]

            for N in range(self.Nmax):
                # 0th order term is lnQ at this T
                ZOT=self.lnQ_array_ref0[N,0]
                ZOT_analytic=ZOT

                # 1st order tailor series term
                FOT         =-1*self.E_moment[N][1]         *(thisB-1/self.constants['temperature'])
                FOT_analytic=-1*self.E_moment_analytic[N][1]*(thisB-1/self.constants['temperature'])

                # extrapolating lnQ at this new T
                self.lnQ_array_ref0[N,i]= ZOT + FOT

                # statistics to write to file
                self.taylor_expansion[i-1,N,0]=ZOT
                self.taylor_expansion[i-1,N,1]=FOT
                self.taylor_expansion_analytic[i-1,N,0]=ZOT_analytic
                self.taylor_expansion_analytic[i-1,N,1]=FOT_analytic

                # writing fluctuation terms to file
                self.fluctuation_terms[N,0]=0.0
                self.fluctuation_terms[N,1]=self.E_moment[N][1]
                
                # obtain higher moments
                for m in range(2,max_moment+1):
                    # NOTE corrected lnQ extrapolation
                    HOT=1/math.factorial(m)*(thisB-1/self.constants['temperature'])**m
                    if(m==2):
                        HOT*=self.E_moment[N][2]
                        HOT_analytic=HOT*self.E_moment[N][2] 
                    elif(m==3):
                        HOT*=-self.E_moment[N][3]
                        HOT_analytic=HOT*-self.E_moment[N][3]
                    elif(m==4):
                        HOT*=(self.E_moment[N][4]-3*self.E_moment[N][2]**2)
                        HOT_analytic=HOT*(self.E_moment[N][4]-3*self.E_moment[N][2]**2)
                    else:
                        raise ValueError("Requested highest order term of m=%d,\
                                          but maximum allowed is m=4"%m)

                    # extrapolate lnQ up to higher derivs
                    #HOT         =1/math.factorial(m)*self.E_moment[N][m]         *(thisB-1/self.constants['temperature'])**m
                    #if(m<5):
                    #    HOT_analytic=1/math.factorial(m)*self.E_moment_analytic[N][m]*(thisB-1/self.constants['temperature'])**m

                    # new muref=0 DOS at new temperature
                    self.lnQ_array_ref0[N,i]+=HOT

                    # statistics to write to file
                    self.taylor_expansion[i-1,N,m]         =HOT
                    self.taylor_expansion_analytic[i-1,N,m]=HOT_analytic
                    self.fluctuation_terms[N,m]=self.E_moment[N][m]

                # compute DOS for when reservoir chem pot ref is ideal gas
                self.lnQ_array[N,i]=N*np.log(thisRTDBW**-1)+self.lnQ_array_ref0[N,i]
                # there is also a constant shift related to the chemical potential
                # of the reservoir if we have a chain molecule
                self.lnQ_array[N,i]+=N*np.log(self.allQidchain[0]/self.allQidchain[i])


            
    def write_lnQ_array(self):

        outname="lnQ_T_%.3f_maxmom_%d.txt"%(self.constants['temperature'],self.max_moment)
        outname1="lnQ_uref0_T_%.3f_maxmom_%d.txt"%(self.constants['temperature'],self.max_moment)
        outfile = open(outname,"w")
        outfile1 = open(outname1,"w")
    
        outfile.write("N sim_T_%.3f "%(self.constants['temperature']))
        outfile1.write("N sim_T_%.3f_uref0 "%(self.constants['temperature']))
        for i in range(1,len(self.allT)):
            outfile.write("extrap_to_%.3f "%(self.allT[i]))
            outfile1.write("extrap_to_%.3f_uref0 "%(self.allT[i]))
        outfile.write("\n")
        outfile1.write("\n")

        for N in range(self.Nmax):
            outfile.write("%d %.5f"%(N,self.lnQ_array[N,0]))
            outfile1.write("%d %.5f"%(N,self.lnQ_array_ref0[N,0]))

            for j in range(1,len(self.allT)):
                #outfile.write(" %.5f %.5f"%(np.log(self.lnQ_array_ref0_newT[i][N]),np.log(self.lnQ_array_ref0_newT[i][N])))
                outfile.write(" %.5f"%(self.lnQ_array[N,j]))
                outfile1.write(" %.5f"%(self.lnQ_array_ref0[N,j]))

            outfile.write("\n")
            outfile1.write("\n")

        outfile.close()
        outfile1.close()
        #print(self.lnQ_array)
        #print(self.lnQ_array_ref0)


    def link_DOS_to_EOS(self):
        self.EOS=EOS(self.constants)
   
 
    def get_reservoir_thermo(self):

        #molecule=preos.Molecule('water', self.constants['Tc'], self.constants['Pc'], self.constants['w']) 

        self.pressures=np.copy(self.allP)
        self.fugacoeff=np.zeros((len(self.pressures),len(self.allT)))
        self.Bu_ref0  =np.zeros((len(self.pressures),len(self.allT)))
        self.Bu       =np.zeros((len(self.pressures),len(self.allT)))
        for i in range(len(self.pressures)):
            if(self.pressures[i]==0.0):
                self.pressures[i]=1e-20

            for j in range(len(self.allT)):
                # preos takes pressure in bar
                #self.fugacoeff[i,j]=preos.preos(molecule, self.allT[j],
                #    self.pressures[i]/10**5, plotcubic=False, printresults=False)['fugacity_coefficient']
                coeff, state=self.EOS.calculate_fugacity_coeff_at_state_pt(self.allT[j],self.pressures[i])
                # NOTE that single component hard-coded for now
                self.fugacoeff[i,j]=coeff[0]

                # this beta has units particles/Joule
                self.Bu_ref0[i,j]=np.log(self.allB_PartPerJ[j]/10**30*self.fugacoeff[i,j]*self.pressures[i])
                self.Bu[i,j]=self.Bu_ref0[i,j]+\
                             np.log(self.allRTDBW[j])+\
                             np.log(self.allRosen[j]**1)
                             #np.log(self.allRosen[j]**-1)

        #print(self.constants['RTDBW'])
        #print(np.log(self.constants['RTDBW']))
        #print(self.constants['beta'])
        # Thermo data to file
        f=open("thermo_data_T_%f.txt"%(self.allT[0]),"w")
        f.write("P phi Bu_ref0 Bu\n")
        for i in range(len(self.pressures)):
            f.write("%f %f %f %f\n"%(self.pressures[i], self.fugacoeff[i,0], self.Bu_ref0[i,0],self.Bu[i,0]))
        f.close()
    
        



    def compute_isotherm(self):
        # TODO need to add a warning when the probability of the final macrostate
        # is greater than some threshold: this means we are missing data from 
        # macrostates not sampled and we need to go back and run more simulations

        gmpy2.set_context(gmpy2.context())

        # quantities to calculate expectation number of particles
        self.Integrands=[mpfr(0.0) for j in range(self.Nmax)]              
        self.PofN=[mpfr(0.0) for j in range(self.Nmax)]                    
        self.NTimesPofN=[mpfr(0.0) for j in range(self.Nmax)] 


        # for each requested chemical potential                                 
        numstatepts=len(self.pressures)
        #self.LnGrandPartFcn=[0.0 for j in range(numstatepts)]  


        # to store for convenient plotting
        #plotpts=np.round(np.geomspace(1,numstatepts,10))
        #plotpts=np.round(np.geomspace(1,numstatepts,numstatepts/10))
        plotpts=np.round(np.linspace(1,len(self.pressures),10))
        self.plot_list_PofN=[]                                                  
        self.plot_list_Bu=[]                                                    
        self.plot_list_P=[]                                                    
        self.plot_list_grand_fcn=[]                                             


        # for each P, T 
        self.final_data=np.zeros((len(self.pressures),len(self.allT)+1)) 
        self.expectationnum=np.zeros((len(self.pressures),len(self.allT)))
        self.LnGrandPartFcn=np.zeros((len(self.pressures),len(self.allT)))
        # for each P, T, N
        self.IntegrandsAll=np.ones((len(self.pressures),len(self.allT),self.Nmax),dtype=object)
        self.PofNAll      =np.ones((len(self.pressures),len(self.allT),self.Nmax),dtype=object)
        self.NTimesPofNAll=np.ones((len(self.pressures),len(self.allT),self.Nmax),dtype=object)
        for T in range(len(self.allT)):
            for P in range(len(self.pressures)):
                for N in range(self.Nmax):                                     
                    self.IntegrandsAll[P,T,N]=mpfr(0.0)
                    self.PofNAll[P,T,N]      =mpfr(0.0)
                    self.NTimesPofNAll[P,T,N]=mpfr(0.0)


        print("Array of T (T[0] == sim temp., T[1..end] == extrap. temp.):")
        print(self.allT)
        for T in range(len(self.allT)):
            print("Computing isotherm at T=%.3f"%self.allT[T])
            print("IGRW for this T=%.3f"%self.allRosen[T])
            # reset quantities for expectation value calc 
            self.Integrands=[mpfr(0.0) for j in range(self.Nmax)]              
            self.PofN=[mpfr(0.0) for j in range(self.Nmax)]                    
            self.NTimesPofN=[mpfr(0.0) for j in range(self.Nmax)] 

            for P in range(len(self.pressures)):

                self.curr_grand_part_fcn=mpfr(0)
        
                for N in range(self.Nmax):                                     
                    #print(self.lnQ_array[j],self.Bu[i],j)   
                    #print(self.lnQ_array[j]+self.Bu[i]*j)   
                    #print(gmpy2.exp(self.lnQ_array[j]+self.Bu[i]*j))
                    # Q*exp(BuN)
                    self.IntegrandsAll[P,T,N]=gmpy2.exp(self.lnQ_array[N,T]+self.Bu[P,T]*N)
                    #print(gmpy2.add(self.curr_part_fcn, self.Integrands[j]))       
                    # grandpartfcn=sum(Q*expBuN)
                    self.curr_grand_part_fcn=gmpy2.add(self.curr_grand_part_fcn, self.IntegrandsAll[P,T,N])
                                                                                    
                for N in range(self.Nmax):                                     
                    self.PofNAll[P,T,N]=self.IntegrandsAll[P,T,N]/self.curr_grand_part_fcn              
                    self.NTimesPofNAll[P,T,N]=self.PofNAll[P,T,N]*N              

        
                # Debug the probability distributions only for the original T 
                #if(i%int(np.floor(numstatepts/numplotpts))==0 and i != 0 and T==0):   
                #if(P in plotpts and T==0):   
                #    #print(int(np.floor(len_chem_pot/num_collect_dist)))            
                #    self.plot_list_PofN.append(np.array(self.PofN,dtype=np.float64))
                #    self.plot_list_Bu.append("%.2f"%(self.Bu[P,T]))                   
                #    self.plot_list_P.append("%.2e"%(self.pressures[P]))                   
                #    #print(self.plot_list_Bu[-1])                                   
                #    #print(self.plot_list_PofN[-1])                                 
                                                                                    
                self.LnGrandPartFcn[P,T]=np.float(gmpy2.log(self.curr_grand_part_fcn))      
                self.expectationnum[P,T]=np.sum(self.NTimesPofNAll[P,T])                          
                                                                                    
                self.final_data[P,0]=self.pressures[P]                         
                self.final_data[P,T+1]=self.expectationnum[P,T]

            if(self.PofNAll[P,T,self.Nmax-1]>1e-5):
                print("Warning: Macrostate N=%d (@ T,P = %.0f, %.1e) has probability=%f, need data for larger macrostate values when consideringt this T,P statepoint "%\
                      (self.Nmax-1, self.allT[T], self.pressures[-1], self.PofNAll[T,-1,self.Nmax-1]))
                print("P(N) for this T,P is:")
                print(self.PofNAll[T,-1])
        
        print(self.final_data)


    def print_isotherm(self):
        #for i in range(np.shape(self.final_data)[0]):
        #    print(self.final_data[i,0], self.final_data[i,1])
        header="sim_T_%.3f "%self.constants['temperature']
        for i in range(1,len(self.allT)):
            header+="extrap_to_T_%.3f "%self.allT[i]
        np.savetxt("isotherm_T_%.3f_maxmom_%d.txt"%(self.constants['temperature'],self.max_moment),
                    self.final_data,delimiter=' ',header=header) 


    def load_DOS_data(self):
        # read in all relevant data so DOS doesn't have to be recomuted each time
        pass

    def output_fluctuation_statistics(self):
        """
        Write the individual Taylor expansion terms to see how much the higher
        order terms contribute to the extrapolation of lnQ
        """

        with open("taylor_expansion.txt", 'w') as f: f.write(json.dumps(self.taylor_expansion, default=lambda x: list(x), indent=4))
        #with open("taylor_expansion.txt", 'w') as f:
        #    for T in range(self.T_extrap):
        #        f.write("Taylor expansion terms for T=%.3f\n"%(self.T_extrap[T]))

        #        for m in range(max_moment+1):
        np.savetxt("fluctuation_terms.txt",self.fluctuation_terms)

        # make a plot for each expansion term at each each extrap temp
        # row = expansion term
        # col = extrap temp
        nr=self.max_moment+1
        nc=len(self.allT)-1

        curr_fig=1                                                              
        plt.figure(curr_fig,figsize=(3.3*nc,2.4*nr))                                  
        gs = gridspec.GridSpec(nr,nc)
        #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
        gs.update(left=0.15, right=0.93, top=0.94, bottom=0.17)

        for m in range(self.max_moment+1):
            for T in range(1, len(self.allT)):
                col=T-1
                row=m
                ind=m*nc+(T-1)
                ax=plt.subplot(gs[ind])
                if(row >= 3):
                    # 3rd order moments usually get very small so better to look at log scale
                    #ax.plot([i for i in range(self.Nmax)],np.log(self.taylor_expansion[T-1,:,m]))
                    ax.plot([i for i in range(self.Nmax)],self.taylor_expansion[T-1,:,m])
                    ax.plot([i for i in range(self.Nmax)],self.taylor_expansion_analytic[T-1,:,m])
                else:
                    ax.plot([i for i in range(self.Nmax)],self.taylor_expansion[T-1,:,m])
                    ax.plot([i for i in range(self.Nmax)],self.taylor_expansion_analytic[T-1,:,m])

                if(col==0):
                    if(row==0):
                        ax.set_ylabel(r"$lnQ$")
                    elif(row>=1):
                        ax.set_ylabel(r"$\frac{\partial^%d lnQ}{\partial \beta} (\beta - \beta_{sim})^%d $"%(row,row))
                if(row == 0):
                    ax.set_title("T=%d K"% self.allT[T])
                if(row == nr-1):
                    ax.set_xlabel(r"N (molecules/simbox)")


        plt.savefig("vis/fluctuation_terms.pdf",transparent=True)


            

                     

    def output_N_PT_diagram(self):
        
        curr_fig=1                                                              
        plt.figure(curr_fig,figsize=(3.3,2.4))                                  
        gs = gridspec.GridSpec(1,1)                                             
        #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
        gs.update(left=0.15, right=0.93, top=0.94, bottom=0.17)       
        ax1=plt.subplot(gs[0])                                                  

        order=np.argsort(self.allT)

        # apply a conversion to change N per UC to something else if desired
        if('conversion' in self.constants.keys()):
            conversion=self.constants['conversion']
        else:
            conversion=1

        y=self.final_data[::,0]/1e5 # P's
        x=self.allT[1::] # T's
        X, Y = np.meshgrid(x,y)
        Z=self.final_data[::,2::]*conversion #n's
        self.CS_N_PT=ax1.contour(X, Y, Z, 20, cmap=plt.cm.viridis)
        #ax1.clabel(CS,inline=1,fontsize=8,fmt='%.1f')

        cbar = plt.colorbar(self.CS_N_PT)
        #cbar.ax.set_ylabel(r"Loading [cm$^3$ (STP)/cm$^3$]")
        cbar.ax.set_ylabel(r"Loading ["+self.constants['conversion_string']+"]")

        ax1.set_xlabel('Temperature [K]')
        ax1.set_ylabel('Pressure [bar]')

        np.savetxt("diagram_N_PT.txt",self.final_data[::,2::])
        plt.savefig('./vis/N_PT_contour.pdf',transparent=True)
        plt.close()

    def output_H_NT_diagram(self):

        # interpolate the f and T for each N in the system
        lnf_vs_T_const_N=[] 
        dlnf_vs_dT_const_N=[]
        qst_NT=np.zeros((self.Nmax-1,len(self.allT)-1))

        # NOTE assuming T extrap points are uniformly spaced for now, otherwise won't work
        delT=self.allT[2]-self.allT[1]
    
        for N in range(1, self.Nmax):
            lnf_vs_T=np.zeros((len(self.allT)-1,2))
            lnf_vs_T[:,1]=self.allT[1:]
            # only work with the extrapolation temperatures
            out_of_bounds=-1
            breakT=False
            for T in range(1, len(self.allT)):
                #print(self.allT[T])
                for ind in range(len(self.pressures)-1):
                    #print("Upper/lower bounds:")
                    #print(self.final_data[ind,T+1])
                    #print(self.final_data[ind+1,T+1])
                    if(N<self.final_data[ind+1,T+1] and \
                       N>self.final_data[ind,T+1]):
                        # Now we interpolate the corresponding f for the given N, T
                        thisP=\
                            (self.pressures[ind+1]-self.pressures[ind])/\
                            (self.final_data[ind+1,T+1]-self.final_data[ind,T+1])*\
                            (N-self.final_data[ind,T+1])+self.pressures[ind]
                        thisPhi=\
                            (self.fugacoeff[ind+1,T]-self.fugacoeff[ind,T])/\
                            (self.final_data[ind+1,T+1]-self.final_data[ind,T+1])*\
                            (N-self.final_data[ind,T+1])+self.fugacoeff[ind,T]
                        thisF=thisP*thisPhi
                        lnf_vs_T[T-1,0]=np.log(thisF)
                        #lnf_vs_T[T-1,1]=self.allT[T]
                        break
                    elif(N>self.final_data[-1,T+1]):
                        # This means that at this T and loading, the required reservoir
                        # temperature was higher than we have simulated
                        out_of_bounds=T-1
                        breakT=True
                        #print("out_of_bounds index:")
                        #print(out_of_bounds)
                        break

                if(breakT==True):
                    breakT=False
                    break


            #print("For const. loading N=%d:"%N)
            #print("lnf vs T:")
            #print(lnf_vs_T)
            lnf_vs_T_const_N.append(lnf_vs_T)

            # compute dlnf/dT @ constN
            deriv=np.gradient(lnf_vs_T[:,0],delT,edge_order=2)
            # mask end points or out of bounds indices since derivatives are wrong
            # for some reason np.ma.masked not working with splices...
            deriv[0]=np.ma.masked
            deriv[-1]=np.ma.masked
            if(out_of_bounds!=-1):
                # -1 to throw out the endpoint
                deriv[out_of_bounds-1:]=np.nan
            if(out_of_bounds==0):
                deriv[0:]=np.nan
            if(out_of_bounds<4):
                deriv[out_of_bounds-1:]=np.nan
            # keep track of derivative at each N
            dlnf_vs_dT_const_N.append(deriv)

            #print("dlnf vs dT:")
            #print(deriv)

            qst=-deriv*8.314*np.array(self.allT[1:])**2/1000
            # mask endpoints since they give unreliable derivatives
            qst_NT[N-1,:]=qst
            #print("qst vs T [kJ/mol]:")
            #print(qst)
            #print("")

        #print(lnf_vs_T_const_N)
        #print(qst_NT)


        # Plot data            

        # apply a conversion to change N per UC to something else if desired
        if('conversion' in self.constants.keys()):
            conversion=self.constants['conversion']
        else:
            conversion=1

        curr_fig=1                                                              
        plt.figure(curr_fig,figsize=(3.6,2.4))                                  
        gs = gridspec.GridSpec(1,1)                                             
        #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
        gs.update(left=0.15, right=0.90, top=0.94, bottom=0.17)       
        ax1=plt.subplot(gs[0])                                                  

        #diagramNmax=self.Nmax
        y = np.arange(1,self.Nmax)*conversion # N's
        #y = y[0::diagramNmax]
        x = np.array(self.allT[1:]) # T's
        X, Y = np.meshgrid(x,y)

        #Z=qst_NT[::diagramNmax,::] #n's
        Z=qst_NT

        #print(x)
        #print(y)
        #print(Z)

        #zeros=np.where(Z==0)
        #print("Coords for zeros:")
        #print(zeros)
        #Z[zeros]=np.ma.masked

        # plot data
        CS=ax1.contour(X, Y, Z, 10)
        CS_clear=ax1.contourf(X, Y, Z, 10,alpha=1)
        #ax1.clabel(CS,inline=1,fontsize=8,fmt='%.1f')

        # make color bar
        cbar = plt.colorbar(CS_clear)
        cbar.ax.set_ylabel(r"q$_{st}$ [kJ/mol]")
        # Add the contour line levels to the colorbar
        #cbar.add_lines(CS)

        ax1.set_xlabel('Temperature [K]')
        #ax1.set_ylabel(r"Loading [cm$^3$ (STP)/cm$^3$]")
        ax1.set_ylabel(r"Loading ["+self.constants['conversion_string']+"]")
        #ax1.set_ylabel(r"Loading [molec./sim. box]")
        #ax1.set_ylim((0,10))

        np.savetxt("diagram_qst_NT.txt",Z)
        plt.savefig('./vis/H_NT_contour.pdf',transparent=True)
        plt.close()


        # REPRESENTATION #2
        # regular plot with qst on y-axis, t on x-axis, lines repersent const N 
        curr_fig=1                                                              
        plt.figure(curr_fig,figsize=(3.3,2.4))                                  
        gs = gridspec.GridSpec(1,1)                                             
        #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
        gs.update(left=0.15, right=0.90, top=0.94, bottom=0.17)       
        ax1=plt.subplot(gs[0])                                                  

        numContours=10
        numN=np.shape(qst_NT)[0]
        numT=np.shape(qst_NT)[1]
        spacing=int(np.floor(numN/numContours))
        all_xs=[list(x) for i in range(numN)]
        all_ys=[list(qst_NT[i,:]) for i in range(numN)]
        for i in range(spacing):
            ax1.plot(all_xs[i*numContours],all_ys[i*numContours])
        plt.savefig('./vis/H_NT_contour_v2.pdf',transparent=True)
        plt.close()

        # REPRESENTATION #3
        # regular plot with qst on y-axis, N on x-axis, lines repersent const T 
        curr_fig=1                                                              
        plt.figure(curr_fig,figsize=(3.3,2.4))                                  
        gs = gridspec.GridSpec(1,1)                                             
        #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
        gs.update(left=0.15, right=0.90, top=0.94, bottom=0.17)       
        ax1=plt.subplot(gs[0])                                                  

        this_x=np.array([N for N in range(self.Nmax-1)])
        this_y=(np.array([self.E_moment[N+1][1]-self.E_moment[N][1]\
            for N in range(self.Nmax-1)])-self.constants['temperature'])*MOLAR_GAS_CONSTANT/1000
        index_of_sim_T=self.allT[1:].index(self.constants['temperature'])
        qst_at_sim_T=qst_NT[:,index_of_sim_T]
        ax1.plot(this_x, this_y, this_x, qst_at_sim_T)
        #print(this_y)
        f=open('./vis/qst_simT.dat','w')
        for x,y in zip(this_x, qst_at_sim_T):
            f.write("%.3f %.3f\n"%(x,y))
        f.close()

        plt.savefig('./vis/H_NT_contour_v3.pdf',transparent=True)
        plt.close()

    def output_DOS_data_vis(self):                                              

        self.collec_var=[N for N in range(self.Nmax)]
        plotTs=[0,4,len(self.allT)-1]
                                           
        curr_fig=0 

        for T in range(len(plotTs)):
            indT = plotTs[T]
                                     
            curr_fig+=1
            plt.figure(curr_fig,figsize=(2.9,8.0))                                  
            gs = gridspec.GridSpec(4,1)                                             
            #gs.update(left=0.15, right=0.95, top=0.95, bottom=0.2)                     
            gs.update(left=0.18, right=0.95, top=0.95, bottom=0.1,hspace=0.5)       
            ax1=plt.subplot(gs[0])                                                  

            # plot lnQ for original T (T of the simulation)
            ax1.plot(self.collec_var,self.lnQ_array[:,indT],color='blue')             
            #ax1.tick_params(axis='y',colors='blue')                                
            ax1.set_xlabel(r"N [molecules]")                                        
            ax1.set_ylabel(r"Ln[Q(NVT)]")                                           
                                                                                    
                                                                                    
            ax2=plt.subplot(gs[1])                                                  
            ax2.plot(self.expectationnum[:,indT],self.LnGrandPartFcn[:,indT])                           
            ax2.set_xlabel(r"<N>($\mu$VT) [molecules]")                             
            ax2.set_ylabel(r"Ln[$\Xi$($\mu$VT)]")                                   
                                                                                    
            
            # Which T,P state points to plot P(N) vs N                                                                        

            #plotPs=[0,7,8,9,10,len(self.pressures)-1]
            plotPs=np.geomspace(1,len(self.pressures)-1,10,dtype=int)
            #plot_max_N=self.expectationnum[len(self.expectationnum[:,0])-1,0]*1.25
            #plot_max_PofN=np.max(self.plot_list_PofN)*1.25
            ax3=plt.subplot(gs[2])                                                  
            #for i in range(len(self.plot_list_PofN)):                               
            for i in range(len(plotPs)):     
                #print(self.collec_var)                                              
                #print(self.plot_list_Bu[i])                                         
                #print(self.plot_list_PofN[i])                                       
                indP=plotPs[i]
                ax3.plot(self.collec_var,self.PofNAll[indP,indT,:],label="P=%.1e [Pa]"%self.pressures[indP])

            #ax2.tick_params(axis='y',colors='red')                                 
            ax3.set_xlabel(r"N [molecules]")                                        
            ax3.set_ylabel(r"P(N)")                                                 
            #ax3.set_xlim((0,plot_max_N))                                                   
            #ax3.set_ylim((0,plot_max_PofN)) 
            ax3.legend(loc='best',prop={'size': 3})
                                                                                    
                                                                                    
                                                                                    
            ax4=plt.subplot(gs[3])                                                  
            ax4.plot(self.pressures/100000,self.expectationnum[:,indT])                    
            ax4.set_xlabel(r"P [Pa$\cdot10^5$]")                                    
            ax4.set_ylabel(r"<N>($\mu$VT) [molecules]")                             
                                                                                    
                                                                                    
            #plt.legend(loc='best',prop={'size':8})                                 
                                                                                    
            #plt.tight_layout()                                                     
            plt.savefig("./vis/DOS_data_T%.3f.pdf"%self.allT[indT],transparent=True)                      
            plt.close()          
            
            

    def output_acc_distribution(self, N):
        """
        Output the forward acceptance distribution of going N->N+1 state
        Output the energy distribution in the N state
        """
        max_N=0
        found=False
        for fname in self.flist:
            thisN = int(re.findall(r'N0-(.*?).all_energies',fname)[0])
            if(thisN==N):
                data=np.loadtxt(fname,dtype=np.float64)
                found=True
                break
            if(thisN>max_N):
                max_N=thisN
        if(N > max_N and found == False):
            raise ValueError("Could not load data for N=%d for visualization (max N in this data is %d)"%(N,max_N))

        # acceptance distribution
        a=data[:,0]
        min_dist=np.min(a[np.nonzero(a)])
        min_dist=-9
        max_dist=np.max(data[:,0])
        max_dist=9

        num_bins=np.logspace(min_dist,max_dist,100)
        norm=False
        hist,bin_edges=np.histogram(data[:,0],bins=num_bins,normed=norm,density=norm)
        bin_centers=(bin_edges[:-1] + bin_edges[1:])/2
        
        curr_fig=0                                                                  
        plt.figure(curr_fig,figsize=(3.3,2.4))                                      
        gs = gridspec.GridSpec(1,1)                                                 
        gs.update(left=0.17, right=0.87, top=0.95, bottom=0.17)                     
        ax1=plt.subplot(gs[0])                                                      
        plot_max=0                                                                  
        
        min_x=1e-9
        max_x=1e9
        ax1.plot(bin_centers,hist)
        ax1.plot((1,1),(0,np.max(hist)))
        ax1.set_xlim((min_x,max_x))
        ax1.set_xscale("log",nonposx='clip')
        #ax1.set_yscale("log",nonposx='clip')
        ax1.set_xlabel(r"$acc$(N$\rightarrow$N+1)")
        ax1.set_ylabel(r"Probability")

        plt.savefig('./vis/acceptance_distribution_N%d_to_N%d.pdf'%(N,N+1))
        plt.close()

        print("Acc histogram:")
        print(bin_edges)
        print(hist)

        # energy distribution
        a=data[:,2]
        min_dist=np.min(a[np.nonzero(a)])
        max_dist=np.max(data[:,2])

        num_bins=np.linspace(min_dist,max_dist,100)
        norm=True
        hist,bin_edges=np.histogram(data[:,2],bins=num_bins,normed=norm,density=norm)
        bin_centers=(bin_edges[:-1] + bin_edges[1:])/2
        
        curr_fig=1                                                                  
        plt.figure(curr_fig,figsize=(3.3,2.4))                                      
        gs = gridspec.GridSpec(1,1)                                                 
        gs.update(left=0.17, right=0.87, top=0.95, bottom=0.17)                     
        ax1=plt.subplot(gs[0])                                                      
        plot_max=0                                                                  
        
        ax1.plot(bin_centers,hist)
        ax1.set_xlabel(r"E(N=%d)"%N)
        ax1.set_ylabel(r"Probability")


        # Now fit the E distribution to the maxwell-boltzmann like dist
        # NOTE important to get the initial guess relatively accurate
        p0=[3.78e-5,1,1e8]
        popt, pcov = curve_fit(maxwell_function, bin_centers-min(bin_centers), hist ,p0=p0)
        # Now fit the E distribution to the generalized extreme value dist
        # NOTE important to get the initial guess relatively accurate
        p0_1=[1,1,0.1]
        popt1, pcov1 = curve_fit(GEV_function, bin_centers-min(bin_centers), hist, 
            bounds=([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf]),p0=p0_1)
        # Now fit the E distribution to the gamma dist
        # NOTE important to get the initial guess relatively accurate
        min_y=min(data[:,2])
        max_y=max(data[:,2])
        y=data[:,2]-min_y+1 # make min value in y = 1
        x=scipy.arange(max_y-min_y+2) # make range of random var from min(y)-1 = 0 to max(y)+1
        print(x)
        x_plot=x+min_y-2 # transform x's back for plotting purposes
        print(x_plot)
        pdf_params, pdf_fitted, moments =scipy_fit_dist(x, y, dist_name="gamma")
        print(len(pdf_params),len(pdf_fitted))
        print(moments)
        print(self.E_moment[N])

        print("Avg energy:")
        print(len(data[:,2]))
        print(np.average(data[0:5000,2]))
        print(np.average(data[-5000:-1,2]))
        print(np.average(data[:,2]))
        

        plt.plot(bin_centers,maxwell_function(bin_centers-min(bin_centers),*popt))
        #plt.plot(bin_centers,GEV_function(bin_centers-min(bin_centers),*popt1))
        plt.plot(x_plot,pdf_fitted)

        plt.savefig('./vis/energy_distribution_N%d.pdf'%N)
        plt.close()
        curr_fig+=1

        plt.figure(curr_fig)
        #xs=np.linspace(-2,4,500)
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.,1.41343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.2,1.41343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.4,1.41343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.6,1.41343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.,0.11343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.,0.051343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.,0.03343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.53373748,1.,0.01343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.63373748,1.,0.11343987]))
        #plt.plot(xs,GEV_function(xs,*[ 1.83373748,1.,0.11343987]))
        #plt.plot(xs,GEV_function(xs,*[ 2.03373748,1.,0.11343987]))
        #plt.plot(xs,GEV_function(xs,*[ 2.23373748,1.,0.11343987]))
        pdf_params, pdf_fitted, dist =scipy_fit_dist(bin_centers-min(bin_centers), data[:,2]-min(data[:,2])+1, dist_name="gamma")
        print(pdf_fitted)
        plt.plot(pdf_fitted)
        plt.savefig("./vis/testGamma.pdf", transparent=True)
        plt.close()

    def output_extrapolation_statistics(self):
        pass
       

if __name__=="__main__":

    # NOTE assumes energy units are same as standard RASPA OUTPUT file [K]

    print(sys.argv)
    fname=sys.argv[1]
    constants_fname=sys.argv[2]

    constants = get_constants(constants_fname)
    flist=get_flist(fname)

    this_EOS=EOS(constants)

    picklefileloc='./Output/System_0/pickled_DOS_object.pkl'
    # first check if we have a saved DOS object
    if(os.path.isfile(picklefileloc)):
        with open(picklefileloc, 'rb') as inputfile:
            this_DOS=pickle.load(inputfile)
    # if not recalculate the DOS object from scratch
    else:
        this_DOS=DOS(flist, constants)
        with open(picklefileloc, 'wb') as outputfile:  
            # dump DOS data
            this_DOS.acc_data=None
            pickle.dump(this_DOS, outputfile, pickle.HIGHEST_PROTOCOL)

    # visualize the important thermodynamic data
    this_DOS.output_N_PT_diagram()
    this_DOS.output_H_NT_diagram()
    this_DOS.output_DOS_data_vis()
    this_DOS.output_acc_distribution(2)
    this_DOS.output_fluctuation_statistics()
    

