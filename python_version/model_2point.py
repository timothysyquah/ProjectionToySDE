import numpy as np
from scipy.optimize import fsolve
class sde_int():
    def __init__(self,x0,F,g):
        self.x0 = x0
        self.F = F
        self.g = g
    def initialize_project(self,gc,djc):
        self.gc = gc
        self.djc = djc
    def Euler_Maruyama(self,tmax,delta_t,lambda_t,SCFT = False):
        x = []
        tlist = []
        shape_array = np.shape(self.x0)
        x.append(self.x0)
        tlist.append(0)
        t = 0
        while t < tmax:
            dt_eff = delta_t*lambda_t
            xtemp = x[-1] + self.F(x[-1])*dt_eff 
            if not SCFT:
                noise = np.vstack((np.random.normal(0,1,shape_array[0]),np.random.normal(0,1,shape_array[0]))).T
                xtemp += self.g(x[-1])*np.sqrt(2*dt_eff)*noise
            x.append(xtemp)
            t += delta_t
            tlist.append(t)
        return x,tlist
    def Euler_Maruyama_Projected(self,tmax,delta_t,lambda_t,SCFT = False):
        x = []
        tlist = []
        shape_array = np.shape(self.x0)
        x.append(self.x0)
        tlist.append(0)
        t = 0
        while t < tmax:
            dt_eff = delta_t*lambda_t
            xtemp_i = x[-1] + self.F(x[-1])*dt_eff 
            if not SCFT:
                noise = np.vstack((np.random.normal(0,1,shape_array[0]),np.random.normal(0,1,shape_array[0]))).T
                xtemp_i += self.g(x[-1])*np.sqrt(2*dt_eff)*noise
            error = 1
            tol = 1e-5
            xtemp = xtemp_i.copy()
            maxiter = 5
            itter = 0
            print(np.shape(xtemp))
            print(self.gc(xtemp))
            while error>tol:
                #create a diagonal matrix
                update = (self.gc(xtemp)/
                          
                          
                          
                          self.djc(xtemp))
                temp = xtemp - update

                error = np.sum(np.square(temp.real-xtemp.real))
                itter +=1
                print('update',update)
                print('wvalue',xtemp)
                print('error',error)
                print('gtol',self.gc(xtemp))
                print('djc',self.djc(xtemp))
                xtemp = temp

                if itter > maxiter:
                    break
            break
            x.append(xtemp)
            t += delta_t
            tlist.append(t)
        return x,tlist

def Q(w):
    return np.sum(np.exp(-w))/len(w)
def rhoc(w,phi):
    _Q = Q(w)
    return phi/_Q*np.exp(-w)
def rhog(w,z):
    return z*np.exp(-w)

def calc_density(w,rho,phiA,phiB):
    wplus = w[:,0]
    wminus = w[:,1]
    wA = (wplus+wminus)/2
    wB = (wplus-wminus)/2
    rhoA = rho(wA,phiA)
    rhoB = rho(wB,phiB)
    return np.vstack((rhoA,rhoB)).T

def calcDensity2(w,rho,phiA,phiB):
    rho = calc_density(w,rho,phiA,phiB)
    rhoA = rho[:,0]-phiA
    rhoB = rho[:,1]-phiB
    correlation_array = np.zeros((2,2), dtype=complex)
    correlation_array[0,0] = np.mean(rhoA*rhoA)-np.mean(rhoA)**2
    correlation_array[1,1] = np.mean(rhoB*rhoB)-np.mean(rhoB)**2
    correlation_array[0,1] = np.mean(rhoA*rhoB)-np.mean(rhoA)*np.mean(rhoB)
    correlation_array[1,0] = correlation_array[0,1]
    return correlation_array

#def calc_isothermal_compressible(w,chi,rho0,phiA,phiB):


class model():
    def __init__(self,chi,rho0,ensemble = 'canonical'):
        self.chi = chi
        self.rho0 = rho0
        
        if ensemble == 'canonical':
            self.rho = rhoc
        elif ensemble == 'grand':
            self.rho = rhog
        self.zeta = 1.0
        self.ensemble = ensemble
    def init_canonical(self,phiA):
        self.phiA = phiA
        self.phiB = 1-phiA
    def init_grandcanonical(self,zA,zB):
        # these should be zA = \rho0^{-1} exp(muA) and zB = \rho0^{-1} exp(muB)
        self.phiA = zA
        self.phiB = zB
    def init_weakcompressible(self,zeta):
        self.zeta = zeta
    def dHdw_compressible(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        wA = (wplus+wminus)/2
        wB = (wplus-wminus)/2
        rhoA = self.rho(wA,self.phiA)
        rhoB = self.rho(wB,self.phiB)
        dHdwplus = (rhoA+rhoB)-2/self.chi*wplus
        dHdwminus = (rhoB-rhoA)+2/self.chi*wminus
        return self.rho0*np.vstack((dHdwplus,-dHdwminus)).T
    def dHdw_weakcompressible(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        wA = (wplus+wminus)/2
        wB = (wplus-wminus)/2
        rhoA = self.rho(wA,self.phiA)
        rhoB = self.rho(wB,self.phiB)
        dHdwplus = (rhoA+rhoB)-2/(self.chi+2*self.zeta)*(wplus-self.zeta)
        dHdwminus = (rhoB-rhoA)+2/self.chi*wminus
        return self.rho0*np.vstack((dHdwplus,-dHdwminus)).T
    def dHdw_incompressible(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        wA = (wplus+wminus)/2
        wB = (wplus-wminus)/2
        rhoA = self.rho(wA,self.phiA)
        rhoB = self.rho(wB,self.phiB)
        dHdwplus = (rhoA+rhoB)-1
        dHdwminus = (rhoB-rhoA)+2/self.chi*wminus
        return self.rho0*np.vstack((dHdwplus,-dHdwminus)).T
    def wick(self,w):
        return np.vstack((np.zeros(len(w),dtype = complex)+1j,np.ones(len(w),dtype = complex))).T
    def g(self,w):
        if len(np.shape(w)) == 1:
            w = w.reshape((1,2))
        wplus = w[:,0]
        wminus = w[:,1]
        wA = (wplus+wminus)/2
        wB = (wplus-wminus)/2
        rhoA = self.rho(wA,self.phiA)
        rhoB = self.rho(wB,self.phiB)
        g = self.rho0*(rhoA+rhoB-1)
        return (np.vstack((g,g)).T)
    def dgdw(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        wA = (wplus+wminus)/2
        wB = (wplus-wminus)/2
        if self.ensemble == 'grand':
            rhoA = self.rho(wA,self.phiA)
            rhoB = self.rho(wB,self.phiB)
            dgdwplus = -(rhoA+rhoB)
            dgdwminus = (rhoA-rhoB)
            #there might be a sign here?
            return np.vstack((dgdwplus,dgdwminus)).T
    def getDensity(self,w):
        return self.rho0*calc_density(w,self.rho,self.phiA,self.phiB)
    def getCorrelation(self,w):
        return self.rho0*calcDensity2(w,self.rho,self.phiA,self.phiB)
    def getCorrelation_list(self,wlist):
        return np.array([self.rho0*calcDensity2(w,self.rho,self.phiA,self.phiB) for w in wlist],dtype = complex)    
    
