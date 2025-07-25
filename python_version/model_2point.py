import numpy as np
def LSolver(w0,lag_mult,gl,dgldl,tol = 1e-5,maxiter = 10000,maxupdate = 1000,verbose = False):
    error = 1
    itter=0

    while error > tol:
        update= (gl(w0,lag_mult))/dgldl(w0,lag_mult)
        print(update)
        # actually we can apply this logic locally to each element of the update
        loc  = np.where(np.abs(update)>maxupdate)
        if len(loc)>0:
           update[loc] = update[loc]/np.abs(update[loc])*maxupdate
            # update[loc] = (gl(w0,lag_mult))
        if verbose:
            print(np.max(np.abs(update)))
        lag_mult+= -(update)
        #use Linf nnorm
        error = np.max(np.abs(gl(w0,lag_mult).real))
        error2 = np.max(np.abs(gl(w0,lag_mult).imag))
        if error2>error:
            error = error2
        itter+=1
        if itter == maxiter:
            break
    if verbose:
        return lag_mult,error,itter
    return lag_mult



def getw_rho(wplus,wminus,phiA,phiB,rho):
    wplus = wplus
    wminus = wminus
    wA = (wplus+wminus)/2
    wB = (wplus-wminus)/2
    rhoA = rho(wA,phiA)
    rhoB = rho(wB,phiB)
    return rhoA,rhoB
def remove_k0(field):
    
    # if field is complex
    if np.iscomplexobj(field):

        mean_real = np.mean(field.real)
        mean_imag = np.mean(field.imag)
        field.real = field.real-mean_real
        field.imag = field.imag-mean_imag
    else:
        mean = np.mean(field)
        field = field-mean
    return field



class sde_int():
    def __init__(self,x0,F,g,ensemble = 'canonical'):
        self.x0 = x0
        self.F = F
        self.g = g
        self.ensemble = ensemble
    def initialize_project(self,dgdw,gl,dgldl,tol,maxiter,maxupdate):
        self.dgdw = dgdw
        self.gl = gl
        self.dgldl = dgldl
        self.tol = tol
        self.maxiter = maxiter
        self.maxupdate = maxupdate
    def Euler_Maruyama(self,tmax,delta_t,lambda_t,SCFT = False):
        x = []
        tlist = []
        shape_array = np.shape(self.x0)
        if self.ensemble == 'canonical':
            self.x0[:,0] = remove_k0(self.x0[:,0])

        x.append(self.x0)
        tlist.append(0)
        t = 0
        while t < tmax:
            dt_eff = delta_t*lambda_t
            Force = self.F(x[-1])
            if self.ensemble == 'canonical':
                Force = remove_k0(Force)

            xtemp = x[-1] + Force*dt_eff 
            if not SCFT:
                wplus_noise = np.random.normal(0,1,shape_array[0])
                if self.ensemble=='canonical':
                    wplus_noise = remove_k0(wplus_noise)
                noise = np.vstack((wplus_noise,np.random.normal(0,1,shape_array[0]))).T

                xtemp += self.g(x[-1])*np.sqrt(2*dt_eff)*noise
            if np.any(np.isnan(xtemp)):
                print('nan')
                break
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
                wplus_noise = np.random.normal(0,1,shape_array[0])
                if self.ensemble=='canonical':
                    wplus_noise = remove_k0(wplus_noise)

                noise = np.vstack((wplus_noise,np.random.normal(0,1,shape_array[0]))).T
                xtemp_i += self.g(x[-1])*np.sqrt(2*dt_eff)*noise
            xtemp = xtemp_i.copy()
            lag_mult = np.zeros(len(xtemp),dtype = complex)
            lag_mult = LSolver(xtemp,lag_mult,self.gl,self.dgldl,self.tol,self.maxiter,self.maxupdate)
            # dgdl 

            xtemp[:,0]+=lag_mult*self.dgdw(xtemp)[:,0]
            xtemp[:,1]+=lag_mult*self.dgdw(xtemp)[:,1]

            if np.any(np.isnan(xtemp)):
                print('nan')
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

def calcmean(w,rho,phiA,phiB):
    rho = calc_density(w,rho,phiA,phiB)
    return np.mean(rho,axis = 0)

def calcDensity2(w,rho,phiA,phiB):
    rho = calc_density(w,rho,phiA,phiB)
    rhoA = rho[:,0]
    rhoB = rho[:,1]

    d_rhoA = rhoA-phiA
    d_rhoB = rhoB-phiB
    correlation_array = np.zeros((2,2), dtype=complex)
    correlation_array[0,0] = np.mean(d_rhoA*d_rhoA)-np.mean(d_rhoA)**2
    correlation_array[1,1] = np.mean(d_rhoB*d_rhoB)-np.mean(d_rhoB)**2
    correlation_array[0,1] = np.mean(d_rhoA*d_rhoB)-np.mean(d_rhoA)*np.mean(d_rhoB)
    correlation_array[1,0] = correlation_array[0,1]
    return correlation_array
def calcTotalDensity2(w,rho,phiA,phiB):
    rho = calc_density(w,rho,phiA,phiB)
    rhoA = rho[:,0]
    rhoB = rho[:,1]
    rho_total = rhoA+rhoB
    drho = rho_total-1
    return np.mean(drho*drho)-np.mean(drho)**2


#def calcDensity2(w,rho,phiA,phiB):
#    rho = calc_density(w,rho,phiA,phiB)
#    rhoA = rho[:,0]-phiA
#    rhoB = rho[:,1]-phiB
#    correlation_array = np.zeros((2,2), dtype=complex)
#    correlation_array[0,0] = np.mean(rhoA*rhoA)-np.square(phiA)
#    correlation_array[1,1] = np.mean(rhoB*rhoB)-np.square(phiB)
#    correlation_array[0,1] = np.mean(rhoA*rhoB)-(phiA*phiB)
#    correlation_array[1,0] = correlation_array[0,1]
#    return correlation_array


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
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
        dHdwplus = (rhoA+rhoB)-2/self.chi*wplus
        dHdwminus = (rhoB-rhoA)+2/self.chi*wminus
        return self.rho0*np.vstack((dHdwplus,-dHdwminus)).T
    def dHdw_weakcompressible(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
        dHdwplus = (rhoA+rhoB)-2/(self.chi+2*self.zeta)*(wplus+self.zeta)
        dHdwminus = (rhoB-rhoA)+2/self.chi*wminus
        return self.rho0*np.vstack((dHdwplus,-dHdwminus)).T
    def dHdw_incompressible(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
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
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
        g = self.rho0*(rhoA+rhoB-1)
        return g
    def dgdw(self,w):
        wplus = w[:,0]
        wminus = w[:,1]
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
        if self.ensemble == 'grand':
            dgdwplus = -(rhoA+rhoB)
            dgdwminus = -(rhoB-rhoA)
            return self.rho0*np.vstack((dgdwplus,dgdwminus)).T
    def getDensity(self,w):
        return self.rho0*calc_density(w,self.rho,self.phiA,self.phiB)
    def getCorrelation(self,w):
        return calcDensity2(w,self.rho,self.phiA,self.phiB)
    def getCorrelation_list(self,wlist):
        return np.array([calcDensity2(w,self.rho,self.phiA,self.phiB) for w in wlist],dtype = complex)    
    def getCorrelationTotal(self,w):
        return calcTotalDensity2(w,self.rho,self.phiA,self.phiB)
    def getCorrelationTotal_list(self,wlist):
        return np.array([calcTotalDensity2(w,self.rho,self.phiA,self.phiB) for w in wlist],dtype = complex)
    def dgldl(self,w,l):
        wplus = w[:,0]
        wminus = w[:,1]
        rhoA,rhoB = getw_rho(wplus,wminus,self.phiA,self.phiB,self.rho)
        rho_plus = rhoA+rhoB
        rho_minus = rhoB-rhoA
        wnew = w.copy()
        wnew[:,0] += l*self.dgdw(w)[:,0]
        wnew[:,1] += l*self.dgdw(w)[:,1]

        if self.ensemble == 'grand':
            wplus_c = wnew[:,0]
            wminus_c = wnew[:,1]
            rhoA_adjust,rhoB_adjust = getw_rho(wplus_c,wminus_c,self.phiA,self.phiB,self.rho)
            dgdl = rhoA_adjust*(rho_plus-rho_minus)+rhoB_adjust*(rho_plus+rho_minus)
            dgdl *= self.rho0*0.25
            return dgdl
    def gl(self,w,l):
        wnew = w.copy()
        wnew[:,0] += l*self.dgdw(w)[:,0]
        wnew[:,1] += l*self.dgdw(w)[:,1]
        return self.rho0*self.g(wnew)
