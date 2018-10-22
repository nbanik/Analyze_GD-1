import numpy as np
import numpy
import pickle
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from galpy.actionAngle import actionAngleTorus
from galpy.util import bovy_conversion, bovy_coords, save_pickles, bovy_plot
from galpy.potential import MWPotential2014, turn_physical_off, vcirc
import astropy.units as u
from galpy.orbit import Orbit
import pal5_util
import pal5_util_MWfit
import MWPotential2014Likelihood
from MWPotential2014Likelihood import _REFR0, _REFV0


def lbd_to_galcencyl(l,b,d,degree=True):
    xyz=bovy_coords.lbd_to_XYZ(l,b,d,degree=degree)
    Rphiz=bovy_coords.XYZ_to_galcencyl(xyz[:,0],xyz[:,1],xyz[:,2],Xsun=1.,Zsun=0.)
    
    return (Rphiz[:,0],Rphiz[:,1],Rphiz[:,2])
    
def set_prog_potential(chain_ind):
    '''
    potname either MWPotential2014
    or index of chain from the param_file
    '''
    _REFV0= MWPotential2014Likelihood._REFV0
            
    paramf=np.genfromtxt('rperi_grid_select.dat',delimiter=',') 
            
    pind=paramf[chain_ind][0]

    peri=round(paramf[chain_ind][1],2)
    print (peri)

    flat_c=paramf[chain_ind][2]
    vc=paramf[chain_ind][3]
    tvo= vc*_REFV0
    sigv=paramf[chain_ind][4]
    prog=list(paramf[chain_ind][5:]) 

    #indices greater than 14: subtract 1
    if pind > 14 :
        pind -=1

    pind=int(pind)   
    potparams_file=np.loadtxt('pot_params.dat',delimiter=',')
    potparams=list(potparams_file[pind])
    
    pot= MWPotential2014Likelihood.setup_potential(potparams,flat_c,False,False,pal5_util_MWfit._REFR0,tvo)
    
    return (prog,pot,sigv,tvo)
    
   
def make_nondefault_pal5stream(chain_ind,leading=False,timpact=None,b=0.8,hernquist=False,td=5.):
        
        
        orb,pot,sigv,tvo=set_prog_potential(chain_ind)
        
        
        
        try :
            sdf= pal5_util.setup_pal5model_MWfit(ro=_REFR0,vo=tvo,timpact=timpact,pot=pot,orb=orb,hernquist=hernquist,leading=leading,age=td,sigv=sigv)
            
        except numpy.linalg.LinAlgError:
            
            print ("using estimateBIsochrone")
            ts= numpy.linspace(0.,td,1001)/bovy_conversion.time_in_Gyr(_REFV0, _REFR0)
            prog = Orbit(orb,radec=True,ro=_REFR0,vo=tvo,solarmotion=[-11.1,24.,7.25])
            prog.integrate(ts,pot)
            estb= estimateBIsochrone(pot,prog.R(ts,use_physical=False),
                                    prog.z(ts,use_physical=False),
                                    phi=prog.phi(ts,use_physical=False))
        
            if estb[1] < 0.3: isob= 0.3
            elif estb[1] > 1.5: isob= 1.5
            else: isob= estb[1]
            
            print ("b=%f"%isob)
            
            sdf=pal5_util.setup_pal5model_MWfit(ro=_REFR0,vo=tvo,leading=leading,pot=pot,orb=orb,timpact=timpact,b=isob,hernquist=hernquist,age=td,sigv=sigv)
                       
        return sdf
            

def add_MCs(pot=MWPotential2014,Mmin=10**6.,rand_rotate=False,ro=_REFR0):
    
    '''
    Setup Molecular clouds and return their
    M,rs,R,vR,vT,z,vz,phi today
    rand_rotate : if True, add random rotation to phi between [0,2pi]
    '''
    
    hdulist=fits.open('molecular_clouds/J_ApJ_834_57_table1.dat.gz.fits')
    aa=hdulist[1].data
    l=aa['GLON']
    b=aa['GLAT']
    #Near or far distance flag (0=near; 1=far) 
    flag=aa['INF']
    Dnear=aa['Dnear']
    Dfar=aa['Dfar']
    znear=aa['znear']
    zfar=aa['zfar']
    #R_sph_gal=aa['Rgal']
    Rnear=aa['Rnear']
    Rfar=aa['Rfar']
    Mnear=aa['Mnear']
    Mfar=aa['Mfar']
    
    D_all=np.empty(len(l))
    zfile=np.empty(len(l))
    rs_all=np.empty(len(l))
    M_all=np.empty(len(l))


    for ii in range(len(l)):
        if flag[ii] == 0 :
            D_all[ii]=Dnear[ii]
            zfile[ii]=znear[ii]
            #rs_all[ii]=Rnear[ii]*0.001  #convert to kpc
            M_all[ii]=Mnear[ii]
        
        
        else :
            D_all[ii]=Dfar[ii]
            zfile[ii]=zfar[ii]
            #rs_all[ii]=Rfar[ii]*0.001 #convert to kpc
            M_all[ii]=Mfar[ii]
    

    R_all,phi_all,z_all= lbd_to_galcencyl(l,b,D_all*(8.5/8.))
    
    def rs(M):
        return 0.1*(M/10**7.)**0.5

    for ii in range(len(M_all)):
        rs_all[ii]=rs(M_all[ii])


    R_all/=ro
    z_all/=ro
    
    M=[]
    rs=[]
    z=[]
    R=[]
    phi=[]

    for ii in range(len(l)):
        if M_all[ii] >= Mmin :
            M.append(M_all[ii])
            rs.append(rs_all[ii])
            z.append(z_all[ii])
            phi.append(phi_all[ii])
            R.append(R_all[ii])
        
    M=np.array(M)
    rs=np.array(rs)
    z=np.array(z)
    R=np.array(R)
    phi=np.array(phi)
    
    if rand_rotate :
        phi+=2*np.pi*np.random.uniform(low=0.,high=1.,size=len(phi))
    print (phi)
    vT=np.empty(len(M))

    for ii in range(len(M)):
        vT[ii]=vcirc(pot,R[ii])
        
    vR=np.zeros(len(M))
    vz=np.zeros(len(M))
    
    coord=[]
    
    for ii in range(len(M)):
        coord.append([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]])
    
    return (M,rs,coord)
    
    
    
def compute_min_separation(x_mc,y_mc,z_mc,apar,x_stream,y_stream,z_stream):
    '''
    given (x,y,z) of each molecular cloud, compute the minimum separation from the stream chunks
    
    input: x_mc,y_mc,z_mc of the MCs,
    x_stream,y_stream,z_stream as arrays of the stream chunks
    apar of the stream chunks, in order to output the apar at which the minimum separation from the 
    MC occured
    '''
    
    diffx=x_stream - x_mc
    diffy=y_stream - y_mc
    diffz=z_stream - z_mc
    
    diffxyz=np.c_[diffx,diffy,diffz]
    
    norm = np.linalg.norm(diffxyz,axis=1)
    
    #print (diffx)
    
    #print (norm)
    
    #print (len(x_stream), len(norm))
    
    min_ind=np.argmin(norm)
    
    min_sep= norm[min_ind]
    
    apar_min=apar[min_ind]
    
    return (min_sep,apar_min)
    

def aparxv_stream(sdf_smooth,sdf_pepper):
    
    timpact=sdf_pepper._timpact
    
    apar_full=[]
    x_full=[]
    y_full=[]
    z_full=[]
    vx_full=[]
    vy_full=[]
    vz_full=[]
    
    for kk in range(len(timpact)):
    
        apar=[]
        x=[]
        y=[]
        z=[]
        vx=[]
        vy=[]
        vz=[]
    
        a= sdf_pepper._sgapdfs_coordtransform[timpact[kk]]._kick_interpolatedObsTrackXY
        apar_all=sdf_pepper._sgapdfs_coordtransform[timpact[kk]]._kick_interpolatedThetasTrack
      
            
        #at each timpact compute apar_max
        apar_max=sdf_smooth.length(tdisrupt=sdf_pepper._tdisrupt-timpact[kk])*sdf_pepper._length_factor
        #print (apar_max)
        #considering the stream until apar_max, store xyzvxvyvz 
        for ii in range(len(apar_all)):
            if apar_all[ii] <= apar_max :
                apar.append(apar_all[ii])
                x.append(a[:,0][ii])
                y.append(a[:,1][ii])
                z.append(a[:,2][ii])
                vx.append(a[:,3][ii])
                vy.append(a[:,4][ii])
                vz.append(a[:,5][ii])
            
        x_full.append(np.array(x))
        y_full.append(np.array(y))
        z_full.append(np.array(z))
        vx_full.append(np.array(vx))
        vy_full.append(np.array(vy))
        vz_full.append(np.array(vz))
        
        #apar_full.append(np.array(apar)*sdf_pepper._sigMeanSign) # _sigMeanSign = -/+ = trail/lead
        #apar_full.append(np.array(apar)*(-1.))
        apar_full.append(np.array(apar))
        
    return (apar_full,x_full,y_full,z_full,vx_full,vy_full,vz_full)
    

def aparxv_stream_from_multiple_pkl(pot=MWPotential2014,sampling=4096,npart=64,td=5.):
    
    '''
    compute apar,x,v from one or multiple pickle files
    pot_ind: can be either default MWPotential2014 or the chain_ind 
    pkl_fname : without fragment index and extension
    '''
    if pot != MWPotential2014 :
        chain_ind=int(pot)
        prog,_pot,sigv,tvo=set_prog_potential(chain_ind)
        sdf_smooth=make_nondefault_pal5stream(chain_ind,leading=False,timpact=None,b=0.8,hernquist=False,td=td)
        pkl_file='pkl_files/pal5pepper_Plummer_td{}_{}sampling_chainind{}'.format(td,sampling,chain_ind)
        pkl_file=pkl_file + '_{}.pkl'
    else : 
        sdf_smooth= pal5_util.setup_pal5model(pot=pot)
        pkl_file='pkl_files/pal5pepper_{}sampling_MW2014'.format(sampling)
        pkl_file=pkl_file + '_{}.pkl'
        
    apar=[]
    x_stream=[]
    y_stream=[]
    z_stream=[]
    vx_stream=[]
    vy_stream=[]
    vz_stream=[]
    timpact=[]
    
    for ii in range(npart):
        pkl_fname=pkl_file.format(ii)
        with open(pkl_fname,'rb') as savefile:
                    
            print (pkl_fname)

            sdf_pepper= pickle.load(savefile,encoding='latin1')
            ap,x,y,z,vx,vy,vz= aparxv_stream(sdf_smooth,sdf_pepper)
            apar.extend(ap)
            x_stream.extend(x)
            y_stream.extend(y)
            z_stream.extend(z)
            vx_stream.extend(vx)
            vy_stream.extend(vy)
            vz_stream.extend(vz)
            timpact.extend(sdf_pepper._timpact)
           
    return (timpact,apar,x_stream,y_stream,z_stream,vx_stream,vy_stream,vz_stream)
    
    
def compute_impact_parameters_GMC(timp,a,xs,ys,zs,pot=MWPotential2014,npart=64,sampling_low_file='',imp_fac=5.,Mmin=10**6.,rand_rotate=False,td=5.):
    
    '''
    timp : timpacts
    a,xs,ys,zs : list of array, each array decribes the stream at that time, no of arrays = timpacts
    sampling_low : low timpact object on to which the impacts from high timpact case will be set
    imp_fac: X where bmax= X.r_s
    Mmin min mass above which all GMCs will be considered for impact
    rand_rotate : give the GMCs an ol' shaka shaka along phi
    
    '''
    
    if pot != MWPotential2014 :
        chain_ind=int(pot)
        prog,pot,sigv,tvo=set_prog_potential(chain_ind)
        t_age= np.linspace(0.,td,1001)/bovy_conversion.time_in_Gyr(tvo,_REFR0)
        bovy_convert_mass=bovy_conversion.mass_in_msol(tvo,_REFR0)
        
    else : 
        #integrate their orbits 5 Gyr back,
        t_age= np.linspace(0.,5.,1001)/bovy_conversion.time_in_Gyr(_REFV0,_REFR0)
        bovy_convert_mass=bovy_conversion.mass_in_msol(_REFV0,_REFR0)
       
    #load the GMCs
    M,rs,coord=add_MCs(pot=pot,Mmin=Mmin,rand_rotate=rand_rotate)
 
    orbits=[]

    N=len(M)

    for ii in range(N):
    
        orbits.append(Orbit(coord[ii]).flip()) # flip flips the velocities for backwards integration
        orbits[ii].integrate(t_age,pot)
        
    min_sep_matrix=np.empty([N,len(timp)])
    apar_matrix=np.empty([N,len(timp)])

    #compute min_sep of each MC
    for kk in range(len(timp)):
        for jj in range(N) :
            x_mc=orbits[jj].x(timp[kk])
            y_mc=orbits[jj].y(timp[kk])
            z_mc=orbits[jj].z(timp[kk])

            min_sep,apar_min=compute_min_separation(x_mc,y_mc,z_mc,a[kk],xs[kk],ys[kk],zs[kk])

            min_sep_matrix[jj,kk]=min_sep
            apar_matrix[jj,kk]=apar_min
            
            
    impactb=[]
    impact_angle=[]
    vx_mc=[]
    vy_mc=[]
    vz_mc=[]
    tmin=[]
    rs_mc=[]
    M_mc=[]
    impactMC_ind=[]

    #just to get timpacts
    with open(sampling_low_file,'rb') as savefile:
        sdf_pepper_low= pickle.load(savefile,encoding='latin1')
            
    timpact_low=sdf_pepper_low._timpact
    
    c=0
    for ii in range(len(orbits)):

        bmax=imp_fac*rs[ii]/_REFR0

        if min(min_sep_matrix[ii]) <= bmax :
            c+=1

            min_timpact_ind=np.argmin(min_sep_matrix[ii])

            impactMC_ind.append(ii)

            t_high=timp[min_timpact_ind]

            #round t_high to the nearest timpact in the low timpact sampling
            t_low=timpact_low[np.argmin(np.abs(timpact_low-t_high))]
            tmin.append(t_low)

            impactb.append(min_sep_matrix[ii,min_timpact_ind])
            impact_angle.append(apar_matrix[ii,min_timpact_ind]) # _sigMeanSign = -/+ = trail/lead

            rs_mc.append(rs[ii]/_REFR0)
            M_mc.append(M[ii]/bovy_convert_mass)
            #flip velocities
            vx_mc.append(-orbits[ii].vx(t_high))
            vy_mc.append(-orbits[ii].vy(t_high))
            vz_mc.append(-orbits[ii].vz(t_high))

    #combine vx,vy,vz to v
    v_mc=np.c_[vx_mc,vy_mc,vz_mc]
    print ("The stream had %i impacts"%c)
        
    return (impactMC_ind,M_mc,rs_mc,v_mc,impactb,impact_angle,tmin)
    
    
