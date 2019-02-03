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
import gd1_util
import MWPotential2014Likelihood
from MWPotential2014Likelihood import _REFR0, _REFV0

ro=8.
vo=220.


def lbd_to_galcencyl(l,b,d,degree=True):   
    xyz=bovy_coords.lbd_to_XYZ(l,b,d,degree=degree) ## These are in physical units, DUMB DUMB DUMB!!!
    Rphiz=bovy_coords.XYZ_to_galcencyl(xyz[:,0]/ro,xyz[:,1]/ro,xyz[:,2]/ro,Xsun=1.,Zsun=0.) 
    return (Rphiz[:,0]*ro,Rphiz[:,1],Rphiz[:,2]*ro)
    
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
            

def add_MCs(Mmin=10**5.,fill_holes=True,rand_rotate=False,Rmin=0.,Rmax=1000000000.,ro=_REFR0,non_zero_z=1):
    # M  rs  R  vR   vT   z   vz  phi
    
    if fill_holes :
        dat=np.loadtxt('GMC_catalog_uniform_Mmin105_Rmax40kpc.dat')
        
    else :
        dat=np.loadtxt('GMC_raw_catalog_all.dat')
       
    M_all=dat[:,0]
    rs_all=dat[:,1]
    R=dat[:,2]
    vR=dat[:,3]
    vT=dat[:,4]
    z=dat[:,5]
    vz=dat[:,6]   
    phi=dat[:,7]
    
    if non_zero_z == 0 :
        print ('Setting z = 0 for all the GMCs')
        z*=0.
     
    M=[]
    rs=[]
    coord=[]
    
    if rand_rotate :
        phi+=2*np.pi*np.random.uniform(low=0.,high=1.,size=len(phi))
       
    for ii in range(len(M_all)):
        if M_all[ii] > Mmin and Rmin <= R[ii]*ro <= Rmax :
            M.append(M_all[ii])
            rs.append(rs_all[ii])
            coord.append([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]])
            
    #def rs(M):
    #    return 0.1*(M/10**7.)**0.5

    #print ("WARNING: Using DM subhalo r_s(M) relation")
    
    #def rs(M):
    #    return 1.62*(M/10**8.)**0.5
    
    
    
    #for ii in range(len(M_all)):
    #    rs_all[ii]=rs(M_all[ii])
        
    M=np.array(M)
    rs=np.array(rs)
        
    
    #print (phi)
       
    #NOTE: coord has all quantities in galpy units, M and rs are still in physical units
    print ("Total %i GMCs, selected %i GMCs"%(len(M_all),len(M)))
              
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
    

## works only for GD-1 stream, will combine this later to the general function for any stream
def aparxv_GD1_stream_from_multiple_pkl(pot=MWPotential2014,sampling=7400,npart=116,td=9.0):
    
    '''
    compute apar,x,v from one or multiple pickle files
    pkl_fname : without fragment index and extension
    '''
     
    sdf_smooth= gd1_util.setup_gd1model(pot=pot)
    pkl_file='pkl_files/gd1pepper_Plummer_td{}_{}sampling_MW2014'.format(td,sampling)
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
    
    
def compute_impact_parameters_GMC(timp,a,xs,ys,zs,pot=MWPotential2014,fill_holes=True,npart=64,
                                  sampling_low_file='',imp_fac=5.,Mmin=10**6.,Rmax=1000000000.,Rmin=0.,
                                  rand_rotate=False,td=5.,non_zero_z=1):
    
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
        print ('MWPotential2014')
        #integrate their orbits td Gyr back,
        t_age= np.linspace(0.,td,1001)/bovy_conversion.time_in_Gyr(_REFV0,_REFR0)
        bovy_convert_mass=bovy_conversion.mass_in_msol(_REFV0,_REFR0)
       
    #load the GMCs
    #M,rs,coord=add_MCs(pot=pot,Mmin=Mmin,rand_rotate=rand_rotate,Rmax=Rmax,Rmin=Rmin)
    M,rs,coord=add_MCs(Mmin=Mmin,Rmin=Rmin,Rmax=Rmax,fill_holes=fill_holes,rand_rotate=rand_rotate,non_zero_z=non_zero_z)
 
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
    
    M_mc=np.array(M_mc)
    rs_mc=np.array(rs_mc)
    v_mc=np.array(v_mc)
    impactb=np.array(impactb)
    impact_angle=np.array(impact_angle)
    tmin=np.array(tmin)
        
    return (impactMC_ind,M_mc,rs_mc,v_mc,impactb,impact_angle,tmin)
    
    
    
def set_GMC_impact(stream='GD1',Rmin=0.,Rmax=10000.,Mmin=10**5.):
    '''
    This function takes in the fiducial
    Pal 5 or GD1 stream and outputs the impact parameters for the GMCs
    '''
    
    
    if stream == 'GD1' :
        td1=9.0
        #dir='/ufrc/tan/nilanjan1/Analyze_GD-1/'
        sample_low='pkl_files/gd1pepper_Plummer_td9.0_128sampling_MW2014.pkl'
        timpact,_apar,_xstream,_ystream,_zstream,_vxstream,_vystream,_vzstream=aparxv_GD1_stream_from_multiple_pkl(pot=MWPotential2014,sampling=7400,
                                                                      npart=116,td=9.0)
        
    elif stream == 'Pal5' :
        td1=5.0
        dir='/ufrc/tan/nilanjan1/streamgap-pepper/'
        sample_low=dir + 'pkl_files/pal5pepper_Plummer_td5.0_128sampling_MWPotential.pkl'
        timpact,_apar,_xstream,_ystream,_zstream,_vxstream,_vystream,_vzstream=aparxv_stream_from_multiple_pkl(pot=MWPotential2014,sampling=4096,
                                                                      npart=64,td=5.0)
                                                                      
    #delete chunk with apar = 0.0
    apar=[]
    x_stream=[]
    y_stream=[]
    z_stream=[]
    vx_stream=[]
    vy_stream=[]
    vz_stream=[]
    
    for ii in range(len(_apar)):
        apar_list= list(_apar[ii])
        xstream_list= list(_xstream[ii])
        ystream_list= list(_ystream[ii])
        zstream_list= list(_zstream[ii])
        vxstream_list= list(_vxstream[ii])
        vystream_list= list(_vystream[ii])
        vzstream_list=list(_vzstream[ii])
        #print (apar_list)
        del apar_list[0]
        del xstream_list[0]
        del ystream_list[0]
        del zstream_list[0]
        del vxstream_list[0]
        del vystream_list[0]
        del vzstream_list[0]
        
        apar.append(np.array(apar_list))
        x_stream.append(np.array(xstream_list))
        y_stream.append(np.array(ystream_list))
        z_stream.append(np.array(zstream_list))
        vx_stream.append(np.array(vxstream_list))
        vy_stream.append(np.array(vystream_list))
        vz_stream.append(np.array(vzstream_list))
        
    
    impactMC_ind,M_mc,rs_mc,v_mc,impactb,impact_angle,tmin=compute_impact_parameters_GMC(timpact,apar,x_stream,y_stream,z_stream,
                                                       pot=MWPotential2014,sampling_low_file=sample_low,Mmin=Mmin,td=td1,Rmax=Rmax,
                                                       Rmin=Rmin,rand_rotate=True,fill_holes=True)
                                                    
               
    return(M_mc,rs_mc,v_mc,impactb,impact_angle,tmin)
    
    
def add_GCs():
    # M  rs RA   DEC   D  PMRA  PMDEC  Vlos
    dat=np.loadtxt('GC_catalog.dat')
       
    M=dat[:,0]
    rs=dat[:,1]
    RA=dat[:,2]
    DEC=dat[:,3]
    D=dat[:,4]
    PMRA=dat[:,5]
    PMDEC=dat[:,6]   
    Vlos=dat[:,7]
     
    coord=[]
       
    for ii in range(len(M)):
             coord.append([RA[ii],DEC[ii],D[ii],PMRA[ii],PMDEC[ii],Vlos[ii]])
               
    return (M,rs,coord)
    
    
def compute_impact_parameters_GC(timp,a,xs,ys,zs,pot=MWPotential2014,npart=64,sampling_low_file='',bmax=0.5,td=9.):
    
    '''
    timp : timpacts
    a,xs,ys,zs : list of array, each array decribes the stream at that time, no of arrays = timpacts
    sampling_low : low timpact object on to which the impacts from high timpact case will be set
    bmax: fixed max impact parameter 
       
    
    '''
    
    if pot != MWPotential2014 :
        chain_ind=int(pot)
        prog,pot,sigv,tvo=set_prog_potential(chain_ind)
        t_age= np.linspace(0.,td,1001)/bovy_conversion.time_in_Gyr(tvo,_REFR0)
        bovy_convert_mass=bovy_conversion.mass_in_msol(tvo,_REFR0)
        
    else : 
        #integrate their orbits td Gyr back,
        t_age= np.linspace(0.,td,1001)/bovy_conversion.time_in_Gyr(_REFV0,_REFR0)
        bovy_convert_mass=bovy_conversion.mass_in_msol(_REFV0,_REFR0)
       
    #load the GMCs
    #M,rs,coord=add_MCs(pot=pot,Mmin=Mmin,rand_rotate=rand_rotate,Rmax=Rmax,Rmin=Rmin)
    M,rs,coord=add_GCs()
 
    orbits=[]

    N=len(M)

    for ii in range(N):
    
        orbits.append(Orbit(coord[ii],radec=True,ro=8.,vo=220.,solarmotion=[-11.1,24.,7.25]).flip()) # flip flips the velocities for backwards integration
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

        bmax=bmax/_REFR0

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
    
    M_mc=np.array(M_mc)
    rs_mc=np.array(rs_mc)
    v_mc=np.array(v_mc)
    impactb=np.array(impactb)
    impact_angle=np.array(impact_angle)
    tmin=np.array(tmin)
        
    return (impactMC_ind,M_mc,rs_mc,v_mc,impactb,impact_angle,tmin)
    
    
