import numpy as np
import pickle
from astropy.io import fits
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from galpy.util import bovy_conversion, bovy_coords, save_pickles, bovy_plot
from galpy.potential import MWPotential2014, turn_physical_off, vcirc
import astropy.units as u
from galpy.orbit import Orbit
from scipy import integrate, interpolate
from scipy.integrate import quad
from optparse import OptionParser
import GMC_GC_util
import gd1_util
import pal5_util_MWfit
import MWPotential2014Likelihood
import numpy

ro=8.
vo=220.

def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    
        
####### SET THE STREAM #####################################                         
                         
    parser.add_option("--prog",dest='prog',default=0.,
                      type='float',
                      help="prog loc in phi1: 0.,-40.0")
                      
    parser.add_option("--sigv",dest='sigv',default=0.3,
                      type='float',
                      help="velocity dispersion")
                      
    parser.add_option("--ind_min",dest='ind_min',default=0,
                      type='int',
                      help="realization index min")
                      
    parser.add_option("--ind_max",dest='ind_max',default=50,
                      type='int',
                      help="realization index max")
                      
    parser.add_option("--miss_ind",dest='miss_ind',default=0,
                      type='int',
                      help="for re-running sims from a particular index incase of failed runs")
    
    parser.add_option("--batch",dest='batch',default=0,
                      type='int',
                      help="batch")
                      
    parser.add_option("--leading_arm",action="store_false",dest='leading_arm',default=True,
                      help="leading or trailing arm")
                      
#########################################################################################################################################
############### GMC STUFF ##############################################################    
    parser.add_option("--rand_seed",dest='rand_seed',default=None,
                      type='int',
                      help="random seed")
                      
    parser.add_option("--Mmin",dest='Mmin',default=5.,
                      type='float',
                      help="minimum mass of GMC in log10 to consider")
                      
    parser.add_option("--Rmax",dest='Rmax',default=100000000.,
                      type='float',
                      help="max galactocentric radius of GMC to consider")
                      
    parser.add_option("--Rmin",dest='Rmin',default=0.,
                      type='float',
                      help="min galactocentric radius of GMC to consider")
                                               
    parser.add_option("--non_zero_z",dest='non_zero_z',default=1,
                      type='int',
                      help="set z of the GMCs = 0 or not, 1: True, 0: False")
                      
#############################################################################################################################################
################# SUBHALO STUFF ############################################################################

# Parameters of this simulation
    
    parser.add_option("-X",dest='Xrs',default=5.,
                      type='float',
                      help="Number of times rs to consider for the impact parameter")
    
    parser.add_option("-M",dest='mass',default='6.5',
                      help="Mass or mass range to consider; given as log10(mass)")
                                            
    parser.add_option("--mwdm",dest='mwdm',default=1.5,type='float',
                      help="Mass of WDM in keV") 
                      
    parser.add_option("--ravg",dest='ravg',default=20.,type='float',
                      help="r_avg of the stream in kpc")                  
                     
    parser.add_option("--cutoff",dest='cutoff',default=None,type='float',
                      help="Log10 mass cut-off in power-spectrum")
                      
    parser.add_option("--massexp",dest='massexp',default=-2.,type='float',
                      help="Exponent of the mass spectrum (doesn't work with cutoff)")
                      
    parser.add_option("--ratemin",dest='ratemin',default=-1.,type='float',
                      help="minimum timescdm")
                      
    parser.add_option("--ratemax",dest='ratemax',default=1.,type='float',
                      help="maximum timescdm")
                      
    parser.add_option("--rsfac",dest='rsfac',default=1.,type='float',
                      help="Use a r_s(M) relation that is a factor of rsfac different from the fiducial one")
                      
    parser.add_option("--plummer",action="store_true", 
                      dest="plummer",default=False,
                      help="If set, use a Plummer DM profile rather than Hernquist")
                                               
    parser.add_option("--sigma",dest='sigma',default=120.,type='float',
                      help="Velocity dispersion of the population of DM subhalos")
    
    return parser

parser= get_options()
options,args= parser.parse_args()



###########SUBHALO STUFF#############
def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
            for ti in times.split(',')]
def parse_mass(mass):   
    return [float(m) for m in mass.split(',')]

# Functions to sample

def nsubhalo(m):
    return 0.3*(10.**6.5/m)

def rs(m,plummer=options.plummer,rsfac=1.):
    if plummer:
        #print ('Plummer')
        return 1.62*rsfac/ro*(m/10.**8.)**0.5
    else:
        return 1.05*rsfac/ro*(m/10.**8.)**0.5

h=0.6774


def alpha(m_wdm):
    return (0.048/h)*(m_wdm)**(-1.11) #in Mpc , m_wdm in keV

def lambda_hm(m_wdm):
    nu=1.12
    return 2*numpy.pi*alpha(m_wdm)/(2**(nu/5.) - 1.)**(1/(2*nu))

def M_hm(m_wdm):
    Om_m=0.3089
    rho_c=1.27*10**11 #Msun/Mpc^3 
    rho_bar=Om_m*rho_c
    return (4*numpy.pi/3)*rho_bar*(lambda_hm(m_wdm)/2.)**3

def Einasto(r):
    al=0.678 #alpha_shape
    rm2=199 #kpc, see Erkal et al 1606.04946 for scaling to M^1/3
    return numpy.exp((-2./al)*((r/rm2)**al -1.))*4*numpy.pi*(r**2)

def dndM_cdm(M,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #c0kpc=2.02*10**(-13) #Msun^-1 kpc^-3 from Denis' paper
    m0=2.52*10**7 #Msun from Denis' paper
    return c0kpc*((M/m0)**mf_slope)

def fac(M,m_wdm):
    beta=-0.99
    gamma=2.7
    return (1.+gamma*(M_hm(m_wdm)/M))**beta
    
def dndM_wdm(M,m_wdm,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    return fac(M,m_wdm)*dndM_cdm(M,c0kpc=2.02*10**(-13),mf_slope=-1.9)

def nsub_cdm(M1,M2,r=20.,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #number density of subhalos in kpc^-3
    m1=10**(M1)
    m2=10**(M2)
    return integrate.quad(dndM_cdm,m1,m2,args=(c0kpc,mf_slope))[0]*integrate.quad(Einasto,0.,r)[0]*(8.**3.)/(4*numpy.pi*(r**3)/3) #in Galpy units

def nsub_wdm(M1,M2,m_wdm,r=20.,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    m1=10**(M1)
    m2=10**(M2)
    return integrate.quad(dndM_wdm,m1,m2,args=(m_wdm,c0kpc,mf_slope))[0]*integrate.quad(Einasto,0.,r)[0]*(8.**3)/(4*numpy.pi*(r**3)/3) #in Galpy units    
    

#def dNencdm(sdf_pepper,m,M1,M2,m_wdm,ravg,Xrs=3.,plummer=options.plummer,rsfac=1.,sigma=120.): #m for sampling r_s, until fixed
#     return sdf_pepper.subhalo_encounters(sigma=sigma/vo,nsubhalo=nsub_wdm(M1,M2,m_wdm,ravg),bmax=Xrs*rs(m,plummer=plummer,rsfac=rsfac))

def powerlaw_wcutoff(massrange,cutoff):
    accept= False
    while not accept:
        prop= (10.**-(massrange[0]/2.)+(10.**-(massrange[1]/2.)\
                         -10.**(-massrange[0]/2.))\
                   *numpy.random.uniform())**-2.
        if numpy.random.uniform() < numpy.exp(-10.**cutoff/prop):
            accept= True
    return prop/bovy_conversion.mass_in_msol(vo,ro)


def simulate_subhalos_vary_amp_slope_mwdm(sdf_pepper,m_wdm,mf_slope=-1.9,c0kpc=2.02*10**(-13),r=20.,Xrs=5.,sigma=120./220.):
    
    '''
    Sample amp and slope such that dN/dM = amp*M^slope and simulate subhalo impacts
    '''
    
    Mbin_edge=[5.,6.,7.,8.,9.]
    Nbins=len(Mbin_edge)-1
    #compute number of subhalos in each mass bin
    nden_bin=np.empty(Nbins)
    rate_bin=np.empty(Nbins)
    for ll in range(Nbins):
        nden_bin[ll]=nsub_wdm(Mbin_edge[ll],Mbin_edge[ll+1],m_wdm=m_wdm,r=r,c0kpc=c0kpc,mf_slope=mf_slope) 
        Mmid=10**(0.5*(Mbin_edge[ll]+Mbin_edge[ll+1]))
        rate_bin[ll]=sdf_pepper.subhalo_encounters(sigma=sigma,nsubhalo=nden_bin[ll],bmax=Xrs*rs(Mmid,plummer=True))

    rate = np.sum(rate_bin)
          
    Nimpact= numpy.random.poisson(rate)
              
    norm= 1./quad(lambda M : fac(M,m_wdm)*((M)**(mf_slope +0.5)),10**(Mbin_edge[0]),10**(Mbin_edge[Nbins]))[0]

    def cdf(M):
        return quad(lambda M : norm*fac(M,m_wdm)*(M)**(mf_slope +0.5),10**Mbin_edge[0],M)[0]

    MM=numpy.linspace(Mbin_edge[0],Mbin_edge[Nbins],10000)

    cdfl=[cdf(i) for i in 10**MM]
    icdf= interpolate.InterpolatedUnivariateSpline(cdfl,10**MM,k=1)
    timpact_sub= numpy.array(sdf_pepper._uniq_timpact)[numpy.random.choice(len(sdf_pepper._uniq_timpact),size=Nimpact,
                                p=sdf_pepper._ptimpact)]
    # Sample angles from the part of the stream that existed then
    impact_angle_sub= numpy.array([sdf_pepper._icdf_stream_len[ti](numpy.random.uniform())
            for ti in timpact_sub])
    
    sample_GM=lambda: icdf(numpy.random.uniform())/bovy_conversion.mass_in_msol(vo,ro)
    GM_sub= numpy.array([sample_GM() for a in impact_angle_sub])
    rs_sub= numpy.array([rs(gm*bovy_conversion.mass_in_msol(vo,ro)) for gm in GM_sub])
    # impact b
    impactb_sub= (2.*numpy.random.uniform(size=len(impact_angle_sub))-1.)*Xrs*rs_sub
    # velocity
    
    subhalovel_sub= numpy.empty((len(impact_angle_sub),3))
    for ii in range(len(timpact_sub)):
        subhalovel_sub[ii]=sdf_pepper._draw_impact_velocities(timpact_sub[ii],sigma,impact_angle_sub[ii],n=1)[0]
    # Flip angle sign if necessary
    #if not sdf_pepper._gap_leading: impact_angles*= -1.
    #angles not flipped, flip them after including angles from GMC and GC impacts     
    return impact_angle_sub,impactb_sub,subhalovel_sub,timpact_sub,GM_sub,rs_sub         
                

sigv=options.sigv
minM=10**(options.Mmin)
            
#prog == -40.0
new_orb_lb=[188.04928416766532, 51.848594007807456, 7.559027173643999, 12.260258757214746, -5.140630283489461, 7.162732847549563]
isob=0.45
tstream=3.2
sampling_hi=3200
sampling_low=64


if options.leading_arm :
            sample_low='pkl_files/pklfiles_3.2Gyr_sigv0.3_streamwidth11.5arcmins/gd1_smooth_pepper_leading_Plummer_sigv0.3_streamwidth11.5arcmins_td3.2_64sampling_progphi1-40.0_MW2014.pkl'
            timpact,_apar,_xstream,_ystream,_zstream,_vxstream,_vystream,_vzstream=GMC_GC_util.aparxv_GD1_stream_from_multiple_pkl(pot=MWPotential2014,prog=-40.,td=3.2,sigv=sigv,npart=50,leading=True)
            arm='leading'
else :
            sample_low='pkl_files/pklfiles_3.2Gyr_sigv0.3_streamwidth11.5arcmins/gd1_smooth_pepper_trailing_Plummer_sigv0.3_streamwidth11.5arcmins_td3.2_64sampling_progphi1-40.0_MW2014.pkl'
            timpact,_apar,_xstream,_ystream,_zstream,_vxstream,_vystream,_vzstream=GMC_GC_util.aparxv_GD1_stream_from_multiple_pkl(pot=MWPotential2014,prog=-40.,td=3.2,sigv=sigv,npart=50,leading=False)
            arm='trailing'

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


rand_dat_all=np.loadtxt('random_amp_slope.dat')
batchsize=options.ind_max - options.ind_min
rand_params=rand_dat_all[options.batch*batchsize:(options.batch+1)*batchsize]
print (options.batch)
           

for ii in range(len(rand_params)):
            
    with open(sample_low,'rb') as savefile:
        sdf_smooth= pickle.load(savefile,encoding='latin1')
        sdf_pepper= pickle.load(savefile,encoding='latin1')
    
    impact_angles=[]
    impactbs=[]
    subhalovels=[]
    timpacts=[]
    GMs=[]
    rss=[]                                                
    print (rand_params[ii+options.miss_ind])
    slope= rand_params[ii+options.miss_ind][0]  
    rand_norm=rand_params[ii+options.miss_ind][1]*2.02*10**(-13)                                                                                         
    impact_angle_sub,impactb_sub,subhalovel_sub,timpact_sub,GM_sub,rs_sub =simulate_subhalos_vary_amp_slope(sdf_pepper,r=options.ravg,mf_slope=slope,c0kpc=rand_norm,Xrs=options.Xrs)    
                    
    print ('%i subhalo impact'%len(GM_sub))   
    
    impact_angles+=list(impact_angle_sub)
    impactbs+=list(impactb_sub)
    subhalovels+=list(subhalovel_sub)
    timpacts+=list(timpact_sub)
    GMs+=list(GM_sub)
    rss+=list(rs_sub) 
     
    _,M_mc,rs_mc,v_mc,impactb_mc,impact_angle_mc,tmin_mc=GMC_GC_util.compute_impact_parameters(timpact,apar,x_stream,y_stream,z_stream,pert_type='GMC',
                                                    pot=MWPotential2014,sampling_low_file=sample_low,Mmin=minM,td=tstream,Rmax=options.Rmax,
                                                    Rmin=options.Rmin,rand_rotate=True,fill_holes=True)
    print ('%i GMC impacts'%len(M_mc)) 
    impact_angles+=list(impact_angle_mc)
    impactbs+=list(impactb_mc)
    subhalovels+=list(v_mc)
    timpacts+=list(tmin_mc)
    GMs+=list(M_mc)
    rss+=list(rs_mc)
                                                        
    
    _,M_gc,rs_gc,v_gc,impactb_gc,impact_angle_gc,tmin_gc=GMC_GC_util.compute_impact_parameters(timpact,apar,x_stream,y_stream,z_stream,pert_type='GC',
                                                    pot=MWPotential2014,sampling_low_file=sample_low,td=tstream)
    print ('%i GC impacts'%len(M_gc)) 
    impact_angles+=list(impact_angle_gc)
    impactbs+=list(impactb_gc)
    subhalovels+=list(v_gc)
    timpacts+=list(tmin_gc)
    GMs+=list(M_gc)
    rss+=list(rs_gc)
    
    
    print ("%i total impacts"%len(GMs))  
    
    if len(GMs) == 0 : #no hits
        print ("no hits")
        apar_out=np.arange(0.01,1.,0.01)
        #sdf_smooth=gd1_util.setup_gd1model(leading=options.leading_arm,age=tstream,new_orb_lb=new_orb_lb,isob=isob,sigv=sigv)
        dens_unp= [sdf_smooth._density_par(a) for a in apar_out]
        omega_unp= [sdf_smooth.meanOmega(a,oned=True) for a in apar_out]
      
        fo=open('dens_Omega/sims_for_ABC_amp_slope/'+arm+'/GD1_prog{}_sigv{}_{}_vary_amp_slope_densOmega_{}_on_{}_subhalo_GMC_Plummer_Mmin10{}_rand_rotate_batch{}_{}.dat'.format(options.prog,sigv,
                                                                                                                    arm,sampling_hi,sampling_low,
                                                                                                                    options.Mmin,options.batch,
                                                                                                                    ii+options.miss_ind),'w')
        fo.write('#apar   dens_unp   dens  omega_unp   omega' + '\n')
        
        for  jj in range(len(apar_out)):
            fo.write(str(slope)+'    '+str(rand_params[ii+options.miss_ind][1])+'    ' + str(apar_out[jj]) + '   ' + str(dens_unp[jj]) + '   ' + str(dens_unp[jj]) + '   ' + str(omega_unp[jj]) + '   ' + str(omega_unp[jj]) + '\n' )
                
        fo.close()
        
    
    else :
        impact_angles=np.array(impact_angles)
        impactbs=np.array(impactbs)
        subhalovels=np.vstack(subhalovels)
        timpacts=np.array(timpacts)
        GMs=np.array(GMs)
        rss=np.array(rss)   
        
        #print (impact_angles,subhalovels) 
                
        # Flip angle sign if necessary
        if not sdf_pepper._gap_leading: impact_angles*= -1.     
        
        sdf_pepper.set_impacts(impactb=impactbs,subhalovel=subhalovels,impact_angle=impact_angles,timpact=timpacts,rs=rss,GM=GMs)
        
        #pepperfilename='impacted_pkl_files/GD1_{}_on_128impact_GMCs_subhalos_Plummer_td{}_Mmin105_MW2014_{}.pkl'.format(options.nsam,options.td,options.ind)
        
        #save_pickles(pepperfilename,sdf_pepper)
     
    
        apar_out=np.arange(0.01,1.,0.01)
        #sdf_smooth=gd1_util.setup_gd1model(leading=options.leading_arm,age=tstream,new_orb_lb=new_orb_lb,isob=isob,sigv=sigv)
        
        dens_unp= [sdf_smooth._density_par(a) for a in apar_out]
        omega_unp= [sdf_smooth.meanOmega(a,oned=True) for a in apar_out]
        densOmega= np.array([sdf_pepper._densityAndOmega_par_approx(a) for a in apar_out]).T
        
        
        fo=open('dens_Omega/sims_for_ABC_amp_slope/'+arm+'/GD1_prog{}_sigv{}_{}_vary_amp_slope_densOmega_{}_on_{}_subhalo_GMC_Plummer_Mmin10{}_rand_rotate_batch{}_{}.dat'.format(options.prog,sigv,
                                                                                                                    arm,sampling_hi,sampling_low,
                                                                                                                    options.Mmin,options.batch,
                                                                                                                    ii+options.miss_ind),'w')
        
                                                                                                                
        fo.write('#apar   dens_unp   dens  omega_unp   omega' + '\n')
        
        for  jj in range(len(apar_out)):
            fo.write(str(slope)+'    '+str(rand_params[ii+options.miss_ind][1])+'    ' + str(apar_out[jj]) + '   ' + str(dens_unp[jj]) + '   ' + str(densOmega[0][jj]) + '   ' + str(omega_unp[jj]) + '   ' + str(densOmega[1][jj]) + '\n' )
                
        fo.close()





