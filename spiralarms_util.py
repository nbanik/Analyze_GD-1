import glob
import pickle
import numpy
import numpy as np
from numpy.polynomial import Polynomial
from scipy import ndimage, signal, interpolate, integrate
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, turn_physical_off, plotDensities,evaluateDensities,plotPotentials,SpiralArmsPotential
from galpy.util import bovy_conversion, save_pickles, bovy_coords, bovy_plot
from scipy import ndimage, signal, interpolate, integrate,optimize
import pal5_util
import gd1_util
import seaborn as sns
import astropy.units as u
from galpy import potential
from matplotlib import cm, pyplot
from galpy.potential import DehnenSmoothWrapperPotential as DehnenWrap
from galpy.potential import SCFPotential
import streamspraydf
from gd1_util_MWhaloshape import phi12_to_lb_6d

ro=8.
vo=220.


def spiral_arms_potential(FR_frac=1.,t_on=-5.,tgrow=2,tstream=5.,axi_pot=MWPotential2014,cos=True,N=2,pat_speed=24.5,pitch_angle=9.9,r_ref=8.,Rs=7.,phi0=26.,H=0.3):
    
    
        phi0=np.radians(phi0)
        omega=pat_speed*(ro/vo)
        alpha=numpy.radians(pitch_angle)
        r_ref/=ro
        Rs/=ro
        H/=ro

        # percentage of the radial force to set the amplitude of the spiral
        FR_frac=FR_frac*0.01  
        
        if cos :
            Cs=[8./(3.*numpy.pi),0.5,8./(15.*numpy.pi)]
            
        else :
            Cs=[1]
            
        #compute max radial force for amp=1
        pp=np.linspace(0.,2.*np.pi,1000)
        FR_nonaxi=[]
        spiral_pot_amp1=SpiralArmsPotential(amp=1.,N=N,omega=omega,alpha=alpha,phi_ref=phi0,r_ref=r_ref,H=H,Rs=Rs,Cs=Cs)
        turn_physical_off(spiral_pot_amp1)
        
        for ii in range(len(pp)):
             FR_nonaxi.append(potential.evaluateRforces(spiral_pot_amp1,R=8./ro,z=0.,phi=pp[ii],t=0.))

        interp_FR_nonaxi= interpolate.InterpolatedUnivariateSpline(pp,FR_nonaxi)

        #fmin, because radial force is negative
        max_phi=optimize.fmin(interp_FR_nonaxi, 0.,disp=0)

        max_FR_nonaxi= interp_FR_nonaxi(max_phi)[0]
                
        #compute the radial force due to the axisymmetric potential
        FR_axi=potential.evaluateRforces(axi_pot,R=8./ro,z=0.,t=0.)

        #compute the correct amplitude
        amp= numpy.abs(FR_frac*FR_axi/max_FR_nonaxi)
        
        #setup spiral potential with correct amplitude
        spiralpot=SpiralArmsPotential(amp=amp,N=N,omega=omega,alpha=alpha,phi_ref=phi0,r_ref=r_ref,H=H,Rs=Rs,Cs=Cs)
        
        #grow the spirals
        
        Tspiral=2.*np.pi/np.abs(omega) #bar period in galpy units.
        t_on=t_on/bovy_conversion.time_in_Gyr(vo,ro)
        tsteady=tgrow*Tspiral
        tform = t_on - tsteady #- because past is negative
        
        #if t_on >= t_pal5_age, then Pal 5 sees the spirals always on
        if np.abs(t_on)*bovy_conversion.time_in_Gyr(vo,ro) >= tstream :
            print ('not growing spiral')
            MWspiralpot = axi_pot + [spiralpot]
            turn_physical_off(MWspiralpot)
            
            return (MWspiralpot)

        elif np.abs(tform)*bovy_conversion.time_in_Gyr(vo,ro) >= tstream:
            print ("tform > age of stream")

        elif np.abs(tform)*bovy_conversion.time_in_Gyr(vo,ro) < tstream :
            print ('growing spiral')

            spiralpot_grow=DehnenWrap(amp=1.,pot=spiralpot,tform=tform,tsteady=tsteady)  
            turn_physical_off(spiralpot_grow)
            MWspiralpot = axi_pot + [spiralpot_grow]
            return MWspiralpot
            
def spiral_only_potential(FR_frac=1.,t_on=-5.,tgrow=2,tstream=5.,axi_pot=MWPotential2014,cos=True,N=2,pat_speed=24.5,pitch_angle=9.9,r_ref=8.,Rs=3.,phi0=26.,H=0.3):
    phi0=np.radians(phi0)
    omega=pat_speed*(ro/vo)
    alpha=numpy.radians(pitch_angle)
    r_ref/=ro
    Rs/=ro
    H/=ro

    # percentage of the radial force to set the amplitude of the spiral
    FR_frac=FR_frac*0.01  

    if cos :
        Cs=[8./(3.*numpy.pi),0.5,8./(15.*numpy.pi)]

    else :
        Cs=[1]

    #compute max radial force for amp=1
    pp=np.linspace(0.,2.*np.pi,1000)
    FR_nonaxi=[]
    spiral_pot_amp1=SpiralArmsPotential(amp=1.,N=N,omega=omega,alpha=alpha,phi_ref=phi0,r_ref=r_ref,H=H,Rs=Rs,Cs=Cs)
    turn_physical_off(spiral_pot_amp1)

    for ii in range(len(pp)):
         FR_nonaxi.append(potential.evaluateRforces(spiral_pot_amp1,R=8./ro,z=0.,phi=pp[ii],t=0.))

    interp_FR_nonaxi= interpolate.InterpolatedUnivariateSpline(pp,FR_nonaxi)

    #fmin, because radial force is negative
    max_phi=optimize.fmin(interp_FR_nonaxi, 0.,disp=0)

    max_FR_nonaxi= interp_FR_nonaxi(max_phi)[0]

    #compute the radial force due to the axisymmetric potential
    FR_axi=potential.evaluateRforces(axi_pot,R=8./ro,z=0.,t=0.)

    #compute the correct amplitude
    amp= numpy.abs(FR_frac*FR_axi/max_FR_nonaxi)

    #setup spiral potential with correct amplitude
    spiralpot=SpiralArmsPotential(amp=amp,N=N,omega=omega,alpha=alpha,phi_ref=phi0,r_ref=r_ref,H=H,Rs=Rs,Cs=Cs)

    #grow the spirals

    Tspiral=2.*np.pi/np.abs(omega) #bar period in galpy units.
    t_on=t_on/bovy_conversion.time_in_Gyr(vo,ro)
    tsteady=tgrow*Tspiral
    tform = t_on - tsteady #- because past is negative

    #if t_on >= t_pal5_age, then Pal 5 sees the spirals always on
    if np.abs(t_on)*bovy_conversion.time_in_Gyr(vo,ro) >= tstream :
        print ('not growing spiral')
        turn_physical_off(spiralpot)
        return (spiralpot)

    elif np.abs(tform)*bovy_conversion.time_in_Gyr(vo,ro) >= tstream:
        print ("tform > age of stream")

    elif np.abs(tform)*bovy_conversion.time_in_Gyr(vo,ro) < tstream :
        print ('growing spiral')

        spiralpot_grow=DehnenWrap(amp=1.,pot=spiralpot,tform=tform,tsteady=tsteady)  
        turn_physical_off(spiralpot_grow)
        
        return (spiralpot_grow)

def sample_streamdf_noprog_spiral(Nsamples,spiralpot,stream='GD1',nospiralpot=MWPotential2014,fo='spiral_trailing.dat',trailing=True,write=False):
        
        '''
        Sample N points for a given stream using the freq-angle framework
        i.e., sample in the nobarpot, then integrate them back to stripping time
        and then integrat them forward in barred potential. Effects of progenitor's motion
        is not considered
        '''
        print ("WARNING: Effects of progenitor's motion neglected")
        
        if stream == 'Pal5' :
            sdf_trailing= pal5_util.setup_pal5model(pot=nospiralpot)
            sdf_leading= pal5_util.setup_pal5model(pot=nospiralpot,leading=True)
            
        elif stream == 'GD1' :
            sdf_trailing= gd1_util.setup_gd1model(pot=nospiralpot,leading=False)
            sdf_leading= gd1_util.setup_gd1model(pot=nospiralpot)
        
        fo= stream + fo
              
        if trailing :
            R,vR,vT,z,vz,phi,dt= sdf_trailing.sample(n=Nsamples,returndt=True)
            fo=open(fo,'w')
          
        else :
            R,vR,vT,z,vz,phi,dt= sdf_leading.sample(n=Nsamples,returndt=True)
            fo_lead=fo.replace('trailing','leading')
            fo=open(fo_lead,'w')
              
        finalR= numpy.empty(Nsamples)
        finalvR=numpy.empty(Nsamples)
        finalvT=numpy.empty(Nsamples)
        finalvz=numpy.empty(Nsamples)
        finalphi= numpy.empty(Nsamples)
        finalz= numpy.empty(Nsamples)
        tt=numpy.empty(Nsamples)

        for ii in range(Nsamples):

                o= Orbit([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]])
                o.turn_physical_off()
                ts= numpy.linspace(0.,-dt[ii],1001)

                o.integrate(ts,nospiralpot)
                orb=Orbit([o.R(ts[-1]),o.vR(ts[-1]),o.vT(ts[-1]),o.z(ts[-1]),o.vz(ts[-1]),o.phi(ts[-1])])
                                
                ts_future= numpy.linspace(-dt[ii],0.,1001)
                #forward integrate in barred potential
                orb.integrate(ts_future,spiralpot)
                finalR[ii]= orb.R(ts_future[-1])
                finalphi[ii]= orb.phi(ts_future[-1])
                finalz[ii]= orb.z(ts_future[-1])
                finalvR[ii]=orb.vR(ts_future[-1])
                finalvT[ii]=orb.vT(ts_future[-1])
                finalvz[ii]=orb.vz(ts_future[-1])
                tt[ii]=dt[ii]
                
           
        if write :
            fo.write("#R   phi   z   vR    vT    vz    ts" + "\n")
        
            for jj in range(Nsamples):
                fo.write(str(finalR[jj]) + "   " + str(finalphi[jj]) + "   " + str(finalz[jj]) + "   " + str(finalvR[jj]) + "   " + str(finalvT[jj]) + "   " + str(finalvz[jj]) + "   " + str(tt[jj]) + "\n")
            
            fo.close()
        
        else :
            
            return (finalR, finalphi, finalz, finalvR, finalvT, finalvz, tt)
        
        
def sample_spraydf_noprog_spiral(Nsamples,spiralpot,stream='GD1',nospiralpot=MWPotential2014,Mprogenitor = 50000.,fo='blah_trailing.dat',trailing=True,tage=5.,write=False):
    
        '''
        Sample N points for a given stream using the Particle-Spray framework
        Effects of progenitor's motion is considered
        '''
        print ("WARNING: Using particle spray technique, effects of progenitor's orbit incorporated")
    
        
        if stream == 'Pal5' :
            o = Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],radec=True,ro=ro,vo=vo,solarmotion=[-11.1,24.,7.25])
            
            #convert to galpy units
            orb=Orbit(o._orb.vxvv)
            
        elif stream == 'GD1':
            o=Orbit(phi12_to_lb_6d(0,-0.82,10.1,-8.5,-2.15,-257.),lb=True,solarmotion=[-11.1,24.,7.25],ro=8.,vo=220.)
            #convert to galpy units
            orb=Orbit(o._orb.vxvv)

        
        fo= stream + fo

        if trailing :
            spdft= streamspraydf.streamspraydf(Mprogenitor*u.Msun,progenitor=orb,pot=nospiralpot,leading=False,tdisrupt=tage*u.Gyr)
            RvR,dt= spdft.sample(n=Nsamples,returndt=True,integrate=False)
            R=RvR[0]
            vR=RvR[1]
            vT=RvR[2]
            z=RvR[3]
            vz=RvR[4]
            phi=RvR[5]
            
            fo=open(fo,'w')
            
     
        else :
            spdf= streamspraydf.streamspraydf(Mprogenitor*u.Msun,progenitor=orb,pot=nospiralpot,tdisrupt=tage*u.Gyr)
            RvR,dt= spdf.sample(n=Nsamples,returndt=True,integrate=False)
            R=RvR[0]
            vR=RvR[1]
            vT=RvR[2]
            z=RvR[3]
            vz=RvR[4]
            phi=RvR[5]
            fo_lead=fo.replace('trailing','leading')
            fo=open(fo_lead,'w')
              
        finalR= numpy.empty(Nsamples)
        finalvR=numpy.empty(Nsamples)
        finalvT=numpy.empty(Nsamples)
        finalvz=numpy.empty(Nsamples)
        finalphi= numpy.empty(Nsamples)
        finalz= numpy.empty(Nsamples)
        tt=numpy.empty(Nsamples)

        for ii in range(Nsamples):

                orb= Orbit([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]])
                orb.turn_physical_off()
                                                           
                ts_future= numpy.linspace(-dt[ii],0.,1001)
                #forward integrate in barred potential
                orb.integrate(ts_future,spiralpot)
                finalR[ii]= orb.R(ts_future[-1])
                finalphi[ii]= orb.phi(ts_future[-1])
                finalz[ii]= orb.z(ts_future[-1])
                finalvR[ii]=orb.vR(ts_future[-1])
                finalvT[ii]=orb.vT(ts_future[-1])
                finalvz[ii]=orb.vz(ts_future[-1])
                tt[ii]=dt[ii]
                
        if write :
            
        
            fo.write("#R   phi   z   vR    vT    vz    ts" + "\n")
        
            for jj in range(Nsamples):
                fo.write(str(finalR[jj]) + "   " + str(finalphi[jj]) + "   " + str(finalz[jj]) + "   " + str(finalvR[jj]) + "   " + str(finalvT[jj]) + "   " + str(finalvz[jj]) + "   " + str(tt[jj]) + "\n")
            
            fo.close()
    
        else :
            return (finalR, finalphi, finalz, finalvR, finalvT, finalvz, tt)
        
        
        
def sample_spraydf_spiral(Nsamples,spiralpot,stream='GD1',fo='blah_trailing.dat',Mprogenitor=50000.,trailing=True,tage=5.,write=False):
    
        '''
        Sample N points for a given stream using the Particle-Spray framework
        Effects of progenitor's motion is considered
        '''
        print ("WARNING: Using particle spray technique, effects of progenitor's orbit incorporated")
    
        
        if stream == 'Pal5' :
            o = Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],radec=True,ro=ro,vo=vo,solarmotion=[-11.1,24.,7.25])
            
            #convert to galpy units
            orb=Orbit(o._orb.vxvv)
            
        elif stream == 'GD1':
            o=Orbit(phi12_to_lb_6d(0,-0.82,10.1,-8.5,-2.15,-257.),lb=True,solarmotion=[-11.1,24.,7.25],ro=8.,vo=220.)
            #convert to galpy units
            orb=Orbit(o._orb.vxvv)

        if trailing :
            spdft= streamspraydf.streamspraydf(Mprogenitor*u.Msun,progenitor=orb,pot=spiralpot,leading=False,tdisrupt=tage*u.Gyr)
            RvR,dt= spdft.sample(n=Nsamples,returndt=True,integrate=True)
            R=RvR[0]
            vR=RvR[1]
            vT=RvR[2]
            z=RvR[3]
            vz=RvR[4]
            phi=RvR[5]
            
            fo=open(fo,'w')
            
     
        else :
            spdf= streamspraydf.streamspraydf(Mprogenitor*u.Msun,progenitor=orb,pot=spiralpot,tdisrupt=tage*u.Gyr)
            RvR,dt= spdf.sample(n=Nsamples,returndt=True,integrate=True)
            R=RvR[0]
            vR=RvR[1]
            vT=RvR[2]
            z=RvR[3]
            vz=RvR[4]
            phi=RvR[5]
            fo_lead=fo.replace('trailing','leading')
            fo=open(fo_lead,'w')
              
                       
        if write :
                           
            fo.write("#R   phi   z   vR    vT    vz    ts" + "\n")
        
            for jj in range(Nsamples):
                fo.write(str(R[jj]) + "   " + str(phi[jj]) + "   " + str(z[jj]) + "   " + str(vR[jj]) + "   " + str(vT[jj]) + "   " + str(vz[jj]) + "   " + str(dt[jj]) + "\n")
            
            fo.close()
            
        else :
            
            return (R,phi,z,vR,vT,vz,dt)