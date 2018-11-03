# script to run simulations of stream peppering
import os, os.path
import csv
import time
import pickle
import numpy
import matplotlib
matplotlib.use('agg')
from scipy import integrate, interpolate
from optparse import OptionParser
from galpy.util import bovy_conversion
import gd1_util
#import pal5_util
from gd1_util import R0,V0
from scipy.integrate import quad
from scipy.optimize import brentq


def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    
    parser.add_option("--ind",dest='ind',default=None,
                      type='int',
                      help="index of apar")
    return parser
    


            
apar=numpy.arange(0.03,1.15,0.01)

parser= get_options()
options,args= parser.parse_args()

ind=options.ind


index = [16,28,37,14,31,6,3,39,18,17,9,33,8,30,36,10,32,38,21,15,13,5,2,34,20]

sdf_smooth= gd1_util.setup_gd1model()
for jj in index :
    

        with open('impacted_pkl_files/GD1_7400_on_128impact_Plummer_td9.0_Mmin105_MW2014_{}.pkl'.format(jj),'rb') as savefile:
                #sdf_smooth=pickle.load(savefile,encoding='latin1')
                sdf_pepper= pickle.load(savefile,encoding='latin1')
        
        
        a=apar[options.ind]
        dens_unp= sdf_smooth._density_par(a)
        omega_unp=sdf_smooth.meanOmega(a,oned=True)
        
        densOmega= sdf_pepper._densityAndOmega_par_approx(a)
        
        
        fo=open('dens_Omega/GD1_densOmega_4096_on_128_Plummer_Mmin105_rand_rotate{}_{}.dat'.format(jj,ind),'w')
        fo.write('#apar   dens_unp   dens  omega_unp   omega' + '\n')
        fo.write(str(a) + '   ' + str(dens_unp) + '   ' + str(densOmega[0]) + '   ' + str(omega_unp) + '   ' + str(densOmega[1]) )
        
        fo.close()

