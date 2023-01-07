import os,sys
import sys

os.system('export SPS_HOME="/homes/bmb121/fsps"')
sys.path.insert(1, '/homes/bmb121/provabgs/src')

import numpy as np
from provabgs import util as UT
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib

from speclite import filters as specFilter
from numpy import asarray
from numpy import save
from numpy import load
import random

total=1000000
samples=2000
test_name="mil20"
filename="/vol/bitbucket/bmb121/"+test_name+".npy"

def mag(flux):
    x=-2.5*np.log10(flux) + 23.9 
    return x 

def get_bands(bands): 
        if bands is None: 
            return None
        if isinstance(bands, str): 
            if bands == 'cosmos2020': 
                # load galex filters 
                cfht_u= specFilter.load_filter('filters/cfht-u.ecsv')
                cfht_ustar = specFilter.load_filter('filters/cfht-ustar.ecsv')

                hsc_g = specFilter.load_filter('hsc2017-g')
                hsc_r = specFilter.load_filter('hsc2017-r')
                hsc_i = specFilter.load_filter('hsc2017-i')
                hsc_z = specFilter.load_filter('hsc2017-z')
                hsc_y = specFilter.load_filter('hsc2017-y')                
                
                IB427 = specFilter.load_filter('filters/IB427.ecsv')
                IB464 = specFilter.load_filter('filters/IB464.ecsv')
                IA484 = specFilter.load_filter('filters/IA484.ecsv')
                IB505 = specFilter.load_filter('filters/IB505.ecsv')
                IA527 = specFilter.load_filter('filters/IA527.ecsv')
                IB574 = specFilter.load_filter('filters/IB574.ecsv')
                IA624 = specFilter.load_filter('filters/IA624.ecsv')
                IA679 = specFilter.load_filter('filters/IA679.ecsv')
                IB709 = specFilter.load_filter('filters/IB709.ecsv')
                IA738 = specFilter.load_filter('filters/IA738.ecsv')
                IA767 = specFilter.load_filter('filters/IA767.ecsv')
                IB827 = specFilter.load_filter('filters/IB827.ecsv')
                
                uvista_y = specFilter.load_filter('filters/UVISTA-Y.ecsv')
                uvista_j = specFilter.load_filter('filters/UVISTA-J.ecsv')
                uvista_h = specFilter.load_filter('filters/UVISTA-H.ecsv')
                uvista_ks = specFilter.load_filter('filters/UVISTA-Ks.ecsv')
                
                ch1 = specFilter.load_filter('filters/irac-ch1.ecsv')
                ch2 = specFilter.load_filter('filters/irac-ch2.ecsv')
                ch3 = specFilter.load_filter('filters/irac-ch3.ecsv')
                ch4 = specFilter.load_filter('filters/irac-ch4.ecsv')

                filters = [cfht_u,cfht_ustar,hsc_g,hsc_r,hsc_i,hsc_z,hsc_y,IB427,IB464, 
                            IA484,IB505,IA527,IB574,IA624, IA679, IB709, IA738, IA767, IB827,
                          uvista_y,uvista_j,uvista_h,uvista_ks,ch1,ch2,ch3,ch4]
            else:
                raise NotImplementedError("specified bands not implemented") 
        else: 
            raise NotImplementedError("specified bands not implemented") 

        return specFilter.FilterSequence(filters)

    
m_nmf = Models.NMF(burst=True, emulator=False)


prior = Infer.load_priors([
    Infer.UniformPrior(7., 12.5, label='sed'),
    Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
    Infer.UniformPrior(0., 1., label='sed'), # burst fraction
    Infer.UniformPrior(1e-2, 13.27, label='sed'), # tburst
    Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
    Infer.UniformPrior(-2., 1., label='sed')    # uniform priors on dust_index
])

again=True
r=1
while again:
	thetas=np.zeros((samples,12))
	redshifts=[]
	for _ in range(samples):
    		thetas[_,:]=prior.transform(prior.sample())
    		redshifts+=[random.uniform(0, 6)]

	params=np.concatenate((np.array([redshifts]).T,np.array(thetas)),axis=1)
	w_obs, f_obs, filter_mags=m_nmf.seds(thetas,redshifts,filters=get_bands("cosmos2020"))
	for v in range(27):
		params=params[mag(filter_mags[:,v])<45,:]
		filter_mags=filter_mags[mag(filter_mags[:,v])<45,:]
	new_data = np.concatenate((params, filter_mags), axis=1)
	if r!=0:
		old_data = load(filename)
		new_data=np.concatenate((old_data, new_data), axis=0)
	if len(new_data)<total:

		save(filename, new_data)
		with open("/homes/bmb121/"+test_name+'_log.txt', 'w') as f:
			f.write('finished '+str(len(new_data)))
	else:
		save(filename,new_data[:total,:])
		with open("/homes/bmb121/"+test_name+'_log.txt', 'w') as f:
                	f.write('finished all '+str(len(new_data[:total,:])))
		again=False
	r+=1
	print(len(new_data[:total,:]))
