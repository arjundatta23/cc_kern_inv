#!/usr/bin/python

# General purpose modules
import os
import sys
import numpy as np
import scipy.signal as ss
import scipy.stats as sst
import scipy.special as ssp
import scipy.optimize as sop
import scipy.integrate as spi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.path.append('../modules_common')
    # path to common modules for the cc source and structure inversion codes
    sys.path.append(os.path.expanduser('~/code_general/modules.python'))
    # path to the "SW1D_earthsr" set of modules
    sys.path.append(os.path.expanduser('~/code_general/Bharath_Shekar'))
    # path to the "Helmholtz_equation_FD" and "devito_solvers_TD" modules

# Custom modules (unconditional)
import config_file as config
import anseicca_utils1 as u1

###########################################################################################################################################################
# Documentation for reference

# 1. Curve fitting basic: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# 2. Curve fitting with weights example: https://scipython.com/book/chapter-8-scipy/examples/weighted-and-non-weighted-least-squares-fitting/

###########################################################################################################################################################

if not __name__ == '__main__':
# get essential variables from main (calling) program
    try:
        efile_ray = sys.modules['__main__'].egn_ray
        dfile_ray = sys.modules['__main__'].disp_ray
    except AttributeError:
        # this is the 2D-scalar case
        scalar=True
        elastic=False
    else:
        scalar=False
        elastic=True
        nzmax = sys.modules['__main__'].nz
        discon_mod = sys.modules['__main__'].hif_mod
        Zpts_all = sys.modules['__main__'].dep_pts_mod
        try:
            efile_lov = sys.modules['__main__'].egn_lov
            dfile_lov = sys.modules['__main__'].disp_lov
            love=True
        except AttributeError:
            love=False

############################################## Global variables ##########################################################
# pertaining to modelling domain geometry
dg = config.dom_geom
dx = dg.dx
ngpmb = dg.ngp_box
hlbox_outer = dg.box_len
""" NB: meaning of main box and outer box - main box represents the actual modelling domain, while the outer box
is a trick used for expedient calculation of GFs IN CASE OF LATERALLY HOMOGENEOUS MODELS (0-D and 1-D).
(GF computed over outer box for source at domain centre, gives GF over main box, for a source located anywhere within the main box)

omost_fac" is an integer factor that determines the size of the true model in the synthetic case
	It must be >=2;
		if = 2, standard case (true and inverted models are of the same size)
		if = 3, the true model is the size of the "outer box" (four times the area of the main box)
In the standard/default case, all models are the size of the "main box" (half the side length of outer box)
"""

# pertaining to source model parameterization
param_mod_tru = config.tru_mod_type
param_mod_inv = config.inv_mod_type
mod_specs = {'mg': config.somod_mg_specs, 'rg': config.somod_rg_specs}

# pertaining to structure model
c_scal = config.scal_mod.wavspeed_scal2D
lat_homo_tru = config.tru_struc_lat_homo
lat_homo_inv = config.inv_struc_lat_homo

# pertaining to modelling system
dc_xyz = config.comp_dic_xyz
dc_rtz = config.comp_dic_rtz
tensor_rtz = config.green_tensor_rtz

modelling_tru = config.tru_mdlng_type
modelling_inv = config.inv_mdlng_type

if modelling_tru == modelling_inv:
    init_amp_scaling = False
else:
    init_amp_scaling = config.amp_scaling

def sORe(instr):
    if 'scal' in instr:
        return 'scal'
    elif 'elas' in instr:
        return 'elas'

ncomp = {'scal': 1, 'elas': 3}
# number of components (of motion) in the scalar and elastic cases
use_p = {'scal': 'comp_scal', 'elas': 'comp_p'}
use_q = {'scal': 'comp_scal', 'elas': 'comp_q'}

sco = 2 #3 - ncomp[sORe(modelling_tru)]
scs = 2 #3 - ncomp[sORe(modelling_inv)]

# Reset, if necessary, some config-file parameters when working with real data
# (this is just to allow for the possibility that these parameters may not have been set correctly in the config file)
if config.reald:
    lat_homo_tru = False
    init_amp_scaling = True

# pertinaing to the inverse problem
omost_fac = config.invc.ofac
gamma = config.invc.gamma_inv

# Finally, custom modules required conditionally
mlg1='anal_elas_1D'
mlg2='num_scal_2D_FD'
mlg3='num_scal_2D_TD'
if modelling_tru==mlg1 or modelling_inv==mlg1:
    import SW1D_earthsr.Green_functions_3D as gf3
if modelling_tru==mlg2 or modelling_inv==mlg2:
    import Helmholtz_equation_FD.solver_2D as hs2d
if modelling_tru==mlg3 or modelling_inv==mlg3:
    import devito_solvers_TD.Helmholtz_2D.acoustic_solver as dhas

##########################################################################################################################

class inv_cc_amp:

    def __init__(self, rlocsx, rlocsy, signal, iterate, only1_iter, dobs, dobs_info):

        """
        rlocsx (type 'type 'numpy.ndarray'): x-coordinates of all receivers (in grid-point units)
        rlocsy (type 'type 'numpy.ndarray'): y-coordinates of all receivers (in grid-point units)
        signal (type 'instance'): object of class "SignalParameters" containing various signal characteristics of the data
        dobs (optional, type 'numpy.ndarray'): the data (REAL DATA ONLY)
        dobs_info (optional, type 'tuple'): Tuple containing the S/N ratio and actual (non-gridded) receiver locations (REAL DATA ONLY)
        """

        try:
            assert len(rlocsx)==len(rlocsy)
            assert np.all(rlocsx<ngpmb/2) and np.all(rlocsy<ngpmb/2)
            # all receivers must lie within the main box
        except AssertionError:
            raise SystemExit("Problem with receiver coordinates: must specify all (x,y) and all must lie within domain")

        # preliminaries - define class attributes
        self.nrecs = len(rlocsx)
        self.c = c_scal

        self.nom = signal.nsam
        self.deltat = signal.dt
        self.f0 = signal.cf
        # fl = signal.lf
        # fh = signal.hf
        self.lf = signal.lf
        self.hf = signal.hf
        altuk = signal.altukey

        self.t = signal.tt
        self.fhz = signal.fhz
        self.dom = signal.domega
        self.omega = signal.omega_rad
        self.nom_nneg = signal.n_nn_fsam

        npairs=int(self.nrecs*(self.nrecs-1)/2)

        if not (dobs is None):
        # real data case
            self.reald = True
            self.obscross = dobs
            self.obscross_aspec_temp = np.abs(np.fft.fft(dobs,axis=0))
        else:
        # synthetic data case
            self.reald = False
            # self.obscross = np.zeros((self.nom, self.nrecs, self.nrecs))#, dtype='complex')
            self.obscross_tensor = np.zeros((3, 3, self.nom, self.nrecs, self.nrecs), dtype='complex')
            if init_amp_scaling:
                self.obscross_aspec_temp = np.zeros(self.obscross_tensor[0,0,...].shape)

        try:
            assert self.reald == config.reald
        except AssertionError:
            raise SystemExit("Config paramter 'reald' inconsistent with input to core module. This will\
            mess up various settings and break the code. Please rectify.")

        compute_cc = {True: self.compute_cc_lathomo, False: self.compute_cc_lathet}

        self.dvar_pos = np.ones(npairs)
        self.dvar_neg = np.ones(npairs)

        self.rlocsx = rlocsx*dx
        self.rlocsy = rlocsy*dx

        # self.setup(f0, fl, fh, altuk, dobs_info)
        self.setup(dobs_info)

        self.num_mparams=self.basis.shape[0]
        self.distribs_inv=np.copy(self.distribs_start)
        self.allit_mc = []
        self.allit_misfit = []
        self.flit_indmis_p = []
        self.flit_indmis_n = []
        self.allit_synenv = []
        self.allit_syncross = []
        # variables with names ending in "_inv"  contain values for current (ulimately last) iteration only
        # variables with names starting with "allit_" are lists where each element corresponds to an iteration of the inversion.
        # variables with names starting with "flit_" are two-element lists, storing first (f) and last (l) iteration values only, of certain quantities.

        self.allit_mc.append(np.copy(self.mc_start))

        if __name__ == '__main__':
        	if self.nrecs<nrth:
        		self.skers=[] #range(npairs)

        itnum=0
        forced=False

        while iterate:

            self.iter = itnum
            # self.syncross = np.zeros((self.nom, self.nrecs, self.nrecs))#, dtype='complex')
            self.syncross_tensor = np.zeros((3, 3, self.nom, self.nrecs, self.nrecs), dtype='complex')
            # self.syncross_aspec_temp = np.zeros(self.obscross.shape)

            #*************** inversion related variables
            self.Gmat_pos=np.zeros((npairs,self.num_mparams))
            self.Gmat_neg=np.zeros((npairs,self.num_mparams))

            self.deltad_pos=np.zeros(npairs)
            self.deltad_neg=np.zeros(npairs)

            self.ngrad1_pos=np.empty(self.num_mparams); self.ngrad2_pos=np.empty(self.num_mparams)
            self.ngrad1_neg=np.empty(self.num_mparams); self.ngrad2_neg=np.empty(self.num_mparams)
            # there are two ways of computing the gradient of chi: with and without explicit use of
            # the G-matrix. In other words: using individual kernels or using the total misfit kernel.
            # I compute the gradient in both ways (hence subscripts 1, 2 on the variables) and ensure
            # they are equal, for confidence in the calculations.

            mfit_kern_pos = np.zeros((ngpmb, ngpmb))
            mfit_kern_neg = np.zeros((ngpmb, ngpmb))
            # mfit_kern -> misfit_kernel

            #*************** compute cross-correlations and make measurements
            if (not self.reald) and (self.iter==0):
            # SYNTHETIC TEST CASE, compute "observed data" synthetically
                compute_cc[lat_homo_tru]('obs')

            compute_cc[lat_homo_inv]('pre')
            self.make_measurement()

            print("Starting computation of source kernels for each receiver pair...")
            cp=0 # cp stands for count_pair
            for j in range(self.nrecs-1):
            	for i in range(j+1,self.nrecs):
            		print("...receivers ", i,j)
            		sker_p, sker_n = self.diffkernel(i,j)
            		# Computing individual source kernels (eq. 15)

            		# build the G-matrix
            		kb_prod = sker_p*self.basis
            		self.Gmat_pos[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2
            		kb_prod = sker_n*self.basis
            		self.Gmat_neg[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2

            		if __name__ == '__main__':
            			if self.nrecs<nrth and itnum==0:
            				self.skers.append(sker_p)

            		self.deltad_pos[cp] = np.log(self.obsamp_pos[i,j]/self.synamp_pos[i,j])
            		# print("obsamp_pos and synamp_pos for receivers ", i, j, self.obsamp_pos[i,j], self.synamp_pos[i,j])
            		# Computing misfit kernels, i.e. eq. 30 (positive branch)
            		mfit_kern_pos += sker_p * self.deltad_pos[cp]

            		self.deltad_neg[cp] = np.log(self.obsamp_neg[i,j]/self.synamp_neg[i,j])
            		# print("obsamp_neg and synamp_neg for receivers ", i, j, self.obsamp_neg[i,j], self.synamp_neg[i,j])
            		# Computing misfit kernels, i.e. eq. 30 (negative branch)
            		mfit_kern_neg += sker_n * self.deltad_neg[cp]

            		cp+=1

            #*********** things to do on first iteration
            if itnum==0:
            	if self.reald:
            	# complete the calculation of the data errors. NB: we consider two types of error.
            	# The first one (energy decay) is independent of the measurements and is already computed.
            	# The second (SNR) is defined relative to the measurements, so we must get the absolute values here.

            		dvar_snr_pos = np.square(self.esnrpd_ltpb * self.obsamp_pos)
            		dvar_snr_neg = np.square(np.transpose(self.esnrpd_ltpb) * self.obsamp_neg)

            		# combine different errors
            		dvar_pos = dvar_snr_pos #+ self.dvar_egy_ltpb
            		dvar_neg = dvar_snr_neg #+ np.transpose(self.dvar_egy_ltpb)

            		# finally, convert data variance from matrix-form (2D) to vector-form (1D)
            		dv_mat = {'p': dvar_pos, 'n': dvar_neg}
            		dv_vec = {'p': self.dvar_pos, 'n': self.dvar_neg}
            		for br in dv_mat:
            			start=0
            			for col in range(dv_mat[br].shape[1]):
            				ntc = dv_mat[br].shape[0] - col - 1
            				# ntc -> number(of pairs)_this_col
            				dv_vec[br][start:start+ntc] = dv_mat[br][col+1:,col]
            				start+=ntc

            			print("dv_vec: ", dv_vec[br])

            	# regardless of real or synthetic data, store the first-iteration values of certain quantities
            	self.mfit_kern_pos = mfit_kern_pos
            	self.mfit_kern_neg = mfit_kern_neg

            def record_flit():
            	self.flit_indmis_p.append(self.deltad_pos)
            	self.flit_indmis_n.append(self.deltad_neg)

             	# record inversion progress
            # wmp = self.deltad_pos * self.dvar_pos
            # wmn = self.deltad_neg * self.dvar_neg
            wmp = self.deltad_pos / np.sqrt(self.dvar_pos)
            wmn = self.deltad_neg / np.sqrt(self.dvar_neg)
            total_misfit = 0.5*(np.dot(wmp,wmp) + np.dot(wmn,wmn))
            if itnum==0:
            	record_flit()
            self.allit_misfit.append(total_misfit)
            self.allit_synenv.append(self.synenv)
            self.allit_syncross.append(self.syncross)

            if itnum==1:
            	record_flit()
            	if only1_iter:
            	# FORCED STOP FOR TESTING: last misfit stored will correspond to first updated model
            		forced=True; iterate=False


            if (itnum>0) and (not forced):
            # determine whether to terminate inversion or iterate further
            	mf_curr = self.allit_misfit[-1]
            	mf_prev = self.allit_misfit[-2]
            	pchange = 100*(mf_prev - mf_curr)/mf_prev
            	if (pchange>0 and pchange<5) or itnum>15:
            		iterate=False
            		#inversion terminated.
            		# store the individual misfits corresponding to the final iteration model
            		record_flit()

            if iterate:
            	#*********** do actual inversion (model update)
            	update_mod = self.inversion(mfit_kern_pos,mfit_kern_neg)
            	self.distribs_inv += update_mod
            	itnum +=1
            	print("END OF ITERATION %d" %(itnum))

        #*********************** End of loop over iterations *******************

	########################################################################################################################

    def inversion(self,mfk_pos,mfk_neg):

    	""" Performs inversion using a standard Gauss-Newton iterative scheme """

    	# NB: the data covariance matrix is assumed to be diagonal. Instead of storing and using the potentially HUGE
    	# diagonal matrix, we work with just the vector of data variances.

    	#**************************** fix the damping (model covarance matrix) *********************************
    	if not self.reald:
    	# in case of synthetic data, we can get away with a diagonal model covariance matrix (covariances = 0)
    		Dmat=np.identity(self.num_mparams)
    		CmInv = (gamma**2)*Dmat
    	else:
    	# in case of real data, we use a banded model covariance matrix (non-zero covariances)
    		self.Cm = np.zeros((self.num_mparams,self.num_mparams))
    		cord = 3
    		# cord -> correlation_distance
    		for r in range(self.Cm.shape[0]):
    			col = np.arange(float(self.Cm.shape[1]))
    			self.Cm[r,:] = ((1./gamma)**2)*np.exp(-0.5*(((r-col)/cord)**2))

    		CmInv = np.linalg.inv(self.Cm)

    		# to view the model covariance matrix (with iPython), use:
    		# x=np.arange(kcao.Cm.shape[0]); y=np.arange(kcao.Cm.shape[1])
    		# gx,gy=np.meshgrid(x,y)
    		# plt.pcolor(gx,gy,kcao.Cm)

    	#*************************************** End of damping ************************************************#

    	m_iter = self.allit_mc[-1]
    	m_prior = self.mc_start

    	G = {'p': self.Gmat_pos, 'n': self.Gmat_neg}
    	dd = {'p': self.deltad_pos, 'n': self.deltad_neg}
    	mfk = {'p': mfk_pos, 'n': mfk_neg}
    	ng1 = {'p': self.ngrad1_pos, 'n': self.ngrad1_neg}
    	ng2 = {'p': self.ngrad2_pos, 'n': self.ngrad2_neg}
    	dvi = {'p': 1./self.dvar_pos, 'n': 1./self.dvar_neg}
    	mod_update = np.zeros((ngpmb, ngpmb))
    	deltam = np.zeros((2,self.num_mparams))

    	for b,br in enumerate(G):
    	# br -> branch (positive or negative)

    		# compute basic (unweighted, undamped) gradient: method 1
    		kb_prod2 = mfk[br]*self.basis
    		ng1[br][:] = np.sum(kb_prod2, axis=(1,2)) * dx**2
    		# compute basic gradient: method 2
    		ng2[br] = np.matmul(G[br].T,dd[br])
    		if (not np.allclose(ng1[br],ng2[br],rtol=1e-03)):
    			#sys.exit("Quitting. Problem computing gradient.")
    			pass

    		# effect of weighting
    		Gt_CdInv = (G[br].T)*dvi[br]
    		ngrad = np.matmul(Gt_CdInv,dd[br])
    		# effect of damping
    		ngrad_use = ngrad - np.matmul(CmInv,(m_iter - m_prior))

    		# Hessian with weighting and damping
    		hess_apx = np.matmul(Gt_CdInv,G[br])
    		hess_use = hess_apx + CmInv

    		#********** solve the linear system for the model update
    		deltam[b,:] = np.linalg.solve(hess_use,ngrad_use)

    	# combine the results from the positive and negative branches
    	deltam_use = np.mean(deltam,axis=0)
    	#deltam_use = deltam[0,:]
    	self.allit_mc.append(m_iter + deltam_use)
    	mod_update = np.einsum('k,klm',deltam_use,self.basis)

    	return mod_update

	########################################################################################################################

    def setup(self, dinfo):

        if scalar:
            # DO NOT EDIT
            self.comp_src = 'z'
            cctc = 8
            # cctc -> cross-correlation_tensor_component(s)
            # the code is written so as to implement the vector/elastic (3-D) and scalar/acoustic (2-D) cases in a consistent manner,
            # so scalar quantities are treated as the 'z-component' of a 3-D cartesian system.
        elif elastic:
            self.comp_src = config.ccmt.src_dir
            cctc = config.ccmt.GTC[0]

        self.comp_p = tensor_rtz[cctc][0]
        self.comp_q = tensor_rtz[cctc][1]
        self.comp_scal = 'Z' # this is fixed

        tdur=self.nom*self.deltat
        # inverse frequency spacing for DFT of time series
        print("tdur as seen by h13 module: ", tdur)

        self.ft_fac = self.dom/(2*np.pi)*self.nom

        #************************************* build source characteristics ********************************************
        if not self.reald:
        # synthetic data case
            self.pss = config.sig_char.pow_spec_sources

        # *****************************************
        # OLD WAY of defining source spectrum
        	# if self.f0==2:
        	# 	a=0.3
        	# elif self.f0==0.3:
        	# 	a=16
        	# elif self.f0==0.1:
        	# 	a=64
        	# elif self.f0==0.05:
        	# 	a=640
        	# # dependence of parameter "a" -- which controls rate of exponential damping and hence shape of stf -- on peak
        	# # frequency is implemented in an adhoc fashion for the peak frequencies of interest when using this code. The
        	# # criterion behind the estimated values is to obtain a meaningful power spectrum -- one with zero DC power.
        	# # (In the time domain this corresponds to retaining a few cycles (~ 2-3) of the cosine wave before it is damped to 0.)
            #
        	# self.sourcetime = np.exp(-self.t**2/a) * np.cos(2*np.pi*self.f0*self.t)
        	# #self.pss = np.abs(np.fft.fft(self.sourcetime)*self.deltat)**2
        	# # ARJUN: why the multiplication by self.deltat above?
        	# self.pss = np.abs(np.fft.fft(self.sourcetime))**2
        	# # pss stands for power_spectrum_of_sources
        # END: OLD WAY of defining source spectrum
        # *****************************************

        else:
        # real data case
        	max_each_rp = np.max(self.obscross_aspec_temp, axis=0)
        	norm_obs_aspec = np.copy(self.obscross_aspec_temp)
        	with np.errstate(invalid='raise'):
        		try:
        			norm_obs_aspec /= max_each_rp
        		except FloatingPointError as e:
        			errargs=np.argwhere(max_each_rp==0)
        			if not np.all(errargs[:,0]<=errargs[:,1]):
        			# only the upper triangular part + main diagonal of "max_each_rp" should be 0
        				sys.exit("Problem normalizing the observed amplitude spectra (to their individual maxima)")

        	self.obs_aspec_mean = np.nanmean(norm_obs_aspec,axis=(1,2))
        	# NB: this spectrum is only useful for its shape. It is a DUMMY as far as amplitude is concerned.
        	dummy_egy_funcf = (self.obs_aspec_mean)**2/(self.nom)
        	dummy_pow = np.sum(dummy_egy_funcf,axis=0)

        	fhzp=self.fhz[self.fhz>=0]
        	fhzn=self.fhz[self.fhz<0]
        	# taking zero on the positive side ensures that both branches are of equal size, because remember that for
        	# even number of samples, the positive side is missing the Nyquist term.

        	#rvp=sst.skewnorm(a=-5,loc=0.55,scale=0.15)
        	#rvn=sst.skewnorm(a=5,loc=-0.55,scale=0.15)

        	rvp=sst.skewnorm(a=-3,loc=0.5,scale=0.13)
        	rvn=sst.skewnorm(a=3,loc=-0.5,scale=0.13)

        	self.pss = np.concatenate((rvp.pdf(fhzp),rvn.pdf(fhzn)))

        #************************************* end of source characteristics *******************************************

        if __name__ == '__main__':

            # fig_stf=plt.figure()
            # ax_stf=fig_stf.add_subplot(111)
            # ax_stf.plot(self.t,self.sourcetime)
            # ax_stf.set_xlabel('Time [s]')
            # ax_stf.set_title('Source time function')

            fig_ss=plt.figure()
            ax_ss=fig_ss.add_subplot(111)
            ax_ss.plot(np.fft.fftshift(self.fhz),np.fft.fftshift(self.pss))
            # ax_ss.set_xlim(0,1/(2*self.deltat))
            ax_ss.set_xlabel('Frequency [Hz]')
            ax_ss.set_title('Sources power spectrum')
        #***************************************************************************************************************

        self.ndouble=2*ngpmb-1
        ntot_omost = omost_fac*(ngpmb-1) + 1 # NB: should be equal to self.ndouble if omost_fac==2

        hlbox_omost = omost_fac*hlbox_outer/2.0

        # grid points of main box
        # x = np.linspace(-hlbox_outer/2,hlbox_outer/2,ngpmb)
        # y = np.copy(x)

        # grid points of outer box
        # x2=np.linspace(-hlbox_outer,hlbox_outer,self.ndouble)
        # y2=np.copy(x2)

        # grid points of outer-most box (which is separate from the 'outer box' only when omost_fac>2)
        x3=np.linspace(-hlbox_omost,hlbox_omost,ntot_omost)
        y3=np.copy(x3)

        # grids for plotting
        # self.gx, self.gy = dg.gx, dg.gy
        # self.gx2, self.gy2 = dg.gx2, dg.gy2
        self.gx3, self.gy3 = np.meshgrid(x3,y3)

        self.dist_rp=np.zeros((self.nrecs,self.nrecs))
        for j in range(self.nrecs):
            for i in range(self.nrecs):
                self.dist_rp[i,j] = np.sqrt( (self.rlocsx[i]-self.rlocsx[j])**2 + (self.rlocsy[i]-self.rlocsy[j])**2 )

        dist_all = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]
        self.alldist = dist_all[np.argsort(dist_all)]
        # a sorted 1-D array of receiver-pair distances

        print("Computing distances from origin..")
        r = np.sqrt(self.gx3**2 + self.gy3**2)
        r = config.add_epsilon(r)
        # this is done so that r does not contain any zeros; to prevent the Hankel function
        # from blowing up at the origin (r=0)

        if lat_homo_tru or lat_homo_inv:
            """ Laterally HOMOGENEOUS structure model(s) """

            print("Computing Green functions for point source at origin..")
            self.Green = np.zeros((3,3,self.ndouble,self.ndouble,self.nom_nneg), dtype='complex')
            self.Green_obs = np.zeros((3,3,ntot_omost,ntot_omost,self.nom_nneg), dtype='complex')
            # "Green_obs" may be larger (spatially) than "Green", i.e. the 'observed data' may be computed using
            # a domain that is larger than the inverse modelling domain.

            # ------ calling external modules if required ------

            if (lat_homo_tru and modelling_tru==mlg3) or (lat_homo_inv and modelling_inv==mlg3):

                dom_orig = [x3[0], y3[0]]      # MUST be in km
                gs_xy = [dx*1e3, dx*1e3]        # MUST be metres
                ngp_xy = [x3.size, y3.size]     # number of grid points [x,y]
                vp = c_scal * np.ones(ngp_xy)   # MUST be in km/s
                par_dvto = {'t0': 0.,
                            'tn': 49600.,           # Simulation length (ms)
                            'f0': self.f0*1e-3}     # Source peak frequency (kHz)

                # source at centre
                src_locs = np.empty((1, 2)) #, dtype=np.float32)
                src_locs[:, 0] = (ngp_xy[0]-1)*0.5*gs_xy[0] + dom_orig[0]*1e3
                src_locs[:, 1] = (ngp_xy[1]-1)*0.5*gs_xy[0] + dom_orig[1]*1e3

                print("Source location for devito solver: ", src_locs)

                # receivers not required in this case (we work with the wavefield everywhere) but included for completeness
                rec_locs = np.empty((self.nrecs, 2))
                rec_locs[:, 0] = self.rlocsx*1e3
                rec_locs[:, 1] = self.rlocsy*1e3

                dhaso = dhas.sim_int_sources(dom_orig, gs_xy, vp)
                dhaso_t = np.arange(par_dvto['t0'], par_dvto['tn']+dhaso.dt, dhaso.dt)
                stf_dvto = ss.resample(self.sourcetime, num=dhaso_t.size)
                srcdata = stf_dvto.reshape(dhaso_t.size, 1)

                dhaso.solve(par_dvto, srcdata, src_locs, rec_locs, True)
                dhaso.resample(self.deltat*1e3) # convert s to ms
                dhaso.get_Green_FD(dhaso.rec_data_resamp, dhaso.wav_fld_resamp)
                dhaso_freqs=dhaso.freqs*1e3     # convert kHz to Hz

                print(dhaso.rec_data_resamp.time_range.step)
                diff_fs = np.abs(dhaso_freqs[:-1]-self.fhz[:self.nom_nneg])
                assert np.all(diff_fs < (self.fhz[1]/1e6))
                # making sure that dhaso_freqs and self.fhz are practically equal

            # ------ END: calling external modules if required ------

            if omost_fac==2:
                sx=0
                ex=self.ndouble
                sy=sx
                ey=ex
            elif omost_fac>2:
                sx, ex, sy, ey = config.box_indices_largerbox_src([0], [0], self.ndouble)

            for i in range(1,self.nom_nneg):
                # compute Green's function only at those frequencies where the source spectrum is non-zero
                thresh = (0.01*np.amax(self.pss))/100
                if self.pss[i] < thresh:
                    print("(ignoring FREQUENCY %f Hz)" %(self.fhz[i]))
                else:
                    print("...FREQUENCY %f Hz" %(self.fhz[i]))

                    # ------ using external modules if required ------

                    if (lat_homo_tru and modelling_tru==mlg2) or (lat_homo_inv and modelling_inv==mlg2):
                        hs2do = hs2d.single_freq(x3*1e3,y3*1e3,dx*1e3,dx*1e3,self.fhz[i],self.c,[0],[0])
                        num_sol_acou = hs2do.u2_2D
                    elif (lat_homo_tru and modelling_tru==mlg3) or (lat_homo_inv and modelling_inv==mlg3):
                        idx=np.searchsorted(dhaso_freqs, self.fhz[i])
                        assert idx==i
                        nbl = dhaso.vel_model.nbl
                        num_sol_acou = np.conj(dhaso.GF_FD[idx,nbl:-nbl,nbl:-nbl])
                        amp_fac = gs_xy[0]*gs_xy[1]
                        num_sol_acou /= amp_fac
                        # amplitude correction
                    else:
                        num_sol_acou = None

                    if modelling_tru==mlg1 or modelling_inv==mlg1:
                        try:
                            assert elastic, "Elastic modelling chosen in config file but corresponding input not provided"
                        except AssertionError:
                            raise SystemExit("Please supply input for (1-D) elastic modelling or choose scalar modelling")
                        period = 1./self.fhz[i] # seconds
                        swgmfo = gf3.Green_SW_monofreq(period, Zpts_all, discon_mod, nzmax)
                        try:
                            swgmfo.prepare_egn(efile_ray, dfile_ray, 'ray')
                            # if love:
                            #     swgmfo.prepare_egn(efile_lov, dfile_lov, 'lov')
                        except Exception as e:
                            print(e)
                            raise SystemExit("Aborted: cannot compute Green's functions.")
                        else:
                            swgmfo.G_cartesian_grid(self.gx3, self.gy3, r, 2)
                            semian_sol_elas = swgmfo.Gtensor
                    else:
                        try:
                            assert scalar, "Scalar modelling chosen in config file but (1-D) elastic input is provided"
                        except AssertionError:
                            raise SystemExit("Elastic modelling (1-D) input is provided but purely scalar modelling chosen - can lead to errors.\
                            Please check modelling settings in config file.")
                        semian_sol_elas = None

                    # ------ END: using external modules if required ------

                    solution = {'anal_scal_0D': ssp.hankel1(0,self.omega[i]*r/self.c) * 1j * 0.25,
                                'anal_elas_1D': semian_sol_elas,
                                'num_scal_2D_FD': num_sol_acou,
                                'num_scal_2D_TD': num_sol_acou}

                    if elastic:
                        ind0=slice(self.Green.shape[0])
                        ind1=slice(self.Green.shape[1])
                    elif scalar:
                        ind0=2
                        ind1=2

                    self.Green_obs[ind0,ind1,:,:,i] = solution[modelling_tru]
                    self.Green[ind0,ind1,:,:,i] = solution[modelling_inv][sy:ey,sx:ex]

        if not (lat_homo_tru and lat_homo_inv):
            """ Laterally HETEROGENEOUS structure model(s) """

            try:
                assert ('num' in modelling_tru) or ('num' in modelling_inv)
                # modelling type has to be numerical
            except AssertionError:
                raise SystemExit("Laterally heterogeneous structure incompatible with analytical modelling.\
                Please check modelling settings in config file.")

            dom_orig = [x[0], y[0]]         # MUST be in km
            gs_xy = [dx*1e3, dx*1e3]        # MUST be metres
            ngp_xy = [x.size, y.size]       # number of grid points [x,y]
            vp = c_scal * np.ones(ngp_xy)   # MUST be in km/s
            self.par_dvto = {'t0': 0.,
                            'tn': 49600.,           # Simulation length (ms)
                            'f0': self.f0*1e-3}     # Source peak frequency (kHz)

            self.dhaso = dhas.sim_int_sources(dom_orig, gs_xy, vp)
            dhaso_t = np.arange(self.par_dvto['t0'], self.par_dvto['tn']+self.dhaso.dt, self.dhaso.dt)
            self.pss_dvto = ss.resample(self.pss, num=dhaso_t.size)

            self.amp_fac_dvto = gs_xy[0]*gs_xy[1]

            # sources potentially everywhere in domain (distributed source)
            self.src_locs_dvto = np.empty((np.prod(ngp_xy),2))
            self.src_locs_dvto[:, 0] = (self.gx.flatten())*1e3
            self.src_locs_dvto[:, 1] = (self.gy.flatten())*1e3

            self.rec_locs_dvto = np.empty((self.nrecs, 2))
            self.rec_locs_dvto[:, 0] = self.rlocsx*1e3
            self.rec_locs_dvto[:, 1] = self.rlocsy*1e3

            # self.dhasdo = dhas.sim_dist_sources(dom_orig, gs_xy, vp)

        #******************************* source distributions and observation errors ********************************************

        sdist_type = {'mg': u1.somod.mult_gauss, 'rg': u1.somod.ringg, 'rgr': u1.somod.rgring}

        self.distribs_start = np.zeros((ngpmb,ngpmb))
        # used to generate the synthetics for inversion; this source distribution IS involved in computing source kernels

        mag1=1

        nbasis = {'rg': int(360/mod_specs['rg']['as']),
                    'mg': len(mod_specs['mg']['r0'])}

        # setup to use basis functions in case of 'rg'
        if param_mod_inv=='rg' or param_mod_tru=='rg':
            astep = mod_specs['rg']['as']
            alltheta_deg=np.arange(0,360,astep)
            alltheta=alltheta_deg*np.pi/180
            assert alltheta.size==nbasis['rg']

        # setup to use basis functions in case of 'mg'
        allcen = lambda specs: specs['r0']

        basis_loc = {'rg': alltheta, 'mg': allcen(mod_specs['mg'])}

        init_basis = lambda m,n: np.zeros((m,n,n))
        init_model = lambda n: np.zeros((n,n))

        self.mc_start = np.ones(nbasis[param_mod_inv])
        # mc -> model_coefficients
        self.basis = init_basis(nbasis[param_mod_inv],ngpmb)

        self.mc_start *= mag1

        for k,locator in enumerate(basis_loc[param_mod_inv]):
            self.basis[k,:,:] = sdist_type[param_mod_inv](locator, mod_specs[param_mod_inv])
            self.distribs_start += self.mc_start[k]*self.basis[k,:,:]

        # ----------------------------------------------------------------------
        def get_mct_rg(specs):
            ans = self.mc_true * mag1
            t1=specs['t1']
            t2=specs['t2']
            mag2=specs['pert']
            nperts=specs['np']
            for p in range(nperts):
                s1=np.argwhere(alltheta_deg >= t1[p])
                s2=np.argwhere(alltheta_deg <= t2[p])
                relind=np.intersect1d(s1,s2)
                if len(relind)==0:
                    relind=np.union1d(s1,s2)
                ans[relind]=mag2[p]+mag1
            return ans

        get_mct_mg = lambda specs: mag1*np.array(specs['mag'])
        # ----------------------------------------------------------------------

        get_mc_true = {'rg': get_mct_rg, 'mg': get_mct_mg}

        if not self.reald:
        # SYNTHETIC DATA CASE - generate the 'true' model
            self.mc_true = np.ones(nbasis[param_mod_tru])
            if omost_fac==2:
                basis_true = init_basis(nbasis[param_mod_tru],ngpmb)
                self.distribs_true = init_model(ngpmb)
                # used to generate the synthetic "data" in the absence of real data
                # this source distribution is NOT involved in computing source kernels
            elif omost_fac==3:
                basis_true = init_basis(nbasis[param_mod_tru],self.ndouble)
                self.distribs_true = init_model(self.ndouble)

            self.mc_true = get_mc_true[param_mod_tru](mod_specs[param_mod_tru])

            for k,locator in enumerate(basis_loc[param_mod_tru]):
                basis_true[k,:,:] = sdist_type[param_mod_tru](locator, mod_specs[param_mod_tru], omost_fac)
                self.distribs_true += self.mc_true[k]*basis_true[k,:,:]

        else:
        # REAL DATA CASE - compute observation errors

            #********* Errors Part 1: error due to SNR
            snr = dinfo[1]

            self.esnrpd_ltpb = np.zeros((self.dist_rp.shape))
            # esnrpd -> error(due to)_SNR_(as a)_percentage_(of)_data
            # ltpb -> lower_triangle_positive_branch
            # (it is implied that the upper triangle of the matrix is for the negative branch)

            self.esnrpd_ltpb[np.where(snr<2)]=0.8
            self.esnrpd_ltpb[np.where((snr>2) & (snr<3))]=0.5
            self.esnrpd_ltpb[np.where(snr>3)]=0.05

            #********* Errors Part 2: error due to energy decay with distance

            delA = dinfo[2]

            #********************************************************************************************************************
            # NB: uncertainties in the observations contained in dinfo need to be corrected, because the measurement for
            # the kernels involves cc energies computed in a certain window only, whereas the curve fitting above is done using
            # the energy of the entire cc branch. This correction can be made using the waveform's S/N ratio, which indirectly
            # provides a measure of the contribution of the window of interest, to the total energy of the waveform (branch).
            #********************************************************************************************************************

            # refine the error so it applies to the measurement window only
            nsr = 1./snr
            ScT = 1./(1+nsr) # 1./np.sqrt(1+nsr)
            # ScT -> signal_contribution_to_total (energy)
            delA *= ScT

            # convert to variance
            self.dvar_egy_ltpb = np.square(delA)

            # ********* Errors Part 3: position error due to relocation of receivers to grid points
            # origdist_rp = dinfo[0]
            # deltapos = np.square(origdist_rp - self.dist_rp)

        print("Completed initial setup...")

	########################################################################################################################

    def compute_cc_lathomo(self, dat_type):

        """ Algorithm using point sources only (invokes reciprocity at both receivers) """

        print("Computing cross-correlations...")

        # self.mod_spa_int_temp = np.zeros((self.nom_nneg-1, self.nrecs, self.nrecs))

        # account for possible asymmetry in frequency samples (happens when self.nom is even)
        fhzp = len(self.fhz[self.fhz>0])
        fhzn = len(self.fhz[self.fhz<0])
        ssna = abs(fhzn-fhzp)
        # ssna stands for samples_to_skip_due_to_nyquist_asymmetry
        print("SSNA: ", ssna)
        # print "Frequency samples are: ", self.fhz

        nxst, nxfin, nyst, nyfin = config.box_indices_largerbox_src(self.rlocsx, self.rlocsy)
        if omost_fac==2:
            nxst_obs = nxst
            nyst_obs = nyst
            nxfin_obs = nxfin
            nyfin_obs = nyfin
        elif omost_fac>2:
            # true model has size (length of side) = self.ndouble
            nxst_obs, nxfin_obs, nyst_obs, nyfin_obs = config.box_indices_largerbox_src(self.rlocsx, self.rlocsy, self.ndouble)

        for k in range(self.nrecs-1):
            for j in range(k+1,self.nrecs):
                # compute eq. 11 of Hanasoge (2013)
                # print(nyst[k],nyfin[k],nxst[k],nxfin[k])
                # print(nyst[j],nyfin[j],nxst[j],nxfin[j])
                # print(self.nom_nneg)

                if dat_type=='pre':
                    print("...cc for receivers ", j, k)
                    for p in range(scs,3):
                        for q in range(scs,3):
                            print("Component %d-%d" %(p,q))
                            Grec_j = self.Green[dc_xyz[self.comp_src],p,nyst[j]:nyfin[j],nxst[j]:nxfin[j],:]
                            Grec_k = self.Green[dc_xyz[self.comp_src],q,nyst[k]:nyfin[k],nxst[k]:nxfin[k],:]

                            f_inv = np.conj(Grec_k[...,1:self.nom_nneg]) * Grec_j[...,1:self.nom_nneg]

                            fsyn = np.transpose(f_inv,[2,0,1]) * self.distribs_inv
                            spa_int = np.sum(fsyn, axis=(-1,-2)) * dx**2

                            # compute the cross-correlations for positive frequencies
                            # Frequency-domain symmetry: calculations needed only for half the total number of frequencies.
                            self.syncross_tensor[p,q,1:self.nom_nneg,j,k] = spa_int * self.pss[1:self.nom_nneg]
                            # self.mod_spa_int_temp[:,j,k] = np.abs(spa_int)

                            # Negative frequency coefficients are complex conjugates of flipped positive coefficients.
                            self.syncross_tensor[p,q,self.nom_nneg+ssna:,j,k] = np.flipud(np.conj(self.syncross_tensor[p,q,1:self.nom_nneg,j,k]))
                            # 22 June 2018: BEWARE, the negative Nyquist term gets left out in case ssna>0, i.e. in case self.nom is even.
                            # the same holds for obscross too.
                            # this does matter of course, but it appears to make a very minor difference to the event kernels
                            # so I am leaving it for the time being.

                    # take care of constant factors
                    self.syncross_tensor[...,j,k] *= self.ft_fac

                if dat_type=='obs':
                    print("...cc (obscross) for receivers ", j, k)

                    for p in range(sco,3):
                        for q in range(sco,3):
                            print("Component %d-%d" %(p,q))
                            Grec_j = self.Green_obs[dc_xyz[self.comp_src],p,nyst_obs[j]:nyfin_obs[j],nxst_obs[j]:nxfin_obs[j],:]
                            Grec_k = self.Green_obs[dc_xyz[self.comp_src],q,nyst_obs[k]:nyfin_obs[k],nxst_obs[k]:nxfin_obs[k],:]
                            f_true = np.conj(Grec_k[...,1:self.nom_nneg]) * Grec_j[...,1:self.nom_nneg]
                            fobs = np.transpose(f_true,[2,0,1]) * self.distribs_true
                            self.obscross_tensor[p,q,1:self.nom_nneg,j,k] = np.sum(fobs, axis=(-1,-2)) * self.pss[1:self.nom_nneg] * dx**2
                            self.obscross_tensor[p,q,self.nom_nneg+ssna:,j,k] = np.flipud(np.conj(self.obscross_tensor[p,q,1:self.nom_nneg,j,k]))
                    self.obscross_tensor[...,j,k] *= self.ft_fac

        #***** End of double-loop over receiver pairs *****
        # APPLY ROTATION FROM XYZ TO RTZ COORDS
        # *************************************************

        icmp = lambda m,u: dc_rtz[getattr(self, u[sORe(m)])]

        # select desired cross-correlation component
        if dat_type=='pre':
            syncross_FD = self.syncross_tensor[icmp(modelling_inv, use_p), icmp(modelling_inv, use_q),...]
        elif dat_type=='obs':
            obscross_FD = self.obscross_tensor[icmp(modelling_tru, use_p), icmp(modelling_tru, use_q),...]
            self.obscross_aspec_temp = np.abs(obscross_FD)

        if self.iter==0 and dat_type=='pre' and init_amp_scaling:

            need_to_scale=True
            da = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]

            one_by_r = lambda x,k: k/x

            def energy_versus_dist(cc_aspec, dummy_sig):

                ccegy_funcf = np.square(cc_aspec)
                # ccegy_funcf -> cc_power_as_a_function_of_frequency
                cc_egy = np.sum(ccegy_funcf,axis=0)/self.nom
                egy_flat = cc_egy[np.nonzero(np.tril(self.dist_rp))]
                # the matrix is symmetric so it suffices to consider only its lower triangular part
                energies = egy_flat[np.argsort(da)]

                try:
                    popt, pcov = sop.curve_fit(one_by_r, self.alldist, energies, sigma=dummy_sig, absolute_sigma=False)
                except NameError:
                    # sig_dummy does not exist in case of synthetic data
                    popt, pcov = sop.curve_fit(one_by_r, self.alldist, energies)

                ef = popt[0]/self.alldist

                return energies, ef

            # ************ compute observed data energies
            if self.reald:
            # in the real data case, treat 'very short' distances with caution
                nf_dist = 0.5*self.c/self.hf
                # nf_dist -> near_field_distance. Using a very crude estimate: half the shortest wavelength in the data
                sd_ind=np.argwhere(self.alldist<nf_dist)
                # sd_ind -> short_distance_indices
                sig_dummy = np.ones(self.alldist.size)
                sig_dummy[sd_ind] = 5
                # NB: sig_dummy - deliberately called "dummy" - contains basically the relative weights for the data points, NOT
                # the actual standard deviations. This is reflected in the argument "absolute_sigma=False" to scipy's curve fit.
            else:
                sig_dummy = None

            self.egy_obs, self.oef = energy_versus_dist(self.obscross_aspec_temp, sig_dummy)
            # oef -> observed_energy_fitted

            while need_to_scale:

                # ************ compute initial synthetic data energies
                self.syncross_aspec_temp = np.abs(syncross_FD)
                self.egy_syn, self.sef = energy_versus_dist(self.syncross_aspec_temp, None)
                # sef -> synthetic_energy_fitted

                # ************ compare observed and initial synthetic energies
                print("egy_obs: ")
                print(self.egy_obs)
                print("egy_syn: ")
                print(self.egy_syn)

                esf = np.mean(self.oef/self.sef)
                # esf -> energy_scale_factor

                if esf > 0.9 and esf < 1.1:
                    print("esf is %e, scaling of initial synthetics completed." %(esf))
                    need_to_scale=False
                else:
                    print("esf is %e, MULTIPLYING self.pss by %e" %(esf,np.sqrt(esf)))
                    syncross_FD *= np.sqrt(esf)
                    self.pss *= np.sqrt(esf) # for future iterations

            # ***** End of while loop for amplitude scaling *****

        else:
            # Subsequent iterations OR no amplitude scaling
            pass

        # convert to time domain cross-correlations
        if dat_type=='pre':
            self.syncross = np.fft.fftshift(np.fft.ifft(syncross_FD, axis=0).real, axes=(0,))
        elif dat_type=='obs':
            self.obscross = np.fft.fftshift(np.fft.ifft(obscross_FD, axis=0).real, axes=(0,))

        for k in range(self.nrecs):
        	# [k,j] cross-correlation same as flipped [j,k]
            if dat_type=='pre':
            	self.syncross[:,k,k+1:]=np.flipud(self.syncross[:,k+1:,k])
            elif dat_type=='obs':
            	self.obscross[:,k,k+1:]=np.flipud(self.obscross[:,k+1:,k])

        if dat_type=='pre':
            self.synenv=np.abs(ss.hilbert(self.syncross, axis=0))
        elif dat_type=='obs':
            self.obsenv=np.abs(ss.hilbert(self.obscross, axis=0))

        # if self.iter==0:
        # 	self.obscross_aspec_temp = np.abs(np.fft.fft(self.obscross,axis=0))
        	# for the synthetic data case, this MAY BE generated here for the FIRST time; for the real data case
        	# this is a recalculation but now the upper triangular part of the matrix is also filled in.

    #######################################################################################################################

    def compute_cc_lathet(self, dat_type):

        """ Hybrid algorithm using a point source-distributed source combination
            (reciprocity invoked at one of the receivers only)
        """

        stf_dvto = ss.resample(self.sourcetime, num=self.pss_dvto.size)
        pt_src = stf_dvto.reshape(self.pss_dvto.size, 1)

        # ------------------------------- TRIAL --------------------------------

        for k in range(self.nrecs-1):
            for j in range(k+1,self.nrecs):

                if dat_type=='pre':
                    # print("...cc for receivers ", j, k)
                    for p in range(scs,3):
                        for q in range(scs,3):
                            # print("Component %d-%d" %(p,q))
                            pass

                if dat_type=='obs':
                    print("...cc (obscross) for receivers ", j, k)
                    for p in range(sco,3):
                        for q in range(sco,3):
                            print("Component %d-%d" %(p,q))

                            src_loc = self.rec_locs_dvto[k,:].reshape(1,2)
                            # print("Source location for devito solver (point source): ", src_loc)
                            self.dhaso.solve(self.par_dvto, pt_src, src_loc, self.rec_locs_dvto, True)
                            self.dhaso.resample(self.deltat*1e3)
                            self.dhaso.get_Green_FD(self.dhaso.rec_data_resamp, self.dhaso.wav_fld_resamp)
                            # self.dhaso.get_Green_FD(self.dhaso.rec_data, self.dhaso.wav_fld)
                            nbl = self.dhaso.vel_model.nbl
                            GF_k = self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl]/self.amp_fac_dvto

                            src_loc = self.rec_locs_dvto[j,:].reshape(1,2)
                            self.dhaso.solve(self.par_dvto, pt_src, src_loc, self.rec_locs_dvto, True)
                            self.dhaso.resample(self.deltat*1e3)
                            self.dhaso.get_Green_FD(self.dhaso.rec_data_resamp, self.dhaso.wav_fld_resamp)
                            # self.dhaso.get_Green_FD(self.dhaso.rec_data, self.dhaso.wav_fld)
                            GF_j = self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl]/self.amp_fac_dvto

                            f_true = np.conj(GF_k) * GF_j
                            print("Shape of f_true: ", f_true.shape)
                            fobs = f_true * self.distribs_true
                            spa_int = np.sum(fobs, axis=(-1,-2)) * dx**2

                            spa_int_resamp = ss.resample(spa_int, num=126)

                            self.obscross_tensor[p,q,:,j,k] = np.fft.fftshift(np.fft.irfft(spa_int * self.pss[:self.nom_nneg+1]))
                            # self.obscross_tensor[p,q,:,j,k] = np.fft.fftshift(np.fft.irfft(spa_int_resamp * self.pss[:self.nom_nneg+1]))

                    self.obscross_tensor[...,j,k] *= self.ft_fac

        if dat_type=='pre':
            self.syncross = self.syncross_tensor[2,2,...].real
        if dat_type=='obs':
            self.obscross = self.obscross_tensor[2,2,...].real

        for k in range(self.nrecs):
        	# [k,j] cross-correlation same as flipped [j,k]
            if dat_type=='pre':
            	self.syncross[:,k,k+1:]=np.flipud(self.syncross[:,k+1:,k])
            elif dat_type=='obs':
            	self.obscross[:,k,k+1:]=np.flipud(self.obscross[:,k+1:,k])

        # ------------------------- END OF TRIAL --------------------------------

        # if dat_type=='pre':
        #     self.syncross_tensor = self.syncross_tensor.real
        #     # change cc-arrays to real-valued, since a TD solver is used
        # elif dat_type=='obs':
        #     self.obscross_tensor = self.obscross_tensor.real
        #
        # for k in range(self.nrecs-1):
        #     if dat_type=='obs':
        #         print("...cc (obscross) for master receiver ", k)
        #
        #     # STEP 1: GF for point source at receiver
        #     src_loc = self.rec_locs_dvto[k,:].reshape(1,2)
        #     print("Source location for devito solver (point source): ", src_loc)
        #     self.dhaso.solve(self.par_dvto, pt_src, src_loc, self.rec_locs_dvto, True)
        #     self.dhaso.get_Green_FD(self.dhaso.rec_data, self.dhaso.wav_fld)
        #     nbl = self.dhaso.vel_model.nbl
        #     # GF_this_src = self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl]/self.amp_fac_dvto
        #     GF_this_src = np.transpose(self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl],[0,2,1])/self.amp_fac_dvto
        #
        #     # STEP 2: compute distributed source
        #     gstars = GF_this_src * self.distribs_true
        #     dist_src_FD = self.pss_dvto[:gstars.shape[0],None,None] * gstars
        #     dist_src_TD = np.fft.irfft(dist_src_FD, axis=0)
        #
        #     # STEP 3: solve with distributed source
        #     print("Shape of distributed src before reshaping: ", dist_src_TD.shape)
        #     dist_src = dist_src_TD.reshape(self.pss_dvto.size, self.src_locs_dvto.shape[0])
        #     print("Shape of distributed src: ", dist_src.shape)
        #     # print("Source location for devito solver (distributed source): ", self.src_locs_dvto)
        #     self.dhaso.solve(self.par_dvto, dist_src, self.src_locs_dvto, self.rec_locs_dvto, False)
        #
        #     self.dhaso.resample(self.deltat*1e3)
        #     print("Shape of dhaso.data: ", self.dhaso.rec_data_resamp.shape)
        #     self.obscross_tensor[2,2,:,k,:] = np.fft.fftshift(self.dhaso.rec_data_resamp.data/self.amp_fac_dvto) * dx**2
        #     # self.obscross_tensor[2,2,:,k,:] = np.fft.fftshift(self.dhaso.rec_data.data[:250,:]/self.amp_fac_dvto) * dx**2
        #
        # self.obscross_tensor *= self.ft_fac
        #
        # if dat_type=='pre':
        #     self.syncross = self.syncross_tensor[2,2,...]
        # if dat_type=='obs':
        #     self.obscross = self.obscross_tensor[2,2,...]

    #######################################################################################################################

    def make_measurement(self):
    	# from misfit.m

    	print("In function make_measurement...")

    	self.weightpos = np.zeros((self.nom, self.nrecs, self.nrecs))
    	self.weightneg = np.zeros((self.nom, self.nrecs, self.nrecs))
    	self.synamp_pos = np.zeros((self.nrecs, self.nrecs))
    	self.synamp_neg = np.zeros((self.nrecs, self.nrecs))
    	self.obsamp_pos = np.zeros((self.nrecs, self.nrecs))
    	self.obsamp_neg = np.zeros((self.nrecs, self.nrecs))

    	initscal = np.zeros((self.nrecs, self.nrecs))

    	self.negl = np.zeros((self.nrecs, self.nrecs), dtype='int')
    	self.negr = np.zeros((self.nrecs, self.nrecs), dtype='int')
    	self.posl = np.zeros((self.nrecs, self.nrecs), dtype='int')
    	self.posr = np.zeros((self.nrecs, self.nrecs), dtype='int')

    	lefw = -4.0 #-1.0 #-0.25
    	rigw = +4.0 #1.0 #+0.25

    	# cslow = 1.2 #self.c - 1
    	# cfast = 6.0 #self.c + 5

    	for k in range(self.nrecs):
    		for j in np.delete(np.arange(self.nrecs),k):

    			if not self.reald:
    			# SYNTHETIC DATA CASE
    				# Simple windows suitable for synthetic data:
    				# 1. Entire cross-correlation - [0:self.nom]
    				# 2. Entire negative branch - [0:index of (sample 0)]
    				# 3. Entire positive branch - [1 + index of (sample 0):self.nom]

    				is0 = np.searchsorted(self.t,0)
    				self.negl[j,k] = 0
    				self.negr[j,k] = is0
    				self.posl[j,k] = is0 + 1
    				self.posr[j,k] = self.nom
    			else:
    			# REAL DATA CASE
    				lef = max(0,self.dist_rp[j,k]/self.c + lefw) # left boundary of window (seconds)
    				rig = self.dist_rp[j,k]/self.c + rigw # right boundary of window (seconds)

    				#lef = self.dist_rp[j,k]/cfast # left boundary of window (seconds)
    				#rig = self.dist_rp[j,k]/cslow # right boundary of window (seconds)

    				self.negl[j,k] = np.searchsorted(self.t,-rig)
    				self.negr[j,k] = np.searchsorted(self.t,-lef)
    				self.posl[j,k] = np.searchsorted(self.t,lef)
    				self.posr[j,k] = np.searchsorted(self.t,rig)

    			# the chosen windows (positive & negative side) should be of non-zero length, otherwise
    			# the windowed cross-correlation energy, which divides the weight function, will be 0.
    			# The windows can be zero-length if the arrival time for given station pair lies outside
    			# the modelled time range (depending on wavespeed obviously).

    			if self.negr[j,k]==0 or self.posl[j,k]==self.nom:
    				print("Problem with stations ", j, k)
    				sys.exit("Aborted. The chosen window for computing cross-corrrelation energy \
    					 lies outside the modelled time range")

    			# print("Negative side window indices: ", self.negl[j,k], self.negr[j,k])
    			# print("Positive side window indices: ", self.posl[j,k], self.posr[j,k])

    			# now make the measurements

    			print("making measurement for receivers ", j,k)

    			self.weightpos[self.posl[j,k]:self.posr[j,k], j, k] = self.syncross[self.posl[j,k]:self.posr[j,k], j, k]
    			self.weightneg[self.negl[j,k]:self.negr[j,k], j, k] = self.syncross[self.negl[j,k]:self.negr[j,k], j, k]

    			self.synamp_pos[j,k] = np.sqrt(np.sum(self.weightpos[:,j,k]**2))#*self.deltat)
    			#  Computing eq. 24 (numerator only), positive branch
    			self.synamp_neg[j,k] = np.sqrt(np.sum(self.weightneg[:,j,k]**2))#*self.deltat)
    			#  computing eq. 24 (numerator only), negative branch

    			self.obsamp_pos[j,k] = np.sqrt(np.sum(self.obscross[self.posl[j,k]:self.posr[j,k],j,k]**2))#*self.deltat)
    			self.obsamp_neg[j,k] = np.sqrt(np.sum(self.obscross[self.negl[j,k]:self.negr[j,k],j,k]**2))#*self.deltat)

    			with np.errstate(invalid='raise'):
    				try:
    					self.weightpos[:,j,k] /= self.synamp_pos[j,k]**2
    					self.weightneg[:,j,k] /= self.synamp_neg[j,k]**2
    				except FloatingPointError as e :
    					# this should never happen, none of the non-diagonal elements of self.synamp_pos or self.synamp_neg should be zero
    					errargs_p=np.argwhere(self.synamp_pos==0)
    					errargs_n=np.argwhere(self.synamp_neg==0)
    					if not np.all(errargs_p[:,0]==errargs_p[:,1]) or not np.all(errargs_n[:,0]==errargs_n[:,1]) :
    					# some non-diagonal elements of self.synamp_pos or self.synamp_neg or both are somehow zero
    						print("RED FLAG!!!: ", e) #, errargs_p, errargs_n)
    						print(self.syncross[self.posl[j,k]:self.posr[j,k], j, k])
    						# sys.exit("Problem with non-diagonal elements of measurement matrices")

    #######################################################################################################################

    def diffkernel(self, alpha, beta):
    # Computing source kernels for positive and negative branches

        # ARJUN: multiplication by self.deltat is only required here if it is also used in computation of synamp, obsamp
        ccpos = np.fft.fft(np.fft.ifftshift(self.weightpos[:,alpha,beta]))
        ccneg = np.fft.fft(np.fft.ifftshift(self.weightneg[:,alpha,beta]))

        bsx, bex, bsy, bey = config.box_indices_largerbox_src( [self.rlocsx[alpha], self.rlocsx[beta]], [self.rlocsy[alpha], self.rlocsy[beta]])

        nxst1, nxst2 = bsx
        nyst1, nyst2 = bsy
        nxfin1, nxfin2 = bex
        nyfin1, nyfin2 = bey

        GrecA = self.Green[dc_xyz[self.comp_src],dc_rtz[getattr(self, use_p[sORe(modelling_inv)])],nyst1:nyfin1,nxst1:nxfin1,:]
        GrecB = self.Green[dc_xyz[self.comp_src],dc_rtz[getattr(self, use_q[sORe(modelling_inv)])],nyst2:nyfin2,nxst2:nxfin2,:]

        f = np.conj(GrecA[...,1:self.nom_nneg]) * GrecB[...,1:self.nom_nneg]

        # con = self.dom/(2*np.pi)
        con = 1/(2*np.pi)

        kp = 2 * (ccpos[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con
        kn = 2 * (ccneg[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con

        # kernpos = np.sum(kp, axis=2)
        # kernneg = np.sum(kn, axis=2)

        kernpos = spi.simps(kp,None,dx=self.dom,axis=2)
        kernneg = spi.simps(kn,None,dx=self.dom,axis=2)

        norm_kernpos = np.sum(kernpos*self.distribs_inv) * dx**2
        norm_kernneg = np.sum(kernneg*self.distribs_inv) * dx**2
        # kernel normalization, eq. 29

        if norm_kernpos < 0.95 or norm_kernneg < 0.95 or norm_kernpos > 1.05 or norm_kernneg > 1.05:
        	# raise SystemExit("Problem with normalization of source kernel for receivers %d-%d.\
             # Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg))
             pass

        return kernpos, kernneg

############################################ Main program ###########################################################

if __name__ == '__main__':

    #********************************************** Load custom modules ****************************************************
    import SW1D_earthsr.utils_pre_code as u0
    import anseicca_utils2 as u2
    import utils_io as uio

    #********************************************** Read input arguments ****************************************************

    uioo = uio.process_args(sys.argv)
    try:
    	mod1dfile = uioo.mod1dfile
    	efile_ray = uioo.egn_ray
    	dfile_ray = uioo.disp_ray
    	scalar=False
    	elastic=True
    	try:
    		efile_lov = uioo.egn_lov
    		dfile_lov = uioo.disp_lov
    	except AttributeError:
    		# no Love wave input
    		pass
    except AttributeError:
    	scalar=True
    	elastic=False

    #*********************************** Set frequency/time-series characteristics *****************************************

    sig_att = config.sig_char
    # values used in the FIRST version of this code - DO NOT EDIT OR DELETE - dt = 0.2, nsam = 250, cf = 0.3

    #********************************************* Set surface geometry ***************************************************

    # numrecs=20
    # receiver locations (rlocx and rlocy) are specified in number of grid points away from origin (along x,y axes)

    # rlocx = np.array([18, 24, -23, 24, 7, -25, -14, 2, 27, 27, -21, 28, 27, -1, 18, -22, -5, 24, 17, 27])
    # rlocy = np.array([9, -28, 20, 26, 10, 15, 14, -7, 9, -20, 12, -29, -14, -28, -25, 19, 11, -11, 27, -28])

    # numrecs=8 # this should be the size of rlocx and rlocy
    # rlocx=np.array([35, -35, 65, -65, 0, 0, 0, 0])
    # rlocy=np.array([0, 0, 0, 0, 35, -35, 45, -45])

    # numrecs=8
    # rlocx=np.array([6, 12, 40, 90, -6, -12, -40, -90])
    # rlocy=np.array([0, 0, 0, 0, 0, 0, 0, 0])

    # numrecs=6
    # rlocx=np.array([-20, -10, -5, 5, 10, 20])
    # rlocy=np.array([0, 0, 0, 0, 0, 0])

    numrecs=4
    rlocx=np.array([-10, -5, 5, 10])
    rlocy=np.array([0, 0, 0, 0])

    # numrecs=4
    # rlocx=np.array([0, 0, -12, -30])
    # rlocy=np.array([12, 30, 0, 0])

    # numrecs=2
    # rlocx=np.array([-20, 50])
    # rlocy=np.array([40, 10])

    # numrecs=2
    # rlocx=np.array([-5, 5])
    # rlocy=np.array([0, 0])

    #-------------------------------- Model reading (elastic case) ----------------------------------
    if elastic:
    # read input depth-dependent model and fix/extract necessary parameters
        upreo = u0.model_1D(mod1dfile)
        Zpts_all = upreo.deps_all
        discon_mod = upreo.mod_hif
        upreo.fix_max_depth(dg.zmax)
        Zpts_use = upreo.deps_tomax
        # discon_mod_use = discon_mod[discon_mod<=config.dmax]
        print("Layer interfaces in model: ", discon_mod, discon_mod.size)
        print("Depth points to be used in code: ", Zpts_use, Zpts_use.size)
        nzmax = Zpts_use.size
        wspeed = config.scal_mod.wavspeed_scal2D # NEED TO CHANGE
    else:
        nzmax = None
        wspeed = config.scal_mod.wavspeed_scal2D
        # wavespeed everywhere in model (km/s)

    #********************************************* Run code ***************************************************
    nrth=8
    # nrth is just a receiver number threshold below which individual source kernels are stored (and plotted)

    kcao = inv_cc_amp(rlocx, rlocy, sig_att, True, True, None, None)

    u2.post_run(wspeed, sig_att, 0, oica=kcao)

    #********************************************* Make plots ***************************************************

    def see_individual_skernels():

    	lti=np.tril_indices(kcao.nrecs,k=-1)
    	# lower triangular indices in numpy's default ordering
    	ise=np.argsort(lti[1], kind='mergesort')
    	r=lti[0][ise]
    	c=lti[1][ise]
    	cc_pdist=kcao.dist_rp[(r,c)]
    	# now we have picked out lower triangular elements of kcao.dist_rp in the order that
    	# we want them, i.e. in the order in which cross-correlations are done in this code

    	fig_sk=plt.figure()
    	for p in range(len(kcao.skers)):
    		ax_sk=fig_sk.add_subplot(3,2,p+1, aspect='equal')
    		cax_sk=ax_sk.pcolor(dg.gx, dg.gy, kcao.skers[p],cmap=plt.cm.jet) #, vmin=-0.1, vmax=0.1)
    		#ax_sk.plot(dx*rlocx, dx*rlocy, 'kd', markerfacecolor="None")
    		# use above line to mark all receivers on each subplot OR use below two lines to mark only the
    		# relevant receiver pair on each subplot
    		ax_sk.plot(dx*rlocx[r[p]], dx*rlocy[r[p]], 'kd', markerfacecolor="None")
    		ax_sk.plot(dx*rlocx[c[p]], dx*rlocy[c[p]], 'kd', markerfacecolor="None")
    		spname = "Distance %.2f km" %(cc_pdist[p])
    		ax_sk.set_title(spname)
    		plt.colorbar(cax_sk,ax=ax_sk)

    #-------------------------------------------------------------------------------------------------------

    fig0=plt.figure()
    ax0=fig0.add_subplot(111)
    ax0.set_title("True model")
    if omost_fac==2:
    	cax0=ax0.pcolor(dg.gx,dg.gy,kcao.distribs_true,cmap=plt.cm.jet)
    elif omost_fac==3:
    	cax0=ax0.pcolor(dg.gx2,dg.gy2,kcao.distribs_true,cmap=plt.cm.jet)
    ax0.plot(dx*rlocx, dx*rlocy, 'wd', markerfacecolor="None")
    plt.colorbar(cax0,ax=ax0)

    if hasattr(kcao, 'iter'):
        fig1=plt.figure()
        ax1=fig1.add_subplot(111)
        ax1.set_title("Positive branch")
        cax1=ax1.pcolor(dg.gx,dg.gy,kcao.mfit_kern_pos,cmap=plt.cm.jet)
        ax1.plot(dx*rlocx, dx*rlocy, 'wd', markerfacecolor="None")
        fig1.colorbar(cax1)

        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        ax2.set_title("Negative branch")
        cax2=ax2.pcolor(dg.gx,dg.gy,kcao.mfit_kern_neg,cmap=plt.cm.jet)
        ax2.plot(dx*rlocx, dx*rlocy, 'wd', markerfacecolor="None")
        fig2.colorbar(cax2)

        fig4=plt.figure()
        ax4=fig4.add_subplot(111)
        ax4.set_title("Inversion result")
        if omost_fac==2:
        	cax4=ax4.pcolor(dg.gx,dg.gy,kcao.distribs_inv,cmap=plt.cm.jet)
        elif omost_fac==3:
        	cax4=ax4.pcolor(dg.gx2,dg.gy2,kcao.distribs_inv,cmap=plt.cm.jet)
        ax4.plot(dx*rlocx, dx*rlocy, 'wd', markerfacecolor="None")
        plt.colorbar(cax4,ax=ax4)

    print("Inter-receiver distances: ")
    print(kcao.dist_rp)

    if numrecs<nrth:
        nc2=kcao.nrecs*(kcao.nrecs-1)/2
        try:
            assert len(kcao.skers) == nc2
        except AssertionError:
            if len(kcao.skers)==0:
                print("No source kernels to plot")
            else:
                raise SystemExit("Problem with number of source kernels: %d %d" %(len(kcao.skers),nc2))

        else:
            try:
                see_individual_skernels()
            except ValueError:
                print("Problem plotting individual source kernels due to incompatible number of subplots")

    plt.show()
