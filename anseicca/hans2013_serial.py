#!/usr/bin/python

# General purpose modules
import os
import sys
import numpy as np
import scipy.signal as ss
import scipy.special as ssp
import scipy.integrate as spi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.path.append('../modules_common')
    # path to common modules for the cc source and structure inversion codes
    sys.path.append(os.path.expanduser('~/code_general/modules.python'))
    # path to the "SW1D_earthsr" set of modules
    sys.path.append(os.path.expanduser('~/code_general/wav_prop_numerical'))
    # path to the "Helmholtz_equation_FD" and "devito_solvers_TD" modules

# Custom modules (unconditional set1)
import config_file as config

if not __name__ == '__main__':
# get essential variables from main (calling) program
    try:
        efile_ray = sys.modules['__main__'].user_choices['elas_mod_1D']['egn_ray']
        dfile_ray = sys.modules['__main__'].user_choices['elas_mod_1D']['disp_ray']
    except TypeError:
        # this is the 2D-scalar case
        scalar=True
        elastic=False
    else:
        scalar=False
        elastic=True
        nzmax = sys.modules['__main__'].user_choices['nz']
        discon_mod = sys.modules['__main__'].user_choices['elas_mod_1D']['hif_mod']
        Zpts_all = sys.modules['__main__'].user_choices['elas_mod_1D']['dep_pts_mod']
        try:
            efile_lov = sys.modules['__main__'].user_choices['elas_mod_1D']['egn_lov']
            dfile_lov = sys.modules['__main__'].user_choices['elas_mod_1D']['disp_lov']
            love=True
        except KeyError:
            love=False

    vel_2D_acou_obs = sys.modules['__main__'].user_choices['acou_vel_mod_obs']
    vel_2D_acou_syn = sys.modules['__main__'].user_choices['acou_vel_mod_syn']

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
# Custom modules (unconditional set2)
import cctomo_utils1 as u1
import cctomo_utils2 as u2

# pertaining to source model parameterization
param_mod_tru = config.tru_mod_type
param_mod_inv = config.inv_mod_type
mod_specs = {'mg': config.somod_mg_specs, 'rg': config.somod_rg_specs, 'rgr': config.somod_rgr_specs, 'gg': config.somod_gg_specs}
if param_mod_inv=='egp':
    use_basis=False
else:
    use_basis=True

# pertaining to structure model type
c_scal = config.scal_mod.wavspeed_scal2D
lat_homo_tru = config.tru_struc_lat_homo
lat_homo_inv = config.inv_struc_lat_homo

# pertaining to modelling system
dc_xyz = config.comp_dic_xyz
dc_rtz = config.comp_dic_rtz
tensor_rtz = config.green_tensor_rtz

modelling_tru = config.tru_mdlng_type
modelling_inv = config.inv_mdlng_type

if config.ext_data:
    init_amp_scaling = True
    # amplitude scaling usually required in case of real data
else:
    if modelling_tru != modelling_inv:
        # for synthetic tests: may be required in case 'tru_mdlng_type' and 'inv_mdlng_type' are DIFFERENT
        init_amp_scaling = False
        # user must set this paramter judiciously. 'True' setting recommended ONLY when unable to reconcile
        # (orders of magnitude) amplitude discrepancies between results of different modelling methods.
    else:
        init_amp_scaling = False

def sORe(instr):
    if 'scal' in instr:
        return 'scal'
    elif 'elas' in instr:
        return 'elas'

ncomp = {'scal': 1, 'elas': 3}
# number of components of motion, in the scalar and elastic cases
use_p = {'scal': 'comp_scal', 'elas': 'comp_p'}
use_q = {'scal': 'comp_scal', 'elas': 'comp_q'}

sco = 3 - ncomp[sORe(modelling_tru)]
scs = 3 - ncomp[sORe(modelling_inv)]

# pertaining to synthetic tests (synthetic data generated internally)
sd = config.syn_data
omost_fac = sd.ofac
noise_amp = sd.noise_level
noise_band = sd.noise_band

# Finally, custom modules required conditionally
mlg1='anal_elas_1D'
mlg2='num_scal_2D_FD'
mlg3='num_scal_2D_TD'
if modelling_tru==mlg1 or modelling_inv==mlg1:
    import SW1D_earthsr.Green_functions_3D as gf3
# if modelling_tru==mlg2 or modelling_inv==mlg2:
#     import Helmholtz_equation_FD.solver_2D as hs2d
if modelling_tru==mlg3 or modelling_inv==mlg3:
    import devito_solvers_TD.Helmholtz_2D.acoustic_solver as dhas

##########################################################################################################################

class inv_cc_amp:

    def __init__(self, rlocsx, rlocsy, signal, iterate, only1_iter, dobs, dobs_info):

        """
        rlocsx (type 'type 'numpy.ndarray'): x-coordinates of all receivers (in grid-point units)
        rlocsy (type 'type 'numpy.ndarray'): y-coordinates of all receivers (in grid-point units)
        signal (type 'instance'): object of class "SignalParameters" containing various signal characteristics of the data
        dobs (optional, type 'numpy.ndarray'): the data (EXTERNAL DATA ONLY)
        dobs_info (optional, type 'tuple'): Tuple containing auxillary data information such as S/N ratio (REAL DATA ONLY)
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

        self.signal = signal
        self.nom = self.signal.nsam
        self.deltat = self.signal.dt
        self.f0 = self.signal.cf
        # self.lf = self.signal.lf
        # self.hf = self.signal.hf
        altuk = self.signal.altukey

        self.t = self.signal.tt
        self.fhz = self.signal.fhz
        self.fpos_pow = self.signal.fhz_pow_pos
        self.dom = self.signal.domega
        self.omega = self.signal.omega_rad
        self.nom_nneg = self.signal.n_nn_fsam
        self.pss = self.signal.pow_spec_sources
        # print("From h13 (self.t):")
        # print(self.t)

        self.data_availability = dobs_info[0]
        self.snr_val =  dobs_info[1]
        self.meas_win = dobs_info[2]

        npairs_total = int(self.nrecs*(self.nrecs-1)/2)
        npairs_use = np.nonzero(np.tril(self.data_availability))[0].size
        self.npfwdm = npairs_total - npairs_use
        # npfwdm -> number of pairs for which data missing

        if not (dobs is None):
        # external data case
            self.reald = True
            self.obscross = dobs
            self.obscross_aspec = np.abs(np.fft.fft(dobs,axis=0))
        else:
        # internal data case
            self.reald = False
            self.obscross_tensor = np.zeros((3, 3, self.nom, self.nrecs, self.nrecs), dtype='complex')
            if init_amp_scaling:
                self.obscross_aspec = np.zeros(self.obscross_tensor[0,0,...].shape)

        try:
            assert self.reald == config.ext_data
        except AssertionError:
            raise SystemExit("Config paramter 'ext_data' inconsistent with input to core module. This will\
            mess up various settings and break the code. Please rectify.")

        # compute_cc = {True: self.compute_cc_lathomo, False: self.compute_cc_lathet}

        self.dvar_pos = np.ones(npairs_use)
        self.dvar_neg = np.ones(npairs_use)

        self.rlocsx = rlocsx*dx
        self.rlocsy = rlocsy*dx

        self.setup()
        if iterate:
            self.fwd_prep()

        self.num_mparams=self.basis.shape[0]
        # self.distribs_inv=np.copy(self.distribs_start)
        self.allit_mc = []
        self.allit_misfit = []
        # self.allit_synenv = []
        self.allit_syncross = []
        self.flit_indmis_p = []
        self.flit_indmis_n = []
        # variables with names ending in "_inv"  contain values for current (ulimately last) iteration only
        # variables with names starting with "allit_" are lists where each element corresponds to an iteration of the inversion.
        # variables with names starting with "flit_" are two-element lists, storing first (f) and last (l) iteration values only, of certain quantities.

        self.allit_mc.append(np.copy(self.mc_start))

        if __name__ == '__main__':
        	if self.nrecs<nrth:
        		self.skers=[]

        itnum=0
        forced=False

        while iterate:

            iter_mc = self.allit_mc[-1]
            if use_basis:
                self.distribs_inv = np.einsum('k,klm',iter_mc**2,self.basis)
            else:
                self.distribs_inv = iter_mc.reshape(ngpmb,ngpmb)
            if itnum==0:
                assert np.allclose(self.distribs_inv, self.distribs_start)

            # self.syncross = np.zeros((self.nom, self.nrecs, self.nrecs))#, dtype='complex')
            self.syncross_tensor = np.zeros((3, 3, self.nom, self.nrecs, self.nrecs), dtype='complex')
            # self.syncross_aspec = np.zeros(self.obscross.shape)

            #*************** inversion-related variables
            self.Gmat_pos=np.zeros((npairs_use,self.num_mparams))
            self.Gmat_neg=np.zeros((npairs_use,self.num_mparams))

            self.deltad_pos=np.zeros(npairs_use)
            self.deltad_neg=np.zeros(npairs_use)

            mfit_kern_pos = np.zeros((ngpmb, ngpmb))
            mfit_kern_neg = np.zeros((ngpmb, ngpmb))
            # mfit_kern -> misfit_kernel

            #*************** compute cross-correlations and make measurements
            if (not config.ext_data) and (itnum==0):
            # SYNTHETIC TEST (INTERNAL DATA) CASE, compute "observed data" synthetically
                # compute_cc[lat_homo_tru]('obs')
                self.compute_cc_ptsources(lat_homo_tru, 'obs', itnum)

            # compute_cc[lat_homo_inv]('pre')
            self.compute_cc_ptsources(lat_homo_inv, 'pre', itnum)
            self.make_measurement()

            print("Starting computation of source kernels for each receiver pair...")
            cp=0 # cp -> count_pair
            for j in range(self.nrecs-1):
                for i in range(j+1,self.nrecs):
                    # print("...receivers ", i,j)
                    if self.data_availability[i,j] == 0:
                        print("...skipping receivers ", i,j)
                        pass
                    else:
                        sker_p, sker_n = self.diffkernel(i,j)
                        # Computing individual source kernels (eq. 15)

                        # build the G-matrix
                        if use_basis:
                            kb_prod = sker_p * self.basis * 2*iter_mc[:,None,None]
                            self.Gmat_pos[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2
                            kb_prod = sker_n * self.basis * 2*iter_mc[:,None,None]
                            self.Gmat_neg[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2
                        else:
                            self.Gmat_pos[cp,:] = sker_p.flatten()
                            self.Gmat_neg[cp,:] = sker_n.flatten()

                        if __name__ == '__main__':
                        	if self.nrecs<nrth and itnum==0:
                        		self.skers.append(sker_p)

                        with np.errstate(invalid='raise'):
                        # when 'obsamp' and 'synamp' are both zero, that will raise an "invalid" error (division by 0)
                            try:
                                self.deltad_pos[cp] = np.log(self.obsamp_pos[i,j]/self.synamp_pos[i,j])
                                # print("obsamp_pos and synamp_pos for receivers ", i, j, self.obsamp_pos[i,j], self.synamp_pos[i,j])
                                # Computing misfit kernels, i.e. eq. 30 (positive branch)
                                mfit_kern_pos += sker_p * self.deltad_pos[cp]

                                self.deltad_neg[cp] = np.log(self.obsamp_neg[i,j]/self.synamp_neg[i,j])
                                # print("obsamp_neg and synamp_neg for receivers ", i, j, self.obsamp_neg[i,j], self.synamp_neg[i,j])
                                # Computing misfit kernels, i.e. eq. 30 (negative branch)
                                mfit_kern_neg += sker_n * self.deltad_neg[cp]

                            except (FloatingPointError, ZeroDivisionError) as e:
                                # this SHOULD NOT HAPPEN
                                # 'obsamp_pos/neg' as well as 'synamp_pos/neg' should be non-zero for the station pairs considered here
                                raise SystemExit("Problem computing data misfit - unexpected zero values")

                        # print("CHECK sker (pairwise source kernels): ")
                        # print(cp, np.sum(sker_p))
                        # print("CHECK deltad_pos (pairwise): ")
                        # print(cp, self.deltad_pos[cp])
                        cp+=1

            # print("CHECK deltad (total): ")
            # print(np.sum(self.deltad_pos), np.sum(self.deltad_neg))
            # print("CHECK mfit_kern: ")
            # print(np.sum(mfit_kern_pos), np.sum(mfit_kern_neg))

            #*********** things to do on first iteration
            if itnum==0:
                if self.reald:
                # complete the calculation of the data errors. NB: we consider two types of error.
                # The first one (energy decay) is independent of the measurements and is already computed.
                # The second (SNR) is defined relative to the measurements, so we must compute the absolute values here.

                    dvar_snr_pos = np.square(self.esnrpd_ltpb * self.obsamp_pos)
                    dvar_snr_neg = np.square(np.transpose(self.esnrpd_ltpb) * self.obsamp_neg)

                    # combine different errors
                    dvar_pos = dvar_snr_pos #+ self.dvar_egy_ltpb
                    dvar_neg = dvar_snr_neg #+ np.transpose(self.dvar_egy_ltpb)

                    # print("CHECK dvar (matrix form):")
                    # print(dvar_pos)

                    zvar_p=np.argwhere(dvar_pos==0)
                    zvar_n=np.argwhere(dvar_neg==0)
                    if not (np.all(zvar_p[:,0]<=zvar_p[:,1]) and np.all(zvar_n[:,0]<=zvar_n[:,1])):
                    # this means there are 0-values in the lower-triangular part of 'dvar_pos/neg' - should be due to missing data only.
                        ltzv_p=zvar_p[zvar_p[:,0]>zvar_p[:,1]]
                        ltzv_n=zvar_n[zvar_n[:,0]>zvar_n[:,1]]
                        if (ltzv_p.shape[0] != self.npfwdm) or (ltzv_n.shape[0] != self.npfwdm):
                            print(ltzv_p.shape)
                            print(ltzv_n.shape)
                            raise SystemExit("Problem with observed data errors - unexpected zero values.")

                    # finally, convert data variance from matrix-form (2D) to vector-form (1D)
                    dv_mat = {'p': dvar_pos, 'n': dvar_neg}
                    dv_vec = {'p': self.dvar_pos, 'n': self.dvar_neg}
                    for br in dv_mat:
                        count_pair=0
                        count_zero_err=0
                        for col in range(dv_mat[br].shape[1]):
                            for row in range(col+1,dv_mat[br].shape[0]):
                                if dv_mat[br][row,col]==0:
                                    count_zero_err+=1
                                else:
                                    dv_vec[br][count_pair] = dv_mat[br][row,col]
                                    count_pair+=1

                        assert count_zero_err==self.npfwdm

                # regardless of real/external or synthetic/internal data, store the first-iteration values of certain quantities
                self.mfit_kern_pos = mfit_kern_pos
                self.mfit_kern_neg = mfit_kern_neg

            def record_flit():
             	# record inversion progress
                self.flit_indmis_p.append(self.deltad_pos)
                self.flit_indmis_n.append(self.deltad_neg)

            # print("CHECK obsamp: ")
            # print(self.obsamp_pos, self.obsamp_neg)
            # print("CHECK synamp: ")
            # print(self.synamp_pos, self.synamp_neg)

            # print("CHECK dvar (vector form): ")
            # print(self.dvar_pos, self.dvar_neg)

            with np.errstate(invalid='raise', divide='raise'):
                try:
                    wmp = self.deltad_pos / np.sqrt(self.dvar_pos)
                    wmn = self.deltad_neg / np.sqrt(self.dvar_neg)
                except (ZeroDivisionError, FloatingPointError) as e:
                    # this should never happen; the data variance vector should not contain any zero-variances,
                    # because this has been taken care of when creating the vector.
                    print(e)
                    print(self.dvar_pos)
                    print(self.dvar_neg)
                    raise SystemExit("Problem with data errors vector - contains zero values.")

            total_misfit = 0.5*(np.dot(wmp,wmp) + np.dot(wmn,wmn))

            if itnum==0:
            	record_flit()
            self.allit_misfit.append(total_misfit)
            self.allit_syncross.append(self.syncross)
            # self.allit_synenv.append(self.synenv)

            print("TOTAL MISFIT: ", total_misfit)
            print(self.allit_misfit)

            if itnum==1:
                if only1_iter:
                # FORCED STOP FOR TESTING: last misfit stored will correspond to first updated model
                    forced=True
                    iterate=False
                    record_flit()

            if (itnum>0) and (not forced):
            # determine whether to terminate inversion or iterate further
                mf_curr = self.allit_misfit[-1]
                mf_prev = self.allit_misfit[-2]
                pchange = 100*(mf_prev - mf_curr)/mf_prev
                if (pchange>=0 and pchange<2) or itnum>15: #or pchange<0:
                    iterate=False
                    # inversion terminated.
                    # store the individual misfits corresponding to the final iteration model
                    record_flit()
                    if pchange<0:
                        print("Inversion terminated due to increasing misfit.")

            if iterate:
                #*********** do actual inversion (model update)
                self.ido = u2.inversion(self.num_mparams)
                new_mc = self.ido.invert(self.num_mparams, self.basis, self.mc_start, iter_mc,\
                  mfit_kern_pos, mfit_kern_neg, self.Gmat_pos, self.Gmat_neg, self.deltad_pos, self.deltad_neg, self.dvar_pos, self.dvar_neg)
                self.allit_mc.append(new_mc)
                # print("CHECK allit_mc (model coefficients): ")
                # print(self.allit_mc[0])
                # print(self.allit_mc[-1])

                print("END OF ITERATION %d" %(itnum))
                itnum+=1

        #*********************** End of loop over iterations *******************

	########################################################################################################################

    def setup(self):

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
        # print "Frequency samples are: ", self.fhz

        # # hlbox_omost = omost_fac*hlbox_outer/2.0
        self.ndouble=2*ngpmb-1

        umdo = u2.use_modelling_domain(self.rlocsx, self.rlocsy)
        self.dist_rp_grid = umdo.dist_rp
        self.dist_rp_sorted = umdo.alldist_1D

        #******************************* source distributions and observation errors ********************************************

        sdist_type = {'mg': u1.somod.mult_gauss, 'rg': u1.somod.ringg, 'rgr': u1.somod.rgring, 'gg': u1.somod.gcover}

        self.distribs_start = np.zeros((ngpmb,ngpmb))
        # used to generate the synthetics for inversion; this source distribution IS involved in computing source kernels

        mag1_true=1
        mag1_start=1

        nbasis = {'egp': ngpmb**2,
                    'mg': len(mod_specs['mg']['r0']),
                    'rg': int(360/mod_specs['rg']['as']),
                    'rgr': len(mod_specs['rgr']['r0']),
                    'gg': int((dg.box_len/mod_specs['gg']['ls'])**2)}

        gg_loc = None
        alltheta = None

        # ----------------------------------------------------------------------

        # setup to use basis functions, in case of 'mg' OR 'rgr'
        allcen = lambda specs: specs['r0']

        # setup to use basis functions, in case of 'rg'
        if param_mod_inv=='rg' or param_mod_tru=='rg':
            astep = mod_specs['rg']['as']
            alltheta_deg=np.arange(0,360,astep)
            alltheta=alltheta_deg*np.pi/180
            assert alltheta.size==nbasis['rg']

        # setup to use basis functions, in case of 'gg'
        if param_mod_inv=='gg' or param_mod_tru=='gg':
            gg_loc=[]
            astep = mod_specs['gg']['ls']
            row_val=np.arange(-dg.box_len/2+(astep/2),dg.box_len/2,astep)
            xv,yv=np.array(np.meshgrid(row_val,row_val))
            for i in range(len(xv)):
                for j in range(len(xv[0])):
                    gg_loc.append([xv[i,j],yv[i,j]])
            gg_loc=np.array(gg_loc)
            if len(gg_loc)!=nbasis['gg']:
                nbasis['gg']=len(gg_loc)
            # assert gg_loc.size==nbasis['gg']

        # ----------------------------------------------------------------------

        basis_loc = {'rg': alltheta, 'mg': allcen(mod_specs['mg']), 'rgr': allcen(mod_specs['rgr']), 'gg': gg_loc}

        init_basis = lambda m,n: np.zeros((m,n,n))
        init_model = lambda n: np.zeros((n,n))

        self.basis = init_basis(nbasis[param_mod_inv],ngpmb)
        self.mc_start = np.ones(nbasis[param_mod_inv])
        # mc -> model_coefficients

        self.mc_start *= mag1_start
        self.mc_start[int(nbasis[param_mod_inv]/2)] = 1

        if use_basis:
            for k,locator in enumerate(basis_loc[param_mod_inv]):
                self.basis[k,:,:] = sdist_type[param_mod_inv](locator, mod_specs[param_mod_inv])
                self.distribs_start += (self.mc_start[k]**2) * self.basis[k,:,:]
        else:
            self.distribs_start = init_model(ngpmb) + 1.0

        # ----------------------------------------------------------------------
        get_mct_mg = lambda specs: mag1_true*np.array(specs['mag'])
        get_mct_rgr = lambda specs: mag1_true*np.array(specs['mag'])
        # ----------------------------------------------------------------------

        def get_mct_rg(specs):
            ans = self.mc_true * mag1_true
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
                ans[relind]=mag2[p]+mag1_true
            return ans

        # ----------------------------------------------------------------------

        def get_mct_gg1(specs):
            mag2=specs['mag']
            #nperts=specs['np']
            coord=np.array(specs['pos'])
            ans = self.mc_true * mag1_true
            k=0
            for i in coord:
                x_coord=gg_loc.T[0]-i[0]
                y_coord=gg_loc.T[1]-i[1]
                dist=np.sqrt(x_coord**2+y_coord**2)
                index=(np.argwhere(dist==min(dist)).T)
                ans[index]=mag1_true+mag2[k]
                print(gg_loc[index][0])
                k+=1
            return ans

        def get_mct_gg2(specs):
            ans=np.ones(len(gg_loc))
            k=int((len(gg_loc))**0.5)
            grid=gg_loc.reshape((k,k,2))
            for p in range(specs['np']):
                xu=specs['x_u'][p]
                xl=specs['x_l'][p]
                yu=specs['y_u'][p]
                yl=specs['y_l'][p]
                mag=specs['mag'][p]
                sub=grid[xl-1:xu,yl-1:yu,:]
                sub=sub.reshape(((yu-yl+1)*(xu-xl+1),2))
                index=[]
                for i in sub:
                    index=(np.argwhere(np.all(gg_loc==i,axis=1)))
                    ans[index]=mag
            return ans
        # ----------------------------------------------------------------------

        get_mc_true = {'rg': get_mct_rg, 'mg': get_mct_mg, 'rgr': get_mct_rgr, 'gg': get_mct_gg2}

        if not config.ext_data:
        # SYNTHETIC (INTERNAL) DATA CASE - generate the 'true' model
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
                self.distribs_true += (self.mc_true[k]**2) * basis_true[k,:,:]
            self.distribs_true += mag1_true

        if self.reald:
        # EXTERNAL (REAL) DATA CASE - compute observation errors
            aeo = u1.assign_errors(self.snr_val)
            self.esnrpd_ltpb = aeo.snr_error(self.dist_rp_grid)
            # self.dvar_egy_ltpb = aeo.decay_rate_error()

        print("Completed initial setup...")

	########################################################################################################################

    def fwd_prep(self):

        print("Computing distances from origin..")
        r = np.sqrt(dg.gx3**2 + dg.gy3**2)
        r = config.add_epsilon(r)
        # this is done so that r does not contain any zeros; to prevent the Hankel function
        # from blowing up at the origin (r=0)

        ntot_omost = omost_fac*(ngpmb-1) + 1
        if omost_fac==2:
            assert ntot_omost == self.ndouble

        if lat_homo_tru or lat_homo_inv:
            """ Laterally HOMOGENEOUS structure model(s) """

            print("Computing Green functions for point source at origin..")
            self.Green = np.zeros((3,3,self.ndouble,self.ndouble,self.nom_nneg), dtype='complex')
            if not config.ext_data:
                self.Green_obs = np.zeros((3,3,ntot_omost,ntot_omost,self.nom_nneg), dtype='complex')
                # "Green_obs" may be larger (spatially) than "Green", i.e. the 'observed data' may be computed using
                # a domain that is larger than the inverse modelling domain.

            if omost_fac==2:
                sx=0
                ex=self.ndouble
                sy=sx
                ey=ex
            elif omost_fac>2:
                sxl, exl, syl, eyl = config.box_indices_largerbox_src([0], [0], self.ndouble)
                sx = sxl[0]
                ex = exl[0]
                sy = syl[0]
                ey = eyl[0]

            for i in range(1,self.nom_nneg):
                # compute Green's function only at those frequencies where the source spectrum is non-zero
                # self.thresh_pow = (0.01*np.amax(self.pss))/100
                # if self.pss[i] < self.thresh_pow:
                if not (self.fhz[i] in self.fpos_pow):
                    # print("(ignoring FREQUENCY %f Hz)" %(self.fhz[i]))
                    pass
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
                            # swgmfo.G_cartesian_grid(self.gx3, self.gy3, r, 2)
                            swgmfo.G_cartesian_grid(dg.gx3, dg.gy3, r, 2)
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
                                'num_scal_2D_TD': None}

                    if elastic:
                        ind0=slice(self.Green.shape[0])
                        ind1=slice(self.Green.shape[1])
                    elif scalar:
                        ind0=2
                        ind1=2

                    try:
                        self.Green[ind0,ind1,:,:,i] = solution[modelling_inv][sy:ey,sx:ex]
                    except TypeError:
                        print(sx,ex,sy,ey)
                        # function represented by 'solution' is None (time-domain numerical solvers not used to obtain Green functions)
                        assert modelling_inv=='num_scal_2D_TD'
                        pass
                    if not config.ext_data:
                        self.Green_obs[ind0,ind1,:,:,i] = solution[modelling_tru]

        if not (lat_homo_tru and lat_homo_inv):
            """ Laterally HETEROGENEOUS structure model(s) """

            try:
                assert ('num' in modelling_tru) or ('num' in modelling_inv)
                # modelling type has to be numerical
            except AssertionError:
                raise SystemExit("Laterally heterogeneous structure incompatible with analytical modelling.\
                Please check modelling settings in config file.")

            gs_xy = [dx*1e3, dx*1e3]                            # MUST be in metres
            dom_orig = np.asarray([dg.X[0], dg.Y[0]])           # MUST be in kilometres
            ngp_xy = [dg.X.size, dg.Y.size]                     # number of grid points [x,y]
            if omost_fac==2:
                dom_orig_obs = np.copy(dom_orig)
                ngp_xy_obs = np.copy(ngp_xy)
            elif  omost_fac==3:
                dom_orig_obs = np.asarray([dg.X2[0], dg.Y2[0]])
                ngp_xy_obs = [dg.X2.size, dg.Y2.size]

            if vel_2D_acou_syn is None:
                self.vp_dvto_syn = c_scal * np.ones(ngp_xy)              # MUST be in km/s
            else:
                self.vp_dvto_syn = vel_2D_acou_syn

            self.vp_dvto_obs = None
            if not config.ext_data:
                if vel_2D_acou_obs is None:
                    self.vp_dvto_obs = c_scal * np.ones(ngp_xy_obs)              # MUST be in km/s
                else:
                    self.vp_dvto_obs = vel_2D_acou_obs

            if modelling_inv==mlg3:
                self.dhaso = dhas.sim_int_sources(dom_orig, gs_xy, self.vp_dvto_syn, self.deltat*1e3, self.pss)
            if not config.ext_data and modelling_tru==mlg3:
                dhaso_obs = dhas.sim_int_sources(dom_orig_obs, gs_xy, self.vp_dvto_obs, self.deltat*1e3, self.pss)

            self.rec_locs_dvto = np.empty((self.nrecs, 2))
            self.rec_locs_dvto[:, 0] = self.rlocsx*1e3
            self.rec_locs_dvto[:, 1] = self.rlocsy*1e3

            self.numsol_pt_src = np.zeros((self.nrecs, self.nom_nneg, ngpmb, ngpmb), dtype='complex')
            # self.numsol_pt_src = np.zeros((self.nrecs, self.nom, ngpmb, ngpmb), dtype='complex')
            if not config.ext_data:
                if omost_fac==2:
                    self.numsol_obs_pt_src = np.zeros((self.nrecs, self.nom_nneg, ngpmb, ngpmb), dtype='complex')
                elif omost_fac==3:
                    self.numsol_obs_pt_src = np.zeros((self.nrecs, self.nom_nneg, self.ndouble, self.ndouble), dtype='complex')

            tillf = -1 if self.nom_nneg%2==0 else self.nom_nneg

            # we loop over receivers, making each one a point source in turn
            for k in range(self.nrecs):
                src_loc = self.rec_locs_dvto[k,:].reshape(1,2)
                if modelling_inv==mlg3:
                    try:
                        self.dhaso.solve(src_loc)
                        self.dhaso.resample()
                        self.numsol_pt_src[k,...] = self.dhaso.get_FD()[slice(tillf),...]
                        # RHS uses 'rfft', so it may have one extra term in frequency (positive Nyquist term) compared to the LHS - this extra term must be excluded
                        # self.numsol_pt_src[k,...] = self.dhaso.get_FD()
                    except Exception as e:
                        print(e)
                        raise SystemExit("Problem with devito solution (syn), program aborted.")

                if not config.ext_data and modelling_tru==mlg3:
                    try:
                        dhaso_obs.solve(src_loc)
                        dhaso_obs.resample()
                        self.numsol_obs_pt_src[k,...] = dhaso_obs.get_FD()[slice(tillf),...]
                        # RHS uses 'rfft', so it may have one extra term in frequency (positive Nyquist term) compared to the LHS - this extra term must be excluded
                        # self.numsol_pt_src[k,...] = self.dhaso.get_FD()
                    except Exception as e:
                        print(e)
                        raise SystemExit("Problem with devito solution (obs), program aborted.")

            # sources potentially everywhere in domain (distributed source)
            # self.src_locs_dvto = np.empty((np.prod(ngp_xy),2))
            # self.src_locs_dvto[:, 0] = (self.gx.flatten())*1e3
            # self.src_locs_dvto[:, 1] = (self.gy.flatten())*1e3

            # self.dhasdo = dhas.sim_dist_sources(dom_orig, gs_xy, self.vp_dvto_syn)

	########################################################################################################################

    def compute_cc_ptsources(self, struc_hom, dat_type, iter_num):

        """ Algorithm using point sources only (invokes reciprocity at both receivers) """

        print("Computing cross-correlations...")

        struc_type = 'lhom' if struc_hom else 'lhet'

        if struc_type=='lhom':
        # analytical modelling in a laterally homogeneous structure
        # need to pick out the relevant part of the computed GF (larger box source trick)
            nxst, nxfin, nyst, nyfin = config.box_indices_largerbox_src(self.rlocsx, self.rlocsy)
            if omost_fac==2:
                nxst_obs = nxst
                nyst_obs = nyst
                nxfin_obs = nxfin
                nyfin_obs = nyfin
            elif omost_fac>2:
                # true model has size (length of side) = self.ndouble
                nxst_obs, nxfin_obs, nyst_obs, nyfin_obs = config.box_indices_largerbox_src(self.rlocsx, self.rlocsy, self.ndouble)

        elif struc_type=='lhet':
        # numerical modelling, laterally HETEROGENEOUS structure
            pass

        # loop over all possible receiver pairs (combinations)
        for k in range(self.nrecs-1):
            for j in range(k+1,self.nrecs):

                if dat_type=='pre':
                    if self.data_availability[j,k] == 0:
                        # print("...skipping cc calculation for receivers ", j,k)
                        pass
                    else:
                        print("...cc for receivers ", j, k)
                        if struc_type=='lhet':
                            solsyn_k = self.numsol_pt_src[k,...]
                            solsyn_j = self.numsol_pt_src[j,...]

                        for p in range(scs,3):
                            for q in range(scs,3):
                                print("Component %d-%d" %(p,q))

                                if struc_type=='lhom':
                                    Grec_j = self.Green[dc_xyz[self.comp_src],p,nyst[j]:nyfin[j],nxst[j]:nxfin[j],:]
                                    Grec_k = self.Green[dc_xyz[self.comp_src],q,nyst[k]:nyfin[k],nxst[k]:nxfin[k],:]

                                    Gsyn_j = np.transpose(Grec_j,[2,0,1])
                                    Gsyn_k = np.transpose(Grec_k,[2,0,1])

                                    self.syncross_tensor[p,q,:,j,k] = u1.compute_cc_1pair_ptsrc(Gsyn_j, Gsyn_k, self.signal, self.distribs_inv, self.pss)
                                    # print("CHECK pss")
                                    # print(np.sum(self.pss))

                                elif struc_type=='lhet':
                                    self.syncross_tensor[p,q,:,j,k] = u1.compute_cc_1pair_ptsrc(solsyn_j, solsyn_k, self.signal, self.distribs_inv)

                if dat_type=='obs':
                    print("...cc (obscross) for receivers ", j, k)
                    if struc_type=='lhet':
                        solobs_k = self.numsol_obs_pt_src[k,...]
                        solobs_j = self.numsol_obs_pt_src[j,...]

                    for p in range(sco,3):
                        for q in range(sco,3):
                            print("Component %d-%d" %(p,q))

                            if struc_type=='lhom':
                                Grec_j = self.Green_obs[dc_xyz[self.comp_src],p,nyst_obs[j]:nyfin_obs[j],nxst_obs[j]:nxfin_obs[j],:]
                                Grec_k = self.Green_obs[dc_xyz[self.comp_src],q,nyst_obs[k]:nyfin_obs[k],nxst_obs[k]:nxfin_obs[k],:]

                                Gobs_j = np.transpose(Grec_j,[2,0,1])
                                Gobs_k = np.transpose(Grec_k,[2,0,1])

                                self.obscross_tensor[p,q,:,j,k] = u1.compute_cc_1pair_ptsrc(Gobs_j, Gobs_k, self.signal, self.distribs_true, self.pss)
                            elif struc_type=='lhet':
                                self.obscross_tensor[p,q,:,j,k] = u1.compute_cc_1pair_ptsrc(solobs_j, solobs_k, self.signal, self.distribs_true)


        #***** End of double-loop over receiver pairs *****
        # APPLY ROTATION FROM XYZ TO RTZ COORDS
        # *************************************************

        icmp = lambda m,u: dc_rtz[getattr(self, u[sORe(m)])]

        # select desired cross-correlation component
        if dat_type=='pre':
            syncross_FD = self.syncross_tensor[icmp(modelling_inv, use_p), icmp(modelling_inv, use_q),...]
        elif dat_type=='obs':
            obscross_FD = self.obscross_tensor[icmp(modelling_tru, use_p), icmp(modelling_tru, use_q),...]
            self.obscross_aspec = np.abs(obscross_FD)

        if iter_num==0 and dat_type=='pre' and init_amp_scaling:

            # ------ POTENTIAL AMPLITUDE SCALING -----

            need_to_scale=True
            freq_max = np.amax(self.fpos_pow)

            evdo = u1.egy_vs_dist(self.reald, self.dist_rp_sorted, self.c/freq_max)
            self.egy_obs, self.oef = evdo.fit_curve_1byr(self.nom, self.obscross_aspec, 'FD', self.dist_rp_grid, self.npfwdm, evdo.sig_dummy)
            # oef -> observed_energy_fitted

            while need_to_scale:

                # ************ compute initial synthetic data energies
                self.syncross_aspec = np.abs(syncross_FD)
                self.egy_syn, self.sef = evdo.fit_curve_1byr(self.nom, self.syncross_aspec, 'FD', self.dist_rp_grid, self.npfwdm, None)
                # sef -> synthetic_energy_fitted

                # ************ compare observed and initial synthetic energies
                esf = np.nanmean(self.oef/self.sef)
                # esf -> energy_scale_factor

                if esf > 0.9 and esf < 1.1:
                    print("esf is %e, scaling of initial synthetics completed." %(esf))
                    need_to_scale=False
                else:
                    print("esf is %e, MULTIPLYING self.pss by %e" %(esf,np.sqrt(esf)))
                    # immediate correction (this iteration)
                    syncross_FD *= np.sqrt(esf)
                    # permanent correction (subsequent iterations)
                    if lat_homo_inv:
                        self.pss *= np.sqrt(esf)
                    else:
                        self.numsol_pt_src *= np.power(esf,1/4)

            # ------ End: POTENTIAL AMPLITUDE SCALING -----
        else:
            # Subsequent iterations OR no amplitude scaling
            pass

        # convert to time domain cross-correlations
        if dat_type=='pre':
            self.syncross = np.fft.fftshift(np.fft.ifft(syncross_FD, axis=0).real, axes=(0,))
        elif dat_type=='obs':
            obscross_clean = np.fft.fftshift(np.fft.ifft(obscross_FD, axis=0).real, axes=(0,))
            if noise_amp==0:
                self.obscross = obscross_clean
            elif noise_amp>0:
                ano = u1.add_noise_all_cc(obscross_clean, self.signal, noise_amp, noise_band)
                self.obscross = ano.noisy_signal

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
        # 	self.obscross_aspec = np.abs(np.fft.fft(self.obscross,axis=0))
        	# for the synthetic data case, this MAY BE generated here for the FIRST time; for the real data case
        	# this is a recalculation but now the upper triangular part of the matrix is also filled in.

    #######################################################################################################################

    # def compute_cc_distsources(self, dat_type):
    #
    #     """ May be implemented in the future
    #     """
    #
    #     for k in range(self.nrecs-1):
    #         if dat_type=='obs':
    #             print("...cc (obscross) for master receiver ", k)
    #
    #         # STEP 1: GF for point source at receiver
    #         src_loc = self.rec_locs_dvto[k,:].reshape(1,2)
    #         print("Source location for devito solver (point source): ", src_loc)
    #         self.dhaso.solve(self.par_dvto, pt_src, src_loc, self.rec_locs_dvto, True)
    #         self.dhaso.get_Green_FD(self.dhaso.rec_data, self.dhaso.wav_fld)
    #         nbl = self.dhaso.vel_model.nbl
    #         # GF_this_src = self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl]/self.amp_fac_dvto
    #         GF_this_src = np.transpose(self.dhaso.GF_FD[:,nbl:-nbl,nbl:-nbl],[0,2,1])/self.amp_fac_dvto
    #
    #         # STEP 2: compute distributed source
    #         gstars = GF_this_src * self.distribs_true
    #         dist_src_FD = self.pss_dvto[:gstars.shape[0],None,None] * gstars
    #         dist_src_TD = np.fft.irfft(dist_src_FD, axis=0)
    #
    #         # STEP 3: solve with distributed source
    #         print("Shape of distributed src before reshaping: ", dist_src_TD.shape)
    #         dist_src = dist_src_TD.reshape(self.pss_dvto.size, self.src_locs_dvto.shape[0])
    #         print("Shape of distributed src: ", dist_src.shape)
    #         # print("Source location for devito solver (distributed source): ", self.src_locs_dvto)
    #         self.dhaso.solve(self.par_dvto, dist_src, self.src_locs_dvto, self.rec_locs_dvto, False)
    #
    #         self.dhaso.resample(self.deltat*1e3)
    #         print("Shape of dhaso.data: ", self.dhaso.rec_data_resamp.shape)
    #         self.obscross_tensor[2,2,:,k,:] = np.fft.fftshift(self.dhaso.rec_data_resamp.data/self.amp_fac_dvto) * dx**2
    #         # self.obscross_tensor[2,2,:,k,:] = np.fft.fftshift(self.dhaso.rec_data.data[:250,:]/self.amp_fac_dvto) * dx**2
    #
    #     self.obscross_tensor *= self.ft_fac
    #
    #     if dat_type=='pre':
    #         self.syncross = self.syncross_tensor[2,2,...]
    #     if dat_type=='obs':
    #         self.obscross = self.obscross_tensor[2,2,...]

    #######################################################################################################################

    def make_measurement(self):

        print("In function make_measurement...")

        self.weightpos = np.zeros((self.nom, self.nrecs, self.nrecs))
        self.weightneg = np.zeros((self.nom, self.nrecs, self.nrecs))
        self.synamp_pos = np.zeros((self.nrecs, self.nrecs))
        self.synamp_neg = np.zeros((self.nrecs, self.nrecs))
        self.obsamp_pos = np.zeros((self.nrecs, self.nrecs))
        self.obsamp_neg = np.zeros((self.nrecs, self.nrecs))

        # initscal = np.zeros((self.nrecs, self.nrecs))

        self.negl = np.zeros((self.nrecs, self.nrecs), dtype='int')
        self.negr = np.zeros((self.nrecs, self.nrecs), dtype='int')
        self.posl = np.zeros((self.nrecs, self.nrecs), dtype='int')
        self.posr = np.zeros((self.nrecs, self.nrecs), dtype='int')

        if config.ext_data:
        # EXTERNAL (REAL) DATA CASE
            lefw = self.meas_win[0]
            rigw = self.meas_win[1]
        else:
        # INTERNAL (SYNTHETIC) DATA CASE
            lefw = -4.0 #-1.0 #-0.25
            rigw = +4.0 #1.0 #+0.25

        count_zvlt=0

        for k in range(self.nrecs):
            for j in np.delete(np.arange(self.nrecs),k):

                if self.reald:
                # REAL/EXTERNAL DATA CASE
                	lef = max(0,self.dist_rp_grid[j,k]/self.c + lefw) # left boundary of window (seconds)
                	rig = self.dist_rp_grid[j,k]/self.c + rigw # right boundary of window (seconds)

                	#lef = self.dist_rp_grid[j,k]/cfast # left boundary of window (seconds)
                	#rig = self.dist_rp_grid[j,k]/cslow # right boundary of window (seconds)

                	self.negl[j,k] = np.searchsorted(self.t,-rig)
                	self.negr[j,k] = np.searchsorted(self.t,-lef)
                	self.posl[j,k] = np.searchsorted(self.t,lef)
                	self.posr[j,k] = np.searchsorted(self.t,rig)
                else:
                # SYNTHETIC/INTERNAL DATA CASE
                	# Simple windows suitable for synthetic data:
                	# 1. Entire cross-correlation - [0:self.nom]
                	# 2. Entire negative branch - [0:index of (sample 0)]
                	# 3. Entire positive branch - [1 + index of (sample 0):self.nom]

                	is0 = np.searchsorted(self.t,0)
                	self.negl[j,k] = 0
                	self.negr[j,k] = is0
                	self.posl[j,k] = is0 + 1
                	self.posr[j,k] = self.nom

                # the chosen windows (positive & negative side) should be of non-zero length, otherwise
                # the windowed cross-correlation energy, which divides the weight function, will be 0.
                # The windows can be zero-length if the arrival time for given station pair lies outside
                # the modelled time range (depending on wavespeed obviously).

                try:
                    assert self.negr[j,k]>0 and self.posl[j,k]<self.nom
                except AssertionError:
                    print("Problem with stations ", j, k)
                    raise SystemExit("Aborted. The chosen window for computing cross-corrrelation energy \
                    	 lies outside the modelled time range")

                # print("Negative side window indices: ", self.negl[j,k], self.negr[j,k])
                # print("Positive side window indices: ", self.posl[j,k], self.posr[j,k])

                # now make the measurements

                # print("making measurement for receivers ", j,k)

                self.weightpos[self.posl[j,k]:self.posr[j,k], j, k] = self.syncross[self.posl[j,k]:self.posr[j,k], j, k]
                self.weightneg[self.negl[j,k]:self.negr[j,k], j, k] = self.syncross[self.negl[j,k]:self.negr[j,k], j, k]

                self.synamp_pos[j,k] = np.sqrt(np.sum(self.weightpos[:,j,k]**2))
                #  Computing eq. 24 (numerator only), positive branch
                self.synamp_neg[j,k] = np.sqrt(np.sum(self.weightneg[:,j,k]**2))
                #  computing eq. 24 (numerator only), negative branch

                self.obsamp_pos[j,k] = np.sqrt(np.sum(self.obscross[self.posl[j,k]:self.posr[j,k],j,k]**2))#*self.deltat)
                self.obsamp_neg[j,k] = np.sqrt(np.sum(self.obscross[self.negl[j,k]:self.negr[j,k],j,k]**2))#*self.deltat)

                with np.errstate(invalid='raise'):
                    try:
                        self.weightpos[:,j,k] /= self.synamp_pos[j,k]**2
                        self.weightneg[:,j,k] /= self.synamp_neg[j,k]**2
                    except FloatingPointError as e :
                        # this means some non-diagonal elements of self.synamp_pos or self.synamp_neg, are zero
                        # this should only be due to missing data, which is only possible in the external-data scenario
                        errargs_p=np.argwhere(self.synamp_pos==0)
                        errargs_n=np.argwhere(self.synamp_neg==0)
                        if (not config.ext_data) and (not np.all(errargs_p[:,0]==errargs_p[:,1]) or not np.all(errargs_n[:,0]==errargs_n[:,1])) :
                        # in the internal data case, non-diagonal elements of self.synamp_pos or self.synamp_neg, should never be zero
                            print("RED FLAG!!!: ", e) #, errargs_p, errargs_n)
                            print(self.syncross[self.posl[j,k]:self.posr[j,k], j, k])
                            raise SystemExit("Problem with non-diagonal elements of measurement matrices")
                        elif config.ext_data:
                            # print(j,k)
                            # print(self.missing_pairs[count_zvlt])
                            # print(count_zvlt)
                            self.weightpos[:,j,k] = 0.0
                            self.weightneg[:,j,k] = 0.0
                            count_zvlt+=1

        if config.ext_data:
            try:
                assert count_zvlt == 2*self.npfwdm
            except AssertionError:
                print(count_zvlt)
                raise SystemExit("Problem with non-diagonal elements of measurement matrices")

    #######################################################################################################################

    def diffkernel(self, alpha, beta):
    # Computing source kernels for positive and negative branches

        # ARJUN: multiplication by self.deltat is only required here if it is also used in computation of synamp, obsamp
        ccpos = np.fft.fft(np.fft.ifftshift(self.weightpos[:,alpha,beta]))
        ccneg = np.fft.fft(np.fft.ifftshift(self.weightneg[:,alpha,beta]))

        con = 1/(2*np.pi)

        if lat_homo_inv:
        # Green function available

            bsx, bex, bsy, bey = config.box_indices_largerbox_src( [self.rlocsx[alpha], self.rlocsx[beta]], [self.rlocsy[alpha], self.rlocsy[beta]])

            nxst1, nxst2 = bsx
            nyst1, nyst2 = bsy
            nxfin1, nxfin2 = bex
            nyfin1, nyfin2 = bey

            GrecA = self.Green[dc_xyz[self.comp_src],dc_rtz[getattr(self, use_p[sORe(modelling_inv)])],nyst1:nyfin1,nxst1:nxfin1,:]
            GrecB = self.Green[dc_xyz[self.comp_src],dc_rtz[getattr(self, use_q[sORe(modelling_inv)])],nyst2:nyfin2,nxst2:nxfin2,:]

            fgreen = np.conj(GrecA[...,1:self.nom_nneg]) * GrecB[...,1:self.nom_nneg]

            kp = 2 * (ccpos[1:self.nom_nneg] * fgreen * self.pss[1:self.nom_nneg]).real * con
            kn = 2 * (ccneg[1:self.nom_nneg] * fgreen * self.pss[1:self.nom_nneg]).real * con

        else:
        # Green function NOT available

            solA = np.transpose(self.numsol_pt_src[alpha,...], [1,2,0])
            solB = np.transpose(self.numsol_pt_src[beta,...], [1,2,0])

            fsol = np.conj(solA[...,1:self.nom_nneg]) * solB[...,1:self.nom_nneg]

            kp = 2 * (ccpos[1:self.nom_nneg] * fsol).real * con
            kn = 2 * (ccneg[1:self.nom_nneg] * fsol).real * con

        # kernpos = np.sum(kp, axis=2)
        # kernneg = np.sum(kn, axis=2)

        kernpos = spi.simps(kp,None,dx=self.dom,axis=2)
        kernneg = spi.simps(kn,None,dx=self.dom,axis=2)

        norm_kernpos = np.sum(kernpos*self.distribs_inv) * dx**2
        norm_kernneg = np.sum(kernneg*self.distribs_inv) * dx**2
        # kernel normalization, eq. 29

        if norm_kernpos < 0.85 or norm_kernneg < 0.85 or norm_kernpos > 1.15 or norm_kernneg > 1.15:
            if norm_kernpos==0 and norm_kernneg==0:
            # happens in case of missing data
                pass
            else:
                # raise SystemExit("Problem with normalization of source kernel for receivers %d-%d.\
                #  Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg))
                pass

        return kernpos, kernneg

############################################ Main program ###########################################################

if __name__ == '__main__':

    #********************************************** Load custom modules ****************************************************
    # import SW1D_earthsr.utils_pre_code as u0
    import read_velocity_models as u0
    import cctomo_utils2 as u2
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

    u2.post_run(0, sig_att, 0, oica=kcao)

    #********************************************* Make plots ***************************************************

    def see_individual_skernels():

    	lti=np.tril_indices(kcao.nrecs,k=-1)
    	# lower triangular indices in numpy's default ordering
    	ise=np.argsort(lti[1], kind='mergesort')
    	r=lti[0][ise]
    	c=lti[1][ise]
    	cc_pdist=kcao.dist_rp_grid[(r,c)]
    	# now we have picked out lower triangular elements of kcao.dist_rp_grid in the order that
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
    print(kcao.dist_rp_grid)

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
