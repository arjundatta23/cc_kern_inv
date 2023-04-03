#!/usr/bin/python

# General purpose modules
import os
import sys
import numpy as np
import scipy.special as ssp
import scipy.integrate as spi
import matplotlib.pyplot as plt

sys.path.append('../modules_common')
# path to the modules which are common to the cc source- and structure-inversion codes
if __name__ == '__main__':
    sys.path.append(os.path.expanduser('~/code_general/modules.python'))
    # path to the "SW1D_earthsr" set of modules

# Custom modules (unconditional set1)
import config_file as config

if not __name__ == '__main__':
# get essential variables from main (calling) program
    MPI = sys.modules['__main__'].MPI
    comm = sys.modules['__main__'].comm_out
    rank = sys.modules['__main__'].rank_out
    # p_comp = sys.modules['__main__'].comp_p
    # q_comp = sys.modules['__main__'].comp_q
    if rank==0:
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

if config.ext_data:
    init_amp_scaling = True
    # amplitude scaling usually required in case of real data
else:
    if modelling_tru != modelling_inv:
        # for synthetic tests: may be required in case 'tru_mdlng_type' and 'inv_mdlng_type' are DIFFERENT
        init_amp_scaling = True
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

ntot_omost = omost_fac*(ngpmb-1) + 1

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

        self.data_availability = dobs_info[0]
        self.snr_val =  dobs_info[1]
        self.meas_win = dobs_info[2]

        npairs_total=int(self.nrecs*(self.nrecs-1)/2)
        npairs_use = np.nonzero(np.tril(self.data_availability))[0].size
        self.npfwdm = npairs_total - npairs_use
        # npfwdm -> number of pairs for which data missing

        if not (dobs is None):
        # external data case
            self.reald = True
        else:
        # internal data case
            self.reald = False

        try:
            assert self.reald == config.ext_data
        except AssertionError:
            raise SystemExit("Config paramter 'ext_data' inconsistent with input to core module. This will\
            mess up various settings and break the code. Please rectify.")

        # tdur=self.nom*self.deltat
        # inverse frequency spacing for DFT of time series

        self.rlocsx = rlocsx*dx
        self.rlocsy = rlocsy*dx

        self.dobs_info = dobs_info

        self.num_mparams = 0
        self.ndouble=2*ngpmb-1

        self.comp_scal = 'Z' # this is fixed

        if rank==0:
        # On master node

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! setting up !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            one_by_r = lambda x,k: k/x

            forced=False
            iterate_master=iterate

            syncross_brec = np.empty((self.nom, self.nrecs), dtype='float')
            if self.reald:
            # REAL/EXTERNAL DATA
            	self.obscross = dobs
            	self.obscross_aspec = np.abs(np.fft.fft(dobs,axis=0))
            else:
            # SYNTHETIC/INTERNAL DATA
                # obscross_brec = np.empty((self.nom, self.nrecs), dtype='float')
                # obscross_tensor_brec = np.zeros((3, 3, self.nom, self.nrecs), dtype='complex')
                self.obscross = np.zeros((self.nom, self.nrecs, self.nrecs), dtype='complex')
                if init_amp_scaling:
                    self.obscross_aspec = np.zeros(self.obscross_tensor[0,0,...].shape)

            self.dvar_pos = np.ones(npairs_use)
            self.dvar_neg = np.ones(npairs_use)

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

            tdur=self.nom*self.deltat
            # inverse frequency spacing for DFT of time series
            print("tdur as seen by h13 module: ", tdur)

            umdo = u2.use_modelling_domain(self.rlocsx, self.rlocsy)
            self.dist_rp_grid = umdo.dist_rp
            self.dist_rp_sorted = umdo.alldist_1D

            if iterate_master:
                self.fwd_prep()

            #------------------------------- source distributions and observation errors ------------------------------------------

            sdist_type = {'mg': u1.somod.mult_gauss, 'rg': u1.somod.ringg, 'rgr': u1.somod.rgring, 'gg': u1.somod.gcover}

            self.distribs_start = np.zeros((ngpmb,ngpmb))
            # used to generate the synthetics for inversion; this source distribution IS involved in computing source kernels

            mag1_true=1
            mag1_start=1

            nbasis = {'mg': len(mod_specs['mg']['r0']),
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

            self.mc_start = np.ones(nbasis[param_mod_inv])
            # mc -> model_coefficients
            self.basis = init_basis(nbasis[param_mod_inv],ngpmb)

            num_mparams_master = self.basis.shape[0]

            self.mc_start *= mag1_start

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

                #------------------------------------- End of observation errors ------------------------------------------

            print("Completed initial setup...")
            self.distribs_inv=np.copy(self.distribs_start)
            allit_mc_master = []
            self.allit_misfit = []
            self.allit_syncross = []
            self.flit_indmis_p = []
            self.flit_indmis_n = []
            # variables with names ending in "_inv"  contain values for current (ulimately last) iteration only
            # variables with names starting with "allit_" are lists where each element corresponds to an iteration of the inversion.
            # variables with names starting with "flit_" are two-element lists, storing first (f) and last (l) iteration values only, of certain quantities.

            allit_mc_master.append(np.copy(self.mc_start))
            print("All OK on master proc...")
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! End of setup !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
        # On non-master nodes
            # print("All OK-1 on non-master procs...")
            syncross_brec=None
            obscross_brec=None
            iterate_master=None
            allit_mc_master = None
            num_mparams_master=None
            self.pss = np.empty(self.nom, dtype='float')
            self.distribs_inv = np.empty((ngpmb,ngpmb), dtype='float')
            if omost_fac>2:
            	self.distribs_true = np.empty((2*ngpmb-1,2*ngpmb-1), dtype='float')
            else:
            	self.distribs_true = np.empty((ngpmb,ngpmb), dtype='float')
            if lat_homo_inv:
                self.Green = np.empty((3,self.ndouble,self.ndouble,self.nom_nneg), dtype='complex')
            else:
                self.numsol_pt_src = np.empty((self.nrecs, self.nom_nneg, ngpmb, ngpmb), dtype='complex')
            if not config.ext_data:
                if lat_homo_tru:
                    self.Green_obs = np.empty((3,ntot_omost,ntot_omost,self.nom_nneg), dtype='complex')
                else:
                    self.numsol_obs_pt_src = np.empty((self.nrecs, self.nom_nneg, ngpmb, ngpmb), dtype='complex')
            print("All OK-2 on non-master procs...")

        #--------------------------------------- End of part 1: preliminaries -----------------------------------------------------

        # MPI-distribute the required variables

        iterate = comm.bcast(iterate_master, root=0)
        self.brec = comm.scatter(list(range(self.nrecs)), root=0)
        self.allit_mc = comm.bcast(allit_mc_master, root=0)
        self.num_mparams = comm.bcast(num_mparams_master, root=0)
        comm.Bcast(self.pss, root=0)
        print("SFSG 1")
        comm.Bcast(self.distribs_inv, root=0)
        print("SFSG 2")
        if lat_homo_inv:
            # comm.Bcast([self.Green, MPI.DOUBLE], self.Green, root=0)
            comm.Bcast(self.Green, root=0)
        else:
            pass
        print("SFSG 3")
        if not config.ext_data:
            comm.Bcast(self.distribs_true, root=0)
            if lat_homo_tru:
                comm.Bcast(self.Green_obs, root=0)
            else:
                pass
            print("SFSG 4")

        if rank!=0:
            self.basis=np.empty((self.num_mparams, ngpmb, ngpmb))

        comm.Bcast(self.basis, root=0)

        #--------------------------------- Part 2: compute cross-correlations and measurements --------------------------------------

        itnum=0
        self.brec_obscross = np.zeros((self.nom, self.nrecs))
        self.brec_syncross = np.zeros((self.nom, self.nrecs))

        while iterate:

            self.iter = itnum

            iter_mc = self.allit_mc[-1]
            if use_basis:
                self.distribs_inv = np.einsum('k,klm',iter_mc**2,self.basis)
            else:
                self.distribs_inv = iter_mc.reshape(ngpmb,ngpmb)
            if itnum==0 and rank==0:
                assert np.allclose(self.distribs_inv, self.distribs_start)

            if (not config.ext_data) and (self.iter==0):
            # SYNTHETIC TEST CASE, compute "observed data" synthetically
                self.compute_cc_ptsources_lowertri(lat_homo_tru, 'obs', rank)
                obscross_all = np.asarray(comm.gather(self.brec_obscross, root=0))

            # NB: need to go through this hassle of "asarray" and "transpose" only
            # because comm.Gather (for numppy arrays) is not working

            self.compute_cc_ptsources_lowertri(lat_homo_inv, 'pre', rank)
            syncross_all = np.asarray(comm.gather(self.brec_syncross, root=0))

            if rank==0:
            # ON MASTER NODE
                # print("CHECK 0: ", syncross_all.shape)
                self.syncross = np.transpose(syncross_all, [1,2,0])
                if (not config.ext_data) and (self.iter==0):
                    obscross_clean = np.transpose(obscross_all, [1,2,0])
                    if noise_amp==0:
                        self.obscross = obscross_clean
                    elif noise_amp>0:
                        ano = u1.add_noise_all_cc(obscross_clean, self.signal, noise_amp, noise_band)
                        self.obscross = ano.noisy_signal

                if self.iter==0 and init_amp_scaling:
                    # ------ POTENTIAL AMPLITUDE SCALING -----

                    need_to_scale=True
                    freq_max = np.amax(self.fpos_pow)

                    evdo = u1.egy_vs_dist(self.reald, self.dist_rp_sorted, self.c/freq_max)
                    self.egy_obs, self.oef = evdo.fit_curve_1byr(self.nom, self.obscross_aspec, 'FD', self.dist_rp_grid, self.npfwdm, evdo.sig_dummy)
                    # oef -> observed_energy_fitted

                    while need_to_scale:

                        # ************ compute initial synthetic data energies
                        self.syncross_aspec = np.abs(np.fft.fft(self.syncross, axis=0))
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
                            self.syncross *= np.sqrt(esf)
                            # permanent correction (subsequent iterations)
                            if lat_homo_inv:
                                self.pss *= np.sqrt(esf)
                            else:
                                self.numsol_pt_src *= np.power(esf,1/4)

                    # ------ End: POTENTIAL AMPLITUDE SCALING -----
                else:
                    # Subsequent iterations OR no amplitude scaling
                    pass

                #********* run computationally cheap functions on master ***********
                self.get_cc_uppertri()
                self.make_measurement()
                #****** Finished execution of computationally cheap functions ******

                # inversion related variables
                self.Gmat_pos=np.zeros((npairs_use,self.num_mparams))
                self.Gmat_neg=np.zeros((npairs_use,self.num_mparams))

                self.deltad_pos=np.zeros(npairs_use)
                self.deltad_neg=np.zeros(npairs_use)

                mfit_kern_pos = np.zeros((ngpmb, ngpmb))
                mfit_kern_neg = np.zeros((ngpmb, ngpmb))
                # mfit_kern -> misfit_kernel

                print("Starting computation of source kernels for each receiver pair...")
            else:
            # ON NON-MASTER NODES
                self.weightpos=np.empty((self.nom, self.nrecs, self.nrecs))
                self.weightneg=np.empty((self.nom, self.nrecs, self.nrecs))
                self.synamp_pos=np.empty((self.nrecs, self.nrecs))
                self.synamp_neg=np.empty((self.nrecs, self.nrecs))
                self.obsamp_pos=np.empty((self.nrecs, self.nrecs))
                self.obsamp_neg=np.empty((self.nrecs, self.nrecs))
                mfit_kern_pos = None
                mfit_kern_neg = None

            ################################## End of Part 2: cross-correlations and measurements ####################################

            # variables common to all processors
            ekern_pos = np.zeros((ngpmb, ngpmb))
            ekern_neg = np.zeros((ngpmb, ngpmb))
            Gproc_pos=np.zeros((self.nrecs-self.brec-1,self.num_mparams))
            Gproc_neg=np.zeros((self.nrecs-self.brec-1,self.num_mparams))
            ddproc_pos=np.zeros(self.nrecs-self.brec-1)
            ddproc_neg=np.zeros(self.nrecs-self.brec-1)

            # MPI-distribute the variables required for the rest of the program
            comm.Bcast(self.pss, root=0) # required once again because this may be modified on master, for amplitude scaling
            comm.Bcast(self.weightpos, root=0)
            comm.Bcast(self.weightneg, root=0)
            comm.Bcast(self.synamp_pos, root=0)
            comm.Bcast(self.synamp_neg, root=0)
            comm.Bcast(self.obsamp_pos, root=0)
            comm.Bcast(self.obsamp_neg, root=0)
            print("SFSG 5")

            #################################### Part 3: compute kernels and do inversion ###########################################

            if self.brec+1 < self.nrecs:
                for cp,i in enumerate(range(self.brec+1,self.nrecs)):
                    print("...(source kernel) for receivers %d-%d on processor %d " %(i,self.brec,rank))
                    sker_p, sker_n = self.diffkernel(i,self.brec)
                    # Computing individual source kernels (eq. 15)

                    # build (partially) the G-matrix for inversion
                    if use_basis:
                        kb_prod = sker_p * self.basis * 2*iter_mc[:,None,None]
                        Gproc_pos[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2
                        kb_prod = sker_n * self.basis * 2*iter_mc[:,None,None]
                        Gproc_neg[cp,:] = np.sum(kb_prod, axis=(1,2)) * dx**2
                    else:
                        Gproc_pos[cp,:] = sker_p.flatten()
                        Gproc_neg[cp,:] = sker_n.flatten()

                    with np.errstate(invalid='raise'):
                    # when 'obsamp' and 'synamp' are both zero, that will raise an "invalid" error (division by 0)
                        try:
                            ddproc_pos[cp] = np.log(self.obsamp_pos[i,self.brec]/self.synamp_pos[i,self.brec])
                            ddproc_neg[cp] = np.log(self.obsamp_neg[i,self.brec]/self.synamp_neg[i,self.brec])
                            # print("obsamp_pos and synamp_pos: ", i, self.brec, self.obsamp_pos[i,self.brec], self.synamp_pos[i,self.brec])
                            # print("obsamp_neg and synamp_neg: ", i, self.brec, self.obsamp_neg[i,self.brec], self.synamp_neg[i,self.brec])
                            # Computing event kernels, i.e. eq. 30
                            ekern_pos += sker_p * ddproc_pos[cp]
                            ekern_neg += sker_n * ddproc_neg[cp]
                        except (FloatingPointError, ZeroDivisionError) as e:
                            # this will happen in case of MISSING DATA
                            # ignore and move on; "deltad_pos/neg" for this pair will remain ZERO.
                            # print("Ignoring cc-pairs corresponding to missing data: ", cp)
                            pass
                    # if rank<2:
                    #     print("CHECK sker (pairwise source kernels): ")
                    #     print(rank, cp, np.sum(sker_p))
                    #     print("CHECK deltad_pos (pairwise): ")
                    #     print(ddproc_pos[cp])
            else:
                print("...(source kernel) nothing to do on processor %d" %(rank))

            #************ combine results from all processors
            comm.Reduce(ekern_pos, mfit_kern_pos, root=0)
            comm.Reduce(ekern_neg, mfit_kern_neg, root=0)
            list_Gp = comm.gather(Gproc_pos, root=0)
            list_Gn = comm.gather(Gproc_neg, root=0)
            list_ddp = comm.gather(ddproc_pos, root=0)
            list_ddn = comm.gather(ddproc_neg, root=0)

            if rank==0:
                #**************** inversion on master *****************
                # print("CHECK mfit_kern: ")
                # print(np.sum(mfit_kern_pos), np.sum(mfit_kern_neg))

                Gmat = {'p': self.Gmat_pos, 'n': self.Gmat_neg}
                dd = {'p': self.deltad_pos, 'n': self.deltad_neg}
                list_Gmat = {'p': list_Gp, 'n': list_Gn}
                list_dd = {'p': list_ddp, 'n': list_ddn}

                # need to complete the G-matrix and deltad vector
                for br in Gmat:
                # br -> branch (positive or negative)
                    startrow=0
                    for litem in list_Gmat[br]:
                        if litem.shape[0]>0:
                            paths=litem.shape[0]
                            Gmat[br][startrow:startrow+paths,:] = litem
                            startrow += paths
                    startrow=0
                    for litem in list_dd[br]:
                    	if litem.shape[0]>0:
                    		paths=litem.shape[0]
                    		dd[br][startrow:startrow+paths] = litem
                    		startrow += paths

                # print("CHECK deltad (total): ")
                # print(np.sum(self.deltad_pos), np.sum(self.deltad_neg))

                #*********** things to do on first iteration
                if itnum==0:
                    if self.reald:
                    # complete the calculation of the data errors. NB: we consider two types of error.
                    # The first one is independent of the measurements and is already computed.
                    # The second is defined relative to the measurements, so we must get the absolute values here.

                        dvar_snr_pos = np.square(self.esnrpd_ltpb * self.obsamp_pos)
                        dvar_snr_neg = np.square(np.transpose(self.esnrpd_ltpb) * self.obsamp_neg)

                        # combine different errors
                        dvar_pos = dvar_snr_pos #+ self.dvar_egy_ltpb
                        dvar_neg = dvar_snr_neg #+ np.transpose(self.dvar_egy_ltpb)

                        # account for MISSING DATA: make sure that error (variance) values accorded to missing data are "1", not "0",
                        # so as to avoid problems with division by the variance.
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

                    # regardless of real or synthetic data, store the first-iteration values of certain quantities
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

                print("TOTAL MISFIT: ", total_misfit)
                print(self.allit_misfit)

                if itnum==1:
                	record_flit()
                	if only1_iter:
                	# FORCED STOP FOR TESTING: last misfit stored will correspond to first updated model
                		forced=True; iterate_master=False

                if (itnum>0) and (not forced):
                # determine whether to terminate inversion or iterate further
                	mf_curr = self.allit_misfit[-1]
                	mf_prev = self.allit_misfit[-2]
                	pchange = 100*(mf_prev - mf_curr)/mf_prev
                	if (pchange>0 and pchange<2) or itnum>15:
                		iterate_master=False
                		# inversion terminated.
                		# store quantities corresponding to the final iteration model
                		record_flit()

                if iterate_master:
                    #********* do actual inversion (model update) ***********
                    self.ido = u2.inversion(self.num_mparams)
                    new_mc = self.ido.invert(self.num_mparams, self.basis, self.mc_start, iter_mc,\
                      mfit_kern_pos, mfit_kern_neg, self.Gmat_pos, self.Gmat_neg, self.deltad_pos, self.deltad_neg, self.dvar_pos, self.dvar_neg)
                    allit_mc_master.append(new_mc)
                    # print("CHECK allit_mc (model coefficients): ")
                    # print(allit_mc_master[0])
                    # print(allit_mc_master[-1])

                    print("END OF ITERATION %d" %(itnum))

            # MPI-distribute the variables that change through the iterations
            self.allit_mc = comm.bcast(allit_mc_master, root=0)
            iterate = comm.bcast(iterate_master,root=0)

            itnum +=1

            #*********************** End of loop over iterations *******************

    ##################################### All parts completed, end of function init ########################################

    def fwd_prep(self):

        # this function is NOT parallelized

        if omost_fac==2:
            assert ntot_omost == self.ndouble

        print("Computing distances from origin..")
        r = np.sqrt(dg.gx3**2 + dg.gy3**2)
        r = config.add_epsilon(r)
        # this is done so that r does not contain any zeros; to prevent the Hankel function
        # from blowing up at the origin (r=0)

        if lat_homo_tru or lat_homo_inv:
            """ Laterally HOMOGENEOUS structure model(s) """

            print("Computing Green functions for point source at origin..")
            Gtensor = np.zeros((3,3,self.ndouble,self.ndouble,self.nom_nneg), dtype='complex')
            if not config.ext_data:
                Gtensor_obs = np.zeros((3,3,ntot_omost,ntot_omost,self.nom_nneg), dtype='complex')
                # "Gtensor_obs" may be larger (spatially) than "Gtensor", i.e. the 'observed data' may be computed using
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
                # thresh = (0.01*np.amax(self.pss))/100
                # if self.pss[i] < thresh:
                if not (self.fhz[i] in self.fpos_pow):
                    print("(ignoring FREQUENCY %f Hz)" %(self.fhz[i]))
                else:
                    print("...FREQUENCY %f Hz" %(self.fhz[i]))
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
                                'anal_elas_1D': semian_sol_elas}

                    if elastic:
                        ind0=slice(self.Green.shape[0])
                        ind1=slice(self.Green.shape[1])
                    elif scalar:
                        ind0=2
                        ind1=2

                    Gtensor[ind0,ind1,:,:,i] = solution[modelling_inv][sy:ey,sx:ex]
                    if not config.ext_data:
                        Gtensor_obs[ind0,ind1,:,:,i] = solution[modelling_tru]

            self.Green = Gtensor[dc_xyz[self.comp_src],...]
            # NB: Broadcasting the entire Gtensor to all procsessors results in a Segmentation Fault,
            # so I only broadcast the part with source component 'comp_src'.
            if not config.ext_data:
                self.Green_obs = Gtensor_obs[dc_xyz[self.comp_src],...]

        if not (lat_homo_tru and lat_homo_inv):
            """ Laterally HETEROGENEOUS structure model(s) """

            try:
                assert ('num' in modelling_tru) or ('num' in modelling_inv)
                # modelling type has to be numerical
            except AssertionError:
                raise SystemExit("Laterally heterogeneous structure incompatible with analytical modelling.\
                Please check modelling settings in config file.")

            gs_xy = [dx*1e3, dx*1e3]                            # MUST be in metres
            dom_orig = np.asarray([dg.X[0], dg.Y[0]])           # MUST be in km
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
                        raise SystemExit("Problem with devito solution, program aborted.")

                if not config.ext_data and modelling_tru==mlg3:
                    try:
                        dhaso_obs.solve(src_loc)
                        dhaso_obs.resample()
                        self.numsol_obs_pt_src[k,...] = dhaso_obs.get_FD()[slice(tillf),...]
                        # RHS uses 'rfft', so it may have one extra term in frequency (positive Nyquist term) compared to the LHS - this extra term must be excluded
                        # self.numsol_pt_src[k,...] = self.dhaso.get_FD()
                    except Exception as e:
                        print(e)
                        raise SystemExit("Problem with devito solution, program aborted.")

    #####################################################################################################

    def compute_cc_ptsources_lowertri(self, struc_hom, dat_type, rank):

        # this function is parallelized

        struc_type = 'lhom' if struc_hom else 'lhet'

        brec_syncross_tensor = np.zeros((3, 3, self.nom, self.nrecs), dtype='complex')
        if dat_type=='obs':
            brec_obscross_tensor = np.zeros((3, 3, self.nom, self.nrecs), dtype='complex')

        if self.brec+1 < self.nrecs:
            print("Computing cross-correlations...")

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

            for j in range(self.brec+1,self.nrecs):

                if dat_type=='pre':
                    # compute eq. 11 of Hanasoge (2013)
                    print("...(cc) for receivers %d-%d on processor %d" %(j, self.brec, rank))
                    if struc_type=='lhet':
                        solsyn_j = self.numsol_pt_src[j,...]
                        solsyn_brec = self.numsol_pt_src[brec,...]

                    for p in range(scs,3):
                        for q in range(scs,3):

                            if struc_type=='lhom':

                                Grec_j = self.Green[p,nyst[j]:nyfin[j],nxst[j]:nxfin[j],:]
                                G_brec = self.Green[q,nyst[self.brec]:nyfin[self.brec],nxst[self.brec]:nxfin[self.brec],:]

                                Gsyn_j = np.transpose(Grec_j,[2,0,1])
                                Gsyn_brec = np.transpose(G_brec,[2,0,1])

                                brec_syncross_tensor[p,q,:,j] = u1.compute_cc_1pair_ptsrc(Gsyn_j, Gsyn_brec, self.signal, self.distribs_inv, self.pss)

                            elif struc_type=='lhet':
                                brec_syncross_tensor[p,q,:,j] = u1.compute_cc_1pair_ptsrc(solsyn_j, solsyn_brec, self.signal, self.distribs_inv)

                if dat_type=='obs':
                    print("...cc (obscross) for receivers %d-%d on processor %d" %(j, self.brec, rank))
                    if struc_type=='lhet':
                        solobs_j = self.numsol_obs_pt_src[j,...]
                        solobs_brec = self.numsol_obs_pt_src[brec,...]

                    for p in range(sco,3):
                        for q in range(sco,3):
                            print("Component %d-%d" %(p,q))

                            if struc_type=='lhom':
                                Grec_j = self.Green_obs[p,nyst_obs[j]:nyfin_obs[j],nxst_obs[j]:nxfin_obs[j],:]
                                G_brec = self.Green_obs[q,nyst_obs[self.brec]:nyfin_obs[self.brec],nxst_obs[self.brec]:nxfin_obs[self.brec],:]

                                Gobs_j = np.transpose(Grec_j,[2,0,1])
                                Gobs_brec = np.transpose(G_brec,[2,0,1])

                                brec_obscross_tensor[p,q,:,j] = u1.compute_cc_1pair_ptsrc(Gobs_j, Gobs_brec, self.signal, self.distribs_true, self.pss)
                            elif struc_type=='lhet':
                                brec_obscross_tensor[p,q,:,j] = u1.compute_cc_1pair_ptsrc(solobs_j, solobs_brec, sGelf.signal, self.distribs_true)

            #********** End of loop over receivers ************
            # APPLY ROTATION FROM XYZ TO RTZ COORDS
            #*************************************************

            icmp = lambda m,u: dc_rtz[getattr(self, u[sORe(m)])]

            # select desired cross-correlation component
            if dat_type=='pre':
                brec_syncross_FD = brec_syncross_tensor[icmp(modelling_inv, use_p), icmp(modelling_inv, use_q),...]
            elif dat_type=='obs':
                brec_obscross_FD = brec_obscross_tensor[icmp(modelling_tru, use_p), icmp(modelling_tru, use_q),...]

            # convert to time domain cross-correlations
            if dat_type=='pre':
                self.brec_syncross = np.fft.fftshift(np.fft.ifft(brec_syncross_FD, axis=0).real, axes=(0,))
            elif dat_type=='obs':
                self.brec_obscross = np.fft.fftshift(np.fft.ifft(brec_obscross_FD, axis=0).real, axes=(0,))

        else:
        	print("No cross-correlations to compute for processor %d" %(rank))

    #######################################################################################################################

    def get_cc_uppertri(self):

        for k in range(self.nrecs):
            # [k,j] cross-correlation same as flipped [j,k]
            self.syncross[:,k,k+1:]=np.flipud(self.syncross[:,k+1:,k])
            if not config.ext_data:
                self.obscross[:,k,k+1:]=np.flipud(self.obscross[:,k+1:,k])

    #######################################################################################################################

    def make_measurement(self):

        # this function is NOT parallelized

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
            	# print("...(measurement) for receivers ", j,k)

                if not self.reald:
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
                else:
                # REAL/EXTERNAL DATA CASE
                	lef = max(0,self.dist_rp_grid[j,k]/self.c + lefw) # left boundary of window (seconds)
                	rig = self.dist_rp_grid[j,k]/self.c + rigw # right boundary of window (seconds)

                	#lef = self.dist_rp_grid[j,k]/cfast # left boundary of window (seconds)
                	#rig = self.dist_rp_grid[j,k]/cslow # right boundary of window (seconds)

                	self.negl[j,k] = np.searchsorted(self.t,-rig)
                	self.negr[j,k] = np.searchsorted(self.t,-lef)
                	self.posl[j,k] = np.searchsorted(self.t,lef)
                	self.posr[j,k] = np.searchsorted(self.t,rig)

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
                            print(self.syncross[:, j, k], j, k)
                            print(self.syncross[:, k, j], k, j)
                            # print(self.negl[j,k],self.negr[j,k])
                            # print(self.posl[j,k],self.posr[j,k])
                            raise SystemExit("Problem with non-diagonal elements of measurement matrices")
                        elif config.ext_data:
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

    # this function is PARALLELIZED

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

            GrecA = self.Green[dc_rtz[getattr(self, use_p[sORe(modelling_inv)])],nyst1:nyfin1,nxst1:nxfin1,:]
            GrecB = self.Green[dc_rtz[getattr(self, use_q[sORe(modelling_inv)])],nyst2:nyfin2,nxst2:nxfin2,:]

            f = np.conj(GrecA[...,1:self.nom_nneg]) * GrecB[...,1:self.nom_nneg]

            kp = 2 * (ccpos[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con
            kn = 2 * (ccneg[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con

        else:
        # Green function NOT available
            print("COMING SOON")

        # kernpos = np.sum(kp, axis=2)
        # kernneg = np.sum(kn, axis=2)

        kernpos = spi.simps(kp,None,dx=self.dom,axis=2)
        kernneg = spi.simps(kn,None,dx=self.dom,axis=2)

        norm_kernpos = np.sum(kernpos*self.distribs_inv) * dx**2
        norm_kernneg = np.sum(kernneg*self.distribs_inv) * dx**2
        # kernel normalization, eq. 29

        if norm_kernpos < 0.95 or norm_kernneg < 0.95 or norm_kernpos > 1.05 or norm_kernneg > 1.05:
            if norm_kernpos==0 and norm_kernneg==0:
            # happens in case of missing data
                pass
            else:
                # raise SystemExit("Problem with normalization of source kernel for receivers %d-%d.\
                #  Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg))
                pass

        return kernpos, kernneg

############################## End of class "inv_cc_amp" ###########################################################

if __name__ == '__main__':

    from mpi4py import MPI

    # Initialize MPI process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numproc_out = comm.Get_size()

    #********************************************** Load custom modules ****************************************************
    import SW1D_earthsr.utils_pre_code as u0
    import anseicca_utils2 as u2

    #********************************************** Read input arguments ****************************************************
    if len(sys.argv)>1:
        scalar=False
        elastic=True
        mod1dfile=sys.argv[1]
        efile_ray=sys.argv[2] # eigenfunctions Rayleigh
        dfile_ray=sys.argv[3] # dispersion Rayleigh
        try:
            egn_lov=sys.argv[4] # eigenfunctions Love
            disp_lov=sys.argv[5] # dispersion Love
        except IndexError:
            pass
    else:
        scalar=True
        elastic=False

    p_comp='z'
    q_comp='z'

    #*********************************** Set frequency/time-series characteristics *****************************************

    sig_char = u1.SignalParameters()
    sig_char.dt = 0.2 #0.05
    sig_char.nsam = 250 #401
    sig_char.cf = 0.3 #2.0
    sig_char.lf = None
    sig_char.hf = None
    sig_char.altukey = None

    #********************************************* Set surface geometry ***************************************************

    # numrecs=20
    # # receiver locations (rlocx and rlocy) are specified in number of grid points away from origin (along x,y axes)
    #
    # rlocx = np.array([18, 24, -23, 24, 7, -25, -14, 2, 27, 27, -21, 28, 27, -1, 18, -22, -5, 24, 17, 27])
    # rlocy = np.array([9, -28, 20, 26, 10, 15, 14, -7, 9, -20, 12, -29, -14, -28, -25, 19, 11, -11, 27, -28])

    # numrecs=8 # this should be the size of rlocx and rlocy
    # rlocx=np.array([35, -35, 65, -65, 0, 0, 0, 0])
    # rlocy=np.array([0, 0, 0, 0, 35, -35, 45, -45])

    # numrecs=8
    # rlocx=np.array([6, 12, 40, 90, -6, -12, -40, -90])
    # rlocy=np.array([0, 0, 0, 0, 0, 0, 0, 0])

    numrecs=4
    rlocx=np.array([0, 0, -12, -90])
    rlocy=np.array([12, 60, 0, 0])

    # numrecs=2
    # rlocx=np.array([-70, 90])
    # rlocy=np.array([70, 70])

    hlen_obox = 40
    # half the length of side of outer box OR length of side of inner box (km)
    ngp_ibox = 341
    # number of grid points in inner box, in either direction (half the number of grid points in outer box)
    wspeed = 3.0
    # wavespeed everywhere in model (km/s)

    #-------------------------------- Model reading (elastic case) ----------------------------------
    if rank==0:
        if elastic:
        # read input depth-dependent model and fix/extract necessary parameters
            upreo = u0.model_1D(mod1dfile)
            Zpts_all = upreo.deps_all
            discon_mod = upreo.mod_hif
            upreo.fix_max_depth(Zpts_all[1])
            Zpts_use = upreo.deps_tomax
            # discon_mod_use = discon_mod[discon_mod<=config.dmax]
            print("Layer interfaces in model: ", discon_mod, discon_mod.size)
            print("Depth points to be used in code: ", Zpts_use, Zpts_use.size)
            nzmax = Zpts_use.size
            wspeed=3.0 # NEED TO CHANGE
        else:
            nzmax = None
            wspeed = 3.0
            # wavespeed everywhere in model (km/s)

    #************************************** Run code on each processor *********************************************

    kcao = inv_cc_amp(rlocx, rlocy, sig_att, True, True, None, None)

    u2.post_run(0, sig_att, 0, oica=kcao)

    #*********************************** Final actions on master processor *****************************************

    if rank==0:

        print("Completed program")
        print(kcao.dist_rp_grid)

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
        		ax_sk=fig_sk.add_subplot(1,1,p+1, aspect='equal')
        		cax_sk=ax_sk.pcolor(kcao.gx, kcao.gy, kcao.skers[p]) #, vmin=-0.1, vmax=0.1, cmap=plt.cm.jet)
        		#ax_sk.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'kd', markerfacecolor="None")
        		# use above line to plot all receivers on each subplot OR use below two lines to plot only the
        		# relevant receiver pair on each subplot
        		ax_sk.plot(kcao.dx*rlocx[r[p]], kcao.dx*rlocy[r[p]], 'kd', markerfacecolor="None")
        		ax_sk.plot(kcao.dx*rlocx[c[p]], kcao.dx*rlocy[c[p]], 'kd', markerfacecolor="None")
        		spname = "Distance %.2f km" %(cc_pdist[p])
        		ax_sk.set_title(spname)
        		plt.colorbar(cax_sk,ax=ax_sk)

        #-------------------------------------------------------------------------------------------------------

        fig0=plt.figure()
        ax0=fig0.add_subplot(111)
        ax0.set_title("True model")
        if kcao.omost_fac>2:
        	cax0=ax0.pcolor(kcao.gx2,kcao.gy2,kcao.distribs_true,cmap=plt.cm.jet)
        else:
        	cax0=ax0.pcolor(kcao.gx,kcao.gy,kcao.distribs_true,cmap=plt.cm.jet)
        ax0.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
        plt.colorbar(cax0,ax=ax0)

        if hasattr(kcao, 'iter'):
            fig1=plt.figure()
            ax1=fig1.add_subplot(111)
            ax1.set_title("Positive branch")
            cax1=ax1.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_pos,cmap=plt.cm.jet)
            ax1.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
            fig1.colorbar(cax1)

            fig2=plt.figure()
            ax2=fig2.add_subplot(111)
            ax2.set_title("Negative branch")
            cax2=ax2.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_neg,cmap=plt.cm.jet)
            ax2.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
            fig2.colorbar(cax2)

            fig4=plt.figure()
            ax4=fig4.add_subplot(111)
            ax4.set_title("Inversion result")
            if kcao.omost_fac>2:
            	cax4=ax4.pcolor(kcao.gx2,kcao.gy2,kcao.distribs_inv,cmap=plt.cm.jet)
            else:
            	cax4=ax4.pcolor(kcao.gx,kcao.gy,kcao.distribs_inv,cmap=plt.cm.jet)
            ax4.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
            plt.colorbar(cax4,ax=ax4)

        plt.show()
