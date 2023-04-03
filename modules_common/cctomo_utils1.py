import sys
import numpy as np
import scipy.signal as ss
# import scipy.stats as sst
import scipy.optimize as sop


if not __name__ == '__main__':
# get essential variables from main (calling) program
    # print("From u1:")
    # print(sys.modules['__main__'])
    try:
        dom_geom = sys.modules['__main__'].dg
    except AttributeError:
        dom_geom = sys.modules['__main__'].cg_dom_geom

# global variables
dxy = dom_geom.dx
ngp_mb = dom_geom.ngp_box
ngp = {2: ngp_mb, 3: 2*ngp_mb-1}
xall = {2: dom_geom.X, 3: dom_geom.X2}
yall = {2: dom_geom.Y, 3: dom_geom.Y2}

###############################################################################################################
def read_station_file(st_file):

    """ Format of input file MUST be:
    	COLUMNS(4): <Sl no.> <ID> <Easting(x)> <Northing(y)>
    	ROWS(n + 1): one header line followed by n lines; n is no. of stations/receivers
    """

    cfh=open(st_file,'r')

    cfh.readline()
    entire=cfh.readlines()
    try:
        st_no=list(map(lambda p: int(p.split()[0]), entire))
        st_id=list(map(lambda p: p.split()[1], entire))
        xr=np.array([float(p.split()[2])/1e3 for p in entire])
        yr=np.array([float(p.split()[3])/1e3 for p in entire])
    except IndexError:
        raise SystemExit("Problem reading %s. Check file format." %(st_file))

    cfh.close()
    del cfh
    return st_no, st_id, xr, yr

###############################################################################################################
def compute_cc_1pair_ptsrc(uA, uB, sig_par, src_distrib, powspec=None):

    """ Algorithm using point sources (invokes reciprocity at both receivers) """

    # get necessary signal parameters
    fhz = sig_par.fhz
    nom = sig_par.nsam
    dom = sig_par.domega
    nom_nn = sig_par.n_nn_fsam

    # consider possible asymmetry in frequency samples (happens when the number of samples is even)
    fhzp = len(fhz[fhz>0])
    fhzn = len(fhz[fhz<0])
    ssna = abs(fhzn-fhzp)
    # ssna stands for samples_to_skip_due_to_nyquist_asymmetry
    print("SSNA: ", ssna)

    ans = np.zeros(nom, dtype='complex')

    # compute the cross-correlations for positive frequencies
    fsyn = uA[1:nom_nn,:,:] * np.conj(uB[1:nom_nn,:,:]) * src_distrib
    spa_int = np.sum(fsyn, axis=(1,2)) * (dxy**2)
    ans[1:nom_nn] = spa_int if powspec is None else spa_int * powspec[1:nom_nn]

    # Frequency-domain symmetry: negative frequency coefficients are complex conjugates of flipped positive coefficients
    ans[nom_nn+ssna:] = np.flipud(np.conj(ans[1:nom_nn]))
    # 22 June 2018: BEWARE, the negative Nyquist term gets left out in case ssna>0, i.e. in case self.nom is even.
    # this does matter of course, but it appears to make a very minor difference to the event kernels
    # so I am leaving it for the time being.

    ft_fac = nom*(dom/(2*np.pi))
    ans *= ft_fac

    return ans

###############################################################################################################

class add_noise_all_cc:

    def __init__(self, cc_matrix_clean, sig_att, noise_pcent, noise_fband):

        def bandlimited_noise(sig_shape):

            freq_samp=1.0/deltat
            # np.random.seed(seed=42)
            noise_full = noise_amp * np.random.randn(*sig_shape)
            Wn = noise_fband
            b, a = ss.butter(4, Wn, btype='bandpass',  fs=freq_samp)
            noise_banded = ss.filtfilt(b, a, noise_full, axis=0)
            return noise_banded

        deltat = sig_att.dt

        rms_mat = np.sqrt(np.mean(cc_matrix_clean**2, axis=0))
        assert np.all(np.tril(rms_mat) == rms_mat)
        rms_nz = rms_mat[np.nonzero(np.tril(rms_mat))]
        avg_rms_all_cc = np.mean(rms_nz)

        noise_amp = (noise_pcent * avg_rms_all_cc)/100

        self.noisy_signal = cc_matrix_clean + bandlimited_noise(cc_matrix_clean.shape)

###############################################################################################################

class somod:

    # ringg -> ring_of_gaussians (used by Datta_et_al, 2019)
    # rgring -> radially_gaussian_ring (used by Hanasoge, 2013)

    """ All gaussians are constructed such that 'half-width = 2-sigma' """

    @staticmethod
    def mult_gauss(rcen, specs, fac_rel=2):

        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        xp=rcen[0]
        yp=rcen[1]
        ind = specs['r0'].index((xp,yp))
        ampl = specs['mag'][ind]
        sigma_x = specs['w'][ind][0]/4
        sigma_y = specs['w'][ind][1]/4

        # print("From mult_gauss: ", xp)
        # print("From mult_gauss: ", yp)

        for j in range(ngp[fac_rel]):
            ans[:,j] = np.exp( -( (xall[fac_rel][j] - xp)**2/(sigma_x**2) + (yall[fac_rel] - yp)**2/(sigma_y**2) ) )

        return ans

    #******************************************************************************************

    @staticmethod
    def ringg(theta, specs, fac_rel=2):
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        rad=specs['r']
        sigma=specs['w']/4
        rad_use = {2: rad, 3: 2*rad}

        for j in range(ngp[fac_rel]):
            x0 = rad_use[fac_rel]*np.cos(theta)
            y0 = rad_use[fac_rel]*np.sin(theta)
            ans[:,j] = np.exp( -((xall[fac_rel][j] - x0)**2 + (yall[fac_rel] - y0)**2)/(sigma**2) )

        return ans

    #******************************************************************************************

    @staticmethod
    def rgring(rcen, specs, fac_rel=2):
        #         if mag2 is None:
        #         	ampl = mag1
        #         else:
        #         	#if abs(xall[j])<10:
        #         	if xall[j]>-22 and xall[j]<-15:
        #         		ampl = mag2
        #         	else:
        #         		ampl = mag1
        mag2=None
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        xp = rcen[0]
        yp = rcen[1]
        ind = specs['r0'].index((xp,yp))
        rad = specs['r'][ind]
        ampl = specs['mag'][ind]
        sigma=specs['w'][ind]/4

        for j in range(ngp[fac_rel]):
            r_ib = np.sqrt((xall[fac_rel][j]-xp)**2 + (yall[fac_rel]-yp)**2)
            ans[:,j] = ampl * ( np.exp( -(r_ib-rad)**2/(sigma**2)) )

        return ans

    #******************************************************************************************

    @staticmethod
    def gcover(coord, specs, fac_rel=2):
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        sigma = specs['w']/4
        # xv,yv=np.meshgrid(row_val,row_val)
        x0=coord[0] # xv.flatten()
        y0= coord[1] #.flatten()
        # print('x-pos',x0)
        # print('y-pos',y0)
        for j in range(ngp[fac_rel]):
            ans[:,j] = np.exp(-((xall[fac_rel][j] - x0)**2 + (yall[fac_rel] - y0)**2)/(sigma**2) )
        return ans

    #******************************************************************************************

###############################################################################################################

class egy_vs_dist:

    def __init__(self, reald, dist1D, lam_shortest):

        self.dist1D_sorted = dist1D

        if reald:
        # in the real data case, treat 'very short' distances with caution
            nf_dist = 0.5 * lam_shortest
            # nf_dist -> near_field_distance. Using a very crude estimate: half the shortest wavelength in the data
            sd_ind=np.argwhere(self.dist1D_sorted<nf_dist)
            # sd_ind -> short_distance_indices
            self.sig_dummy = np.ones(self.dist1D_sorted.size)
            self.sig_dummy[sd_ind] = 5
            # NB: self.sig_dummy - deliberately called "dummy" - contains basically the relative weights for the data points, NOT
            # the actual standard deviations. This is reflected in the argument "absolute_sigma=False" to scipy's curve fit.
        else:
            self.sig_dummy = None

    #---------------------------------------------------------------------------

    # def fit_curve_1byr(self, nom, cc, domain, dist2D, nmissing, dummy_sig, zvinfo=None):
    def fit_curve_1byr(self, nom, cc, domain, dist2D, nmissing, dummy_sig):

        #***** define local functions
        domain_dic = {'FD': lambda x: x, 'TD': lambda x: np.abs(np.fft.fft(x,axis=0))}
        one_by_r = lambda x,k: k/x

        #***** calculate cross-correlation energies
        cc_aspec = domain_dic[domain](cc)
        da = dist2D[np.nonzero(np.tril(dist2D))]
        ccegy_funcf = np.square(cc_aspec) # ccegy_funcf -> cc_power_as_a_function_of_frequency
        cc_egy = np.sum(ccegy_funcf,axis=0)/nom
        self.egy_flat = cc_egy[np.nonzero(np.tril(dist2D))] # the matrix is symmetric so it suffices to consider only its lower triangular part

        # print(self.egy_flat)
        # print(self.egy_flat.size)

        # #***** ASSIGN zero values (relevant for synthetics, in case of missing data)
        # if not zvinfo is None:
        #     self.egy_flat[zvinfo]=0
        #
        # #***** STORE zero-value indices (for possible later use), in case of missing data
        # self.zvi = np.argwhere(self.egy_flat==0) if nmissing>0 else None

        #***** CHECK zero values (due to missing data)
        n_total = self.egy_flat.size
        n_finite_egy = np.nonzero(self.egy_flat)[0].size
        assert n_total==self.dist1D_sorted.size
        try:
            assert (n_total - n_finite_egy) == nmissing
        except AssertionError:
            print(n_total, n_finite_egy, nmissing)
            raise SystemExit("Problem with cross-correlation energies - unexpected number of zero values.")

        #***** eliminate zero values
        egy_sorted = self.egy_flat[np.argsort(da)]
        use_energies = egy_sorted[np.nonzero(egy_sorted)]
        use_dist = self.dist1D_sorted[np.nonzero(egy_sorted)]
        if not dummy_sig is None:
            use_sigma = dummy_sig[np.nonzero(egy_sorted)]

        #***** do curve fitting (with zero values excluded)
        try:
            popt, pcov = sop.curve_fit(one_by_r, use_dist, use_energies, sigma=use_sigma, absolute_sigma=False)
        except NameError:
            # 'sigma' value not used in case of synthetic data
            # print(use_dist)
            # print(use_dist.size)
            # print(use_energies)
            popt, pcov = sop.curve_fit(one_by_r, use_dist, use_energies)

        ef = popt[0]/use_dist

        #***** initialize arrays containing final output - use 'nan' to allow for missing data
        self.ef = np.nan*np.ones(n_total)
        self.use_energies = np.nan*np.ones(n_total)

        #***** create final output (with zeros replaced by nans)
        self.ef[np.nonzero(egy_sorted)] = ef
        self.use_energies[np.nonzero(egy_sorted)] = use_energies

        return self.use_energies, self.ef

    #---------------------------------------------------------------------------

    def indices_1D_to_2D(self, dist2D, nmissing):

        """ this function converts the 1-D indices obtained above (for missing data),
        to 2-D indices relevant to the 2-D cc matrix used in the code.
        """

        dummy = np.ones(dist2D.shape)
        dummy2 = np.tril(dummy)
        for j in range(dummy.shape[0]):
            dummy2[j,j]=0

        dummy2[np.nonzero(dummy2)] = self.egy_flat
        iaz = np.argwhere(dummy2== 0)
        final = iaz[iaz[:, 0] > iaz[:, 1]]

        try:
            assert final.shape[0]==nmissing
        except AssertionError:
            raise SystemExit("Problem identifying missing data using the zero-energy criterion.")

        return final

##########################################################################################

class source_spectrum:

    def __init__(self, obscross_TD):

        self.obscross_aspec = np.abs(np.fft.fft(obscross_TD, axis=0))

    #---------------------------------------------------------------------------

    def match_obs_spectra(self, nom, freq_array, nmissing):

        fhzp=freq_array[freq_array>=0]
        fhzn=freq_array[freq_array<0]
        # taking zero on the positive side ensures that both branches are of equal size, because remember that for
        # even number of samples, the positive side is missing the Nyquist term.

        norm_obs_aspec = np.copy(self.obscross_aspec)
        max_each_rp = np.max(self.obscross_aspec, axis=0)

        with np.errstate(invalid='raise'):
            try:
                norm_obs_aspec /= max_each_rp
            except FloatingPointError as e:
                errargs=np.argwhere(max_each_rp==0)
                if not np.all(errargs[:,0]<=errargs[:,1]):
                # this means there are 0-values in the lower-triangular part of 'max_each_rp' - should be due to missing data only.
                    # print(errargs.shape)
                    ltzv=errargs[errargs[:,0]>errargs[:,1]]
                    if ltzv.shape[0] != nmissing:
                        raise SystemExit("Problem with observed amplitude spectra - unexpected zero values.")
                    else:
                        norm_obs_aspec = np.copy(self.obscross_aspec)
                        for eg in errargs:
                            max_each_rp[eg[0],eg[1]] = np.nan
                        norm_obs_aspec /= max_each_rp

        obs_aspec_mean = np.nanmean(norm_obs_aspec,axis=(1,2))
        # NB: this spectrum is useful only for its shape. It is a DUMMY as far as amplitude is concerned.
        dummy_egy_funcf = (obs_aspec_mean)**2/(nom)
        dummy_pow = np.sum(dummy_egy_funcf,axis=0)

        # rvp=sst.skewnorm(a=-5,loc=0.55,scale=0.15)
        # rvn=sst.skewnorm(a=5,loc=-0.55,scale=0.15)

        # rvp=sst.skewnorm(a=-3,loc=0.5,scale=0.13)
        # rvn=sst.skewnorm(a=3,loc=-0.5,scale=0.13)
        #
        # pss = np.concatenate((rvp.pdf(fhzp),rvn.pdf(fhzn)))

        # rvp=sst.powerlognorm(c=2.0,s=0.7,loc=0.4,scale=1.0)
        # rvn=sst.powerlognorm(c=2.0,s=0.7,loc=-4.7,scale=1.0)

        # rvp=sst.powerlognorm(c=5.0,s=0.7,loc=2,scale=1.0)
        # rvn=sst.powerlognorm(c=5.0,s=0.7,loc=-8,scale=1.0)
        #
        # pss = np.concatenate((rvp.pdf(fhzp),np.flip(rvn.pdf(fhzn))))

        pss = obs_aspec_mean
        pss_max = np.amax(pss)

        pow_spec_dB = 10 * np.log10(pss/pss_max)

        try:
            assert pow_spec_dB[0] <= -30
            # Negligible DC-component in spectrum
        except AssertionError:
            print(pss[0], pss_max)
            # raise Exception("From u1.match_obs_spectra: Source (power) spectrum has a significant DC component")

        return obs_aspec_mean, pss

##########################################################################################

class assign_errors:

    def __init__(self, metadata):

        self.secdata = metadata

    #---------------------------------------------------------------------------

    def snr_error(self, dist2D):

        #********* Errors Part 1: error due to SNR
        snr = self.secdata

        #***** CHECK that the 'nan' entries in the matrix are symmetrical

        esnrpd_ltpb = np.zeros((dist2D.shape))
        # esnrpd -> error(due to)_SNR_(as a)_percentage_(of)_data
        # ltpb -> lower_triangle_positive_branch
        # (it is implied that the upper triangle of the matrix is for the negative branch)

        esnrpd_ltpb[np.where(snr<2)]=0.8
        esnrpd_ltpb[np.where((snr>2) & (snr<3))]=0.5
        esnrpd_ltpb[np.where(snr>3)]=0.05

        # esnrpd_ltpb[np.where(snr<1000)]=0.8
        # esnrpd_ltpb[np.where((snr>1000) & (snr<5000))]=0.5
        # esnrpd_ltpb[np.where(snr>5000)]=0.05

        return esnrpd_ltpb

    #---------------------------------------------------------------------------

    # def decay_rate_error(self):
    #
    #     #********* Errors Part 2: error due to energy decay with distance
    #
    #     snr = self.secdata[1]
    #     delA = self.secdata[2]
    #
    #     # ********************************************************************************************************************
    #     # NB: uncertainties in the observations contained in dinfo need to be corrected, because the measurement for
    #     # the kernels involves cc energies computed in a certain window only, whereas the curve fitting above is done using
    #     # the energy of the entire cc branch. This correction can be made using the waveform's S/N ratio, which indirectly
    #     # provides a measure of the contribution of the window of interest, to the total energy of the waveform (branch).
    #     # ********************************************************************************************************************
    #
    #     # refine the error so it applies to the measurement window only
    #     nsr = 1./snr
    #     ScT = 1./(1+nsr) # 1./np.sqrt(1+nsr)
    #     # ScT -> signal_contribution_to_total (energy)
    #     delA *= ScT
    #
    #     # convert to variance
    #     return np.square(delA)
    #
    # #---------------------------------------------------------------------------
    #
    # def grid_error(self):
    #
    #     # ********* Errors Part 3: position error due to relocation of receivers to grid points
    #     origdist_rp = self.secdata[0]
    #     deltapos = np.square(origdist_rp - self.dist_rp)

##########################################################################################
