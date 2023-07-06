import os
import numpy as np
import scipy.signal as ss
import scipy.optimize as sop
import config_file as config

# global variables
nrecs_chosen = config.nrecs
reald = config.ext_data
dxy = config.dom_geom.dx
ngpmb = config.dom_geom.ngp_box

if reald:
	import obspy.core as oc

###############################################################################################################

class cc_data:

	class Read:

		def __init__(cc_data, cst_id, cst_no, inp_loc, data_format, new_deltat, fexten):

			FuncDic={'binary_archive_python': cc_data.archive, 'individual_files_ccpairs': cc_data.dir_ifccp}

			global chosen_st_id, chosen_st_no
			global si, reclen, fpband, ccl, data_list_form, d_form
			global primary, secondary, allstored

			chosen_st_id = cst_id
			chosen_st_no = cst_no

			d_form = data_format
			cc_data.nsi = new_deltat if (not new_deltat is None) else None

			FuncDic[data_format](inp_loc, fexten)

			si=cc_data.sami
			fpband=cc_data.fpb
			ccl=cc_data.cclags
			data_list_form=cc_data.cookie

			try:
				primary = cc_data.PRIMARY
				secondary = cc_data.SECONDARY
			except AttributeError:
				assert data_format=='binary_archive_python'
				primary=None
				secondary=None

			try:
				allstored=cc_data.recarray
			except AttributeError:
				assert data_format=='individual_files_ccpairs'
				allstored=None

			reclen=2*cc_data.cclags[-1]

			print("\n\nSignal parameters of read data- ")
			print("Number of samples: ", cc_data.cclags.size)
			print("Sample spacing (time domain): ", si)

		#---------------------------------------------------------------------------------------------------------

		def dir_ifccp(cc_data, inp_dir, f_exten):

			names_list=os.listdir(inp_dir)
			filenames=[n for n in names_list if n.endswith(f_exten)]
			cc_data.SECONDARY = []
			cc_data.PRIMARY = []
			Dat = []
			b_time = []
			e_time = []
			si_list = []
			np_list = []

			for i,f in enumerate(filenames):
				file = os.path.join(inp_dir,f)

				if (oc.read(file)[0].stats.sac.kstnm in chosen_st_id) and (oc.read(file)[0].stats.sac.kevnm in chosen_st_id):
				# read only those files which correspond to selected stations

					if cc_data.nsi is None:
						trace = oc.read(file)
						data = trace[0].data
						SI = trace[0].stats.delta
						NP = trace[0].stats.npts
					else:
						trace = oc.read(file)
						NP_calc = (trace[0].stats.sac.e - trace[0].stats.sac.b) / cc_data.nsi
						## take care of float errors
						if np.abs(NP_calc - np.round(NP_calc)) < trace[0].stats.delta:
							NP = int(np.round(NP_calc))
						else:
							NP = int(NP_calc)
						## ensure new number of points stays even or odd
						if trace[0].stats.npts % 2 !=0 :
							# print(trace[0].stats.sac.e, trace[0].stats.sac.b, trace[0].stats.sac.e - trace[0].stats.sac.b)
							NP += 1

						data = ss.resample_poly(trace[0].data, up=NP, down=trace[0].stats.npts)
						# SI = trace[0].stats.delta*trace[0].stats.npts/NP
						SI = cc_data.nsi

					#data = trace[0].data
					#SI = trace[0].stats.delta
					#NP = trace[0].stats.npts
					# NB: npts and delta from SAC headers must be read as above, NOT with ".stats.sac" as below
					s = trace[0].stats.sac.kstnm
					e = trace[0].stats.sac.kevnm
					b_t = trace[0].stats.sac.b
					e_t = trace[0].stats.sac.e

					cc_data.PRIMARY.append(e)
					cc_data.SECONDARY.append(s)
					Dat.append(data)
					b_time.append(b_t)
					e_time.append(e_t)
					si_list.append(SI)
					np_list.append(NP)

			if np.unique(si_list).size == np.unique(b_time).size == np.unique(e_time).size == 1:
				cc_data.sami = np.unique(si_list)[0]
				cc_data.npts = np.unique(np_list)[0]
				# ************ Use either one of the following, whichever works for the data in question ********
				# cc_data.cclags=np.arange(np.unique(b_time)[0]+cc_data.sami,np.unique(e_time)[0]+cc_data.sami,cc_data.sami)
				cc_data.cclags=np.linspace(np.unique(b_time)[0],np.unique(e_time)[0],cc_data.npts)
				# ***********************************************************************************************
				cc_data.cookie=Dat
				freq_info = input("Frequency band of the processed cross-correlations (lf, hf, alpha (Tukey)): ")
				cc_data.fpb = [float(j) for j in freq_info.split()]
			else:
			    raise SystemExit("Could not find data files OR, could not find unique values in the data, for one or more of the following quantities: si, b_time, e_time")

		#---------------------------------------------------------------------------------------------------------

		def archive(cc_data, inp_file, f_exten):

			# NB: only DECIMATION (downsampling) of data implemented in this case.

			assert "npz" in f_exten

			loaded = np.load(inp_file)
			print("Reading ", inp_file)

			try:
			    cc_data.recarray=loaded['reclist']
			    cc_data.fpb=loaded['fpband']
			    try:
			        sami_orig=loaded['si'][0]
			    except IndexError:
			        sami_orig=loaded['si']
			    cclags_orig=loaded['cclags']
			    cookie_entire=loaded['cookie']
			except KeyError:
			    raise SystemExit("Problem reading %s" %(inp_file))

			cclags_orig=np.array([float("%.2f" %x) for x in cclags_orig])
			# making sure that ccl does not contain elements with more significant digits than are allowed by the sampling interval

			if cc_data.nsi is None:
				cc_data.sami = sami_orig
				cc_data.cclags = cclags_orig
				cookie_resamp = cookie_entire
			else:
				cc_data.sami = cc_data.nsi
				cookie_resamp = ss.decimate(cookie_entire, int(cc_data.nsi/sami_orig), axis=0)
				cc_data.cclags = np.linspace(cclags_orig[0], cclags_orig[-1], cookie_resamp.shape[0])

			# read data corresponding to selected stations
			cc_data.cookie = []
			nrtotal=cc_data.recarray.size

			for b,brec in enumerate(chosen_st_no[:-1]):
				rbefore=list(cc_data.recarray).index(brec)
				pbefore=int(rbefore*(2*(nrtotal-1)-(rbefore-1))/2)	# sum of terms of an AP with d=-1
				urecs=chosen_st_no[b+1:]
				crbr = list(map(lambda x: pbefore - 1 + (x-brec), urecs))
				# crbr is Columns_Relevant_to_Base_Receiver
				if b>0:
					assert len(crbr)==prev-1

				for c in crbr:
					cc_data.cookie.append(cookie_resamp[:,c])
				prev=len(crbr)

			nc2 = int((len(chosen_st_no)*(len(chosen_st_no)-1))/2)
			assert len(cc_data.cookie)==nc2

	#*******************************************************************************************************

	class MatrixForm:

		def __init__(MF):

			global data_matrix_pairs

			try:
				assert nrecs_chosen==len(chosen_st_no)
				assert nrecs_chosen==len(chosen_st_id)
			except AssertionError:
				raise SystemExit("Problem identifying chosen station subset")


			FuncDic={'binary_archive_python': MF.archive, 'individual_files_ccpairs': MF.dir_ifccp}

			MF.selec_data=np.zeros((ccl.size, nrecs_chosen, nrecs_chosen))

			FuncDic[d_form](nrecs_chosen)

			data_matrix_pairs = MF.selec_data

		#---------------------------------------------------------------------------------------------------------

		def dir_ifccp(MF, n):

			nc2 = int(n*(n-1)/2)

			try:
				assert len(data_list_form) == nc2
				MF.nmissing = 0
			except AssertionError:
				if len(data_list_form) > nc2:
					raise SystemExit("\nFound more cc-files (%d) than are allowed (%d) by no. of station selected! Please check.\n" %(len(data_list_form), nc2))
				else:
					print("\nWARNING: Some cc-pairs are missing. Only %d available out of %d\n" %(len(data_list_form), nc2))
					mdud=input("Confirm 'y' to continue, any other key to exit: ")
					if not (mdud == 'y' or mdud == 'Y'):
						print(mdud, len(mdud))
						raise SystemExit("Aborted")
					else:
						MF.nmissing = nc2 - len(data_list_form)

			cc_pairs=zip(primary,secondary)

			file_counter=0
			for p,ccp in enumerate(cc_pairs):
				file_counter+=1
				row=chosen_st_id.index(ccp[0])
				col=chosen_st_id.index(ccp[1])
				print("cc-file number %d, for stations: %s %s (%d %d)" %(file_counter, ccp[0],ccp[1],row,col))
				# for consistency with h13, the LOWER TRIANGULAR part of the cc-matrix MUST be filled in.
				# for completeness however, we fill in the entire matrix (except the main diagonal, which is 0).
				if row>col:
					# lower-triangular part
					MF.selec_data[:,row,col]=data_list_form[p]
					# upper-triangular part
					MF.selec_data[:,col,row]=np.flip(data_list_form[p])
				elif row<col:
					# upper-triangular part
					MF.selec_data[:,row,col]=data_list_form[p]
					# lower-triangular part
					MF.selec_data[:,col,row]=np.flip(data_list_form[p])
				elif row==col:
					raise SystemExit("\nProblem building cc-matrix: encountered same primary and secondary station.\n")

			MF.mark_avail_data = np.sum(np.abs(MF.selec_data), axis=0)
			# print("DATA AVAILABILITY CHECK!")
			# print(MF.mark_avail_data)
			present = np.nonzero(np.tril(MF.mark_avail_data))
			assert present[0].size == len(data_list_form)

		#---------------------------------------------------------------------------------------------------------

		def archive(MF, n):

			MF.nmissing = 0
			# NO MISSING DATA in this case, it is assumed.

			nrtotal=allstored.size

			# NB: in h13, the synthetic ccs computed directly are those corresponding to the LOWER TRIANGULAR part
			# of the cross-correlation matrix, i.e. Cjk is computed, where j>k. In case of the archived data, what is
			# stored is the UPPER TRIANGULAR part, i.e. Ckj is stored. These two are flipped versions of each other,
			# so we need to flip the stored data when filling in the lower triangular part of the matrix.

			counter=0

			for j in range(MF.selec_data.shape[1]):
				for k in range(j+1,MF.selec_data.shape[1]):
					MF.selec_data[:,j,k] = data_list_form[counter]
					# upper triangular part, filled in for completeness
					MF.selec_data[:,k,j] = np.flipud(data_list_form[counter])
					# lower triangular part, necessary for compatibility with h13
					counter+=1

			MF.mark_avail_data = np.sum(np.abs(MF.selec_data), axis=0)

		#********************************************************************************************************

	class Process:

		def __init__(self, act_dist_rp, wsp, num_missing):

			self.dt = si
			self.ccl_used = ccl
			self.nsam = data_list_form[0].shape[0]
			self.use_data = data_matrix_pairs
			self.actdrp = act_dist_rp

			self.SNR_and_taper(wsp, num_missing)

		#---------------------------------------------------------------------------------------------------------

		def SNR_and_taper(self, wspeed, nmissing):

			self.lefw = -4.0
			self.rigw = +4.0
			tstart = self.actdrp/wspeed + self.lefw
			tstart[tstart<0] = 0.
			tend = self.actdrp/wspeed + self.rigw

			tap_mid=False
			# If True, the complete cc is tapered not only at the ends but also in the middle. This is
			# achieved by individually tapering the positive and negative branches on the "start" side.
			# In DHG_2019, we kept this parameter set to False, meaning that the entire cc is
			# treated as one waveform, not separated into positive and negative branches.

			pb_iws=np.searchsorted(self.ccl_used,tstart)
			pb_iwe=np.searchsorted(self.ccl_used,tend)
			nb_iws=np.searchsorted(self.ccl_used,-tend)-1
			nb_iwe=np.searchsorted(self.ccl_used,-tstart)-1
			# p/nb_iws/e stands for positive/negative_branch_index_of_window_start/end

			self.snr = np.eye(nrecs_chosen) - 1.0 # so that diagonal is 0, other terms are non-zero

			hnsam = int((self.nsam)/2) if (self.nsam)%2==0 else int(((self.nsam)-1)/2)
			# hnsam -> half_the_number_of_samples
			cchlen = reclen/2
			# cchlen -> cross-correlation_half_length
			if tap_mid:
				tap_portion=np.round(tstart/cchlen,2)
			else:
				tap_portion=np.round((cchlen - tend)/reclen,2)

			cmissing=0

			for i in range(nrecs_chosen-1):
				for j in range(i+1,nrecs_chosen):
					#****************** first compute the S/N ratio ******************
					ud=self.use_data[:,j,i]
					out_win=np.copy(ud)
					ns_ow = hnsam - (pb_iwe[j,i] + 1 - pb_iws[j,i])
					# ns_ow -> number-of_samples_outside_window

					# positive branch window
					pbwin=ud[pb_iws[j,i]:pb_iwe[j,i]+1]
					out_win[pb_iws[j,i]:pb_iwe[j,i]+1]=0.0
					egy_win_pb=np.mean(np.square(pbwin))
					egy_owin_pb=np.sum(np.square(out_win[hnsam+1:]))/(ns_ow)
					# print(egy_win_pb,egy_owin_pb)
					with np.errstate(invalid='raise'):
						try:
							self.snr[j,i] = egy_win_pb/egy_owin_pb
						except FloatingPointError as e:
							# this should happen only in case of MISSING DATA
							self.snr[j,i] = np.nan
							cmissing+=1

					# negative branch window
					nbwin=ud[nb_iws[j,i]:nb_iwe[j,i]+1]
					out_win[nb_iws[j,i]:nb_iwe[j,i]+1]=0.0
					egy_win_nb=np.mean(np.square(nbwin))
					egy_owin_nb=np.sum(np.square(out_win[:hnsam]))/(ns_ow)
					with np.errstate(invalid='raise'):
					# negative branch of [p,q] is positive branch of [q,p] cross-correlation
						try:
							self.snr[i,j] = egy_win_nb/egy_owin_nb
						except FloatingPointError as e:
							# this should happen only in case of MISSING DATA
							self.snr[i,j] = np.nan
							cmissing+=1

					#**************** then do the tapering ***************************
					if reald:
						# NB: obspy operations are performed in-place. So when you taper a
						# trace, even the NumPy array which forms the tr.data gets tapered!!!
						if tap_mid:
						# tapering the positive and negative branches individually
							branches = np.array_split(ud,2)
							tr_nb=oc.trace.Trace(data=branches[0], header={'delta': self.dt})
							tr_pb=oc.trace.Trace(data=branches[1], header={'delta': self.dt})
							tr_nb.taper(max_percentage=tap_portion[j,i],side='right')
							tr_pb.taper(max_percentage=tap_portion[j,i],side='left')

						tr=oc.trace.Trace(data=ud, header={'delta': self.dt})
						tr.taper(max_percentage=tap_portion[j,i])

			try:
				assert cmissing == 2*nmissing
			except AssertionError:
				print("From u2.Process:")
				print(cmissing, nmissing)
				raise SystemExit("Problem with missing data when computing SNR values.")

	#***************************************************************************************************************

	class Errors:

		def __init__(self, metadata):

			# self.DelE = np.zeros((nrecs_chosen,nrecs_chosen))
			# in h13 we only need one (triangular) half of this matrix to cover ALL receiver pairs
			# However, we fill both halves here to account for the positive and negative branches
			# lower triangular half of DelE -> \Delta E for positive branch of h13-relevant waveform
			# upper triangular half of DelE -> \Delta E for negative branch of h13-relevant waveform

			self.secdata = metadata
			self.snr_error()

		#---------------------------------------------------------------------------

		def snr_error(self):

			snr_mat = self.secdata

			#***** CHECK that the 'nan' entries in the matrix are symmetrical

			self.esnrpd_ltpb = np.zeros((nrecs_chosen,nrecs_chosen))
			# self.esnrpd -> error(due to)_SNR_(as a)_percentage_(of)_data
			# ltpb -> lower_triangle_positive_branch
			# (it is implied that the upper triangle of the matrix is for the negative branch)

			self.esnrpd_ltpb[np.where(snr_mat<2)]=0.8
			self.esnrpd_ltpb[np.where((snr_mat>2) & (snr_mat<3))]=0.5
			self.esnrpd_ltpb[np.where(snr_mat>3)]=0.05

		#---------------------------------------------------------------------------
		#
		# def grid_error(self):
		#
		#     # ********* Errors Part 3: position error due to relocation of receivers to grid points
		#     origdist_rp = self.secdata[0]
		#     deltapos = np.square(origdist_rp - self.dist_rp)

##########################################################################################

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

        # fhzp=freq_array[freq_array>=0]
        # fhzn=freq_array[freq_array<0]
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
        # dummy_egy_funcf = (obs_aspec_mean)**2/(nom)

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