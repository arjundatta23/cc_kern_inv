#!/usr/bin/python

# Standard modules
import sys
import numpy as np

# get essential info from main (calling) program
try:
	# efile = sys.modules['__main__'].egn_ray
	dfile = sys.modules['__main__'].user_choices['elas_mod_1D']['disp_ray']
except TypeError:
	# this is the 2D-scalar case
	assert sys.modules['__main__'].user_choices['scalar']==True

scalar = sys.modules['__main__'].user_choices['scalar']

# Custom modules
import config_file as config
if not scalar:
    import SW1D_earthsr.read_earthsr_io as reo

################################################################################

nrecs = config.nrecs
dx = config.dom_geom.dx
ngpmb = config.dom_geom.ngp_box
nband = config.syn_data.noise_band

# IMPORTANT: sig_char NOT to be taken from config file because it is overwritten
# in the code wrapper when working with external data!!

################################################################################

class check_settings:

	def __init__(self, char_sig, dist_arr=None):

		self.sig_char = char_sig
		pss = self.sig_char.pow_spec_sources

		pss_pos = pss[:self.sig_char.n_nn_fsam]
		fhz_pos = self.sig_char.fhz[:self.sig_char.n_nn_fsam]

		# Check grid symmetry with respect to origin
		# if (config.dom_geom.box_len/2)%dx != 0: NB: does not work due to vagaries of the modulo function
		q = (config.dom_geom.box_len/2)/dx
		if not np.isclose(q,int(q)):
			raise Exception("Domain size (horizontal plane) incompatible with grid spacing; please check configuration file.")

		if dist_arr is None:
			self.dist_max = 2 * config.rloc[-1]
		else:
			self.dist_max = np.round(np.amax(dist_arr))

		# Check receivers lie within computational domain
		try:
			assert self.dist_max < config.dom_geom.box_len
		except AssertionError:
		    raise Exception("Receivers must lie WITHIN computational domain; please check configuration file.")
		
		# Check consistency of forward modelling choices
		try:
			assert int(config.tru_struc_lat_homo) != config.mtl1_tru
			assert int(config.inv_struc_lat_homo) != config.mtl1_inv
		except AssertionError:
			raise Exception("Laterally heterogeneous structure incompatible with analytical modelling.\
				Laterally homogeneous structure (setting) must not be combined with numerical modelling.\
                Please check modelling settings in Par file.")

		self.fmax = config.sig_thresh_dB(fhz_pos, pss_pos, -30)[-1]

		if scalar:
			self.phasespeed_min = config.scal_mod.wavspeed_scal2D
			self.groupspeed_cen = config.scal_mod.wavspeed_scal2D
		else:
			reoobj = reo.read_disp([dfile],0,0)
			reof = np.array([i for i,j in reoobj.modudisp[0][0]][::-1])
			reou = np.array([j for i,j in reoobj.modudisp[0][0]][::-1])
			reoc = np.array([j for i,j in reoobj.modcdisp[0][0]][::-1])
			# in the above, the lists are reversed because they are sorted in DESCENDING order of frequency
			# whereas np.searchsorted (below), works by default on arrays sorted in ASCENDING order
			try:
				self.groupspeed_cen = reou[np.searchsorted(reof,self.sig_char.cf)]
				self.phasespeed_min = reoc[np.searchsorted(reof,self.fmax)]
			except IndexError:
				raise Exception("Looking for frequency: %f Hz, which is beyond the scope of the input dispersion file" %(self.fmax))

		self.aliasing_temporal()
		self.aliasing_spatial()

	#***************************************************************************

	def aliasing_temporal(self):

		freq_nyq = 1./(2*self.sig_char.dt)

		# stipulate that the peak frequency of the source wavelet should be not be greater than HALF the Nyquist frequency
		if (self.sig_char.cf > freq_nyq/2) or (self.fmax >= freq_nyq):
		    print('%f > %f' %(self.fmax, freq_nyq))
		    raise Exception("Possible TEMPORAL ALIASING with current settings! Please check configuration file.")

		# similarly, bandwidth of noise to be added to synthetic data must not breach the Nyquist
		if not config.ext_data:
			if (nband[0] > nband[1]):
				print('%f > %f' %(nband[0], nband[1]))
				raise Exception("Problem with NOISE BAND (for synthetic data)! Must specify lower, then upper bound of pass band.")
			if (nband[1] >= freq_nyq):
				print('%f > %f' %(nband[1], freq_nyq))
				raise Exception("Problem with NOISE BAND (for synthetic data)! Pass band must not breach the Nyquist frequency.")

	#***************************************************************************

	def aliasing_spatial(self):

		wavlen_min = self.phasespeed_min / self.fmax
		grid_spacing = np.round(dx,3)
		lamda_min_by_4 = np.round((wavlen_min/4),3)
		# stipulate that the highest wavenumber being dealt with is not more than HALF the Nyquist wavenumber
		if grid_spacing > lamda_min_by_4:
		    print("%.3f > %.3f" %(dx, wavlen_min/4))
		    print("Used min phasespeed %f km/s and max frequency %f Hz for testing" %(self.phasespeed_min, self.fmax))
		    raise Exception("Possible SPATIAL ALIASING with current settings! Please check configuration file.")

	#***************************************************************************

	def dc_comp_spectrum(self):

		pss = self.sig_char.pow_spec_sources
		pss_max = np.amax(pss)
		pow_spec_dB = 10 * np.log10(pss/pss_max)

		dc_comp = pow_spec_dB[0]
		if dc_comp > -30:
			print("DC-value in power spectrum is: %.2f dB" %(dc_comp))
			raise Exception("Source (power) spectrum has a significant DC component! Please check configuration file.")

	#***************************************************************************

	def time_series_length(self):

		""" NB: currrently this function only works for 'narrow-band' source spectra (chosen in the config file),
		 	for which 'fsigma' is a relevant parameter
		"""

		tarr = self.dist_max/self.groupspeed_cen
		bandw = 4*self.sig_char.fsigma
		dur_TD = 6*(1./bandw)
		# Source BW approx. 4 sigma, total signal duration - TSD - approx. 6*(1/BW) (or 8*(1/BW) if BW = 5 sigma)
		# so need to check for (main arrival time + TSD/2)
		tlen_branch = (self.sig_char.nsam * self.sig_char.dt)/2
		tmax_cc = tarr + dur_TD/2
		if tmax_cc > tlen_branch:
			print("%.3f > %.3f" %(tmax_cc, tlen_branch))
			raise Exception("Time series not long enough for complete cross-correlation with current settings; please check configuration file.")

################################################################################

class memory_reqt:

	def __init__(self, char_sig, nz):

		self.sig_char = char_sig

		self.nz = nz

		self.mem_thresh = 0.95
		# tolerance for how much system memory we are willing to consume

		self.Garr_storage = 16 * 1e-9 * (3 * 3 * self.sig_char.n_nn_fsam * self.nz * config.dom_geom.Y3.size * config.dom_geom.X3.size)

	#***************************************************************************

	def struc_kern(self, kern_todo): #, kern_elast):

		if kern_todo:
			karr_storage = 8 * 1e-9 * (self.nz * config.dom_geom.Y.size * config.dom_geom.X.size)
			# if kern_elast:
			for_div_vol = 16 * 1e-9 * (3 * self.sig_char.n_nn_fsam * self.nz * config.dom_geom.Y3.size * config.dom_geom.X3.size)
			for_grad_vol = 16 * 1e-9 * (2 * 3 * 3 * self.sig_char.n_nn_fsam * self.nz * config.dom_geom.Y3.size * config.dom_geom.X3.size)

			self.Garr_storage += for_div_vol + for_grad_vol
			karr_storage *= 6
			# there are 6 kernels: rho, lam, mu, rhop, vp, vs
			# else:
			# 	pass
			# 	# only 1 kernel (rho)
		else:
			karr_storage = 0.0

		self.arr_storage = self.Garr_storage + karr_storage

		print("MEMORY REQUIREMENT (array storage only): %f + %f = %f GB" %(self.Garr_storage, karr_storage, self.arr_storage))
		self.decision()

	#***************************************************************************

	def source_kern(self):

		for_mod = 2 * 8 * (ngpmb**2) * 1e-9
		# one for starting model, one for model update
		for_mfit_kern = 2 * for_mod
		# one misfit kernel for each cc branch
		for_wforms = 16 * self.sig_char.nsam * (nrecs**2) * 1e-9
		for_dist = 8 * (nrecs**2) * 1e-9

		self.arr_storage = self.Garr_storage + for_mod + for_mfit_kern + for_wforms + for_dist

		print("MEMORY REQUIREMENT (array storage only): %f GB" %(self.arr_storage))
		self.decision()

	#***************************************************************************

	def decision(self):

		if self.arr_storage > self.mem_thresh * config.SYST_RAM:
		    raise Exception("Memory requirement TOO HIGH: %f > %d %% of %d GB. Check simulation parameters." \
			 %(self.arr_storage, int(self.mem_thresh*100), config.SYST_RAM))
