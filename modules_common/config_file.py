#!/usr/bin/python

# Standard modules
import numpy as np
import scipy.signal as ss
import configparser
cp=configparser.SafeConfigParser()
cp.read('INPUT/Par_file')

try:
	import psutil
	RAM_check=True
except ImportError:
	RAM_check=False

################################################################################
# Start line - DO NOT EDIT OR DELETE
################################################################################

################################################################################
# System information
################################################################################

# Available RAM in GB
if RAM_check:
	SYST_RAM = np.floor(psutil.virtual_memory().total*1e-9)
else:
	# e.g., on a laptop
	SYST_RAM = 64

################################################################################
# Constants
################################################################################

epsilon=cp.getfloat('CONSTANTS', 'epsilon')

################################################################################
# BASICS (for both SOURCE/STRUCTURE inversion)
################################################################################

nrecs=cp.getint('BASICS', 'nrecs')
ext_data=cp.getboolean('BASICS', 'ext_data')
# whether the 'data' used in the code is generated internally (e.g synthetic tests), or taken from an external source (e.g. real data)

# dictionaries
comp_dic_xyz={'x':0, 'y':1, 'z':2}
comp_dic_rtz={'R':0, 'T':1, 'Z':2}
green_tensor_rtz = {0:'RR', 1:'RT', 2:'RZ',
 					3:'TR', 4: 'TT', 5: 'TZ',
					6:'ZR', 7: 'ZT', 8: 'ZZ'}
# green_tensor_ENZ = {0:'EE', 1:'EN', 2:'EZ',
#  					3:'NE', 4: 'NN', 5: 'NZ',
# 					6:'ZE', 7: 'ZN', 8: 'ZZ'}
# comp_dic_ENZ={'E':0, 'N':1, 'Z':2}

#*******************************************************************************
# Geometrical setup
#*******************************************************************************

class DomainGeometry():
	""" defines the geometry of the modelling domain used in the code.
	    User-defined attributes:
	        box_len: size of modelling domain - length of (side of) main box OR half-length of outer box [km],
	        dx: uniform grid spacing in x-y directions [km],
	        zmax: maximum depth to consider, when working with a depth-dependent Earth model [km]
	"""
	# the choice of zmax may be dictated, for example, by the peneteration depth of eigenfunctions
	# of the 1D model (for the mode numbers and frequencies considered).

	def grid(self):
		self.ngp_box = int(self.box_len/self.dx) + 1 # number of grid points along one side of main box

		hlen=self.box_len/2
		self.ngp_hlen=int(hlen/self.dx)

		self.X = np.linspace(-hlen,hlen,self.ngp_box)
		self.Y = np.copy(self.X)

		self.X2 = np.linspace(-self.box_len,self.box_len,2*self.ngp_box-1)
		self.Y2 = np.copy(self.X2)

		hl_omost_box = syn_data.ofac * self.box_len/2.0
		ngp_omost_box = syn_data.ofac*(self.ngp_box-1) + 1

		self.X3=np.linspace(-hl_omost_box, hl_omost_box, ngp_omost_box)
		self.Y3=np.copy(self.X3)

		self.gx, self.gy = np.meshgrid(self.X,self.Y)
		self.gx2, self.gy2 = np.meshgrid(self.X2,self.Y2)
		self.gx3, self.gy3 = np.meshgrid(self.X3,self.Y3)

# ------------------------------------------------------------------------------

dom_geom = DomainGeometry()
dom_geom.dx = cp.getfloat('DOM_GEOM', 'dx')
dom_geom.zmax = cp.getfloat('DOM_GEOM', 'zmax')
dom_geom.box_len = cp.getfloat('DOM_GEOM', 'domain_size')
dom_geom.grid_origx = cp.getint('DOM_GEOM', 'grid_origx')
dom_geom.grid_origy = cp.getint('DOM_GEOM', 'grid_origy')

# ngp_rloc=cp.getfloat('DOM_GEOM', 'rec_loc_grid_points')
ngp_rloc_string = cp.get('DOM_GEOM', 'rec_loc_grid_points')
ngp_rloc_arr = np.array([float(i) for i in ngp_rloc_string.split()])
ngp_rloc = np.sort(ngp_rloc_arr)

rloc = ngp_rloc * dom_geom.dx

################################################################################
# Parameters pertaining to cross-correlation modelling theory
################################################################################

class ccModellingTheory():
	""" sets user choices for inversion (including synthetic tests)
		Attributes:
			src_dir: orientation of ambient noise (point-force) sources
			GTC: component(s) of Green tensor to work with

	"""

ccmt = ccModellingTheory()
ccmt.src_dir = cp.get('CCMT', 'src_dir')
ccmt.GTC = [cp.getint('CCMT', 'cc_component_RTZ')]

################################################################################
# Parameters pertaining to inversion
################################################################################

class InversionChoices():
	""" sets user choices for inversion (including synthetic tests)
		User-defined attributes:
			gamma_inv: damping parameter for regularized least-squares inversion

	"""

invc = InversionChoices()
invc.gamma_inv = cp.getfloat('INVC', 'gamma_inv')

################################################################################
# Parameters pertaining to inversion
################################################################################

class SyntheticData:
	""" sets user choices for synthetic test inversions, where (synthetic) data
	 	is generated internally by forward modelling.
		User-defined attributes:
			ofac: parameter defining the size of the true/test model (same size or larger than inverse modelling domain; see h13 module for details)
			noise_band = bandwith of noise added
			noise_level = amplitude of noise added, as a percentage of signal RMS amplitude, to synthetic data

	"""

syn_data = SyntheticData()
syn_data.ofac = cp.getint('SYN_DATA', 'outer_box_size_factor')
syn_data.noise_level = cp.getfloat('SYN_DATA', 'noise_amp_pcent')
noise_band_string = cp.get('SYN_DATA', 'noise_band')
syn_data.noise_band = [float(i) for i in noise_band_string.split()]

dom_geom.grid() # needs to be done AFTER "syn_data.ofac" is defined

################################################################################
# Parameters for acoustic/scalar wave equation modelling
################################################################################

class ScalarModelling():
	""" defines model parameters required for scalar wave equation modelling.
		User-defined attributes:
			rho_scal2D: density [gm/cc]
			wavspeed_scal2D: wave speed [km/s]
	"""
scal_mod = ScalarModelling()
scal_mod.rho_scal2D = cp.getfloat('SCAL_MOD', 'density')
scal_mod.wavspeed_scal2D = cp.getfloat('SCAL_MOD', 'wavespeed')

#*******************************************************************************
# Signal characteristics
#*******************************************************************************

class SignalParameters():
	""" defines the signal (time-series) characteristics used throughout the code.
	    User-defined attributes:
	        nsam: number of samples,
	        dt: temporal sampling interval [s],
	        cf: center (peak) frequency, in case of a narrow band [Hz]
			lf, hf: low and high cut-off frequencies, in the wide-band case [Hz]
			altukey: alpha parameter for Tukey window used to construct thr wide frequency band
	"""

	def __init__(self):

		self.dt = cp.getfloat('SIGNAL_CHAR', 'dt') # sampling interval
		self.nsam = cp.getint('SIGNAL_CHAR', 'num_samples') # number of samples

		# for a gaussian spectrum (narrow band)
		self.cf = cp.getfloat('SIGNAL_CHAR', 'freq_centre') # centre frequency (mean of gaussian)
		fsigma_frac = cp.getint('SIGNAL_CHAR', 'freq_bw_frac')
		self.fsigma = self.cf/fsigma_frac # bandwidth (standard deviation of gaussian) 

		# for a Tukey spectrum (broadband)
		self.lf = cp.getfloat('SIGNAL_CHAR', 'freq_lb') # lower bound frequency
		self.hf = cp.getfloat('SIGNAL_CHAR', 'freq_ub') # upper bound frequency
		self.altukey = cp.getfloat('SIGNAL_CHAR', 'tukey_alpha') # alpha parameter for Tukey window in spectral domain from lf to hf
		self.pst = cp.getfloat('SIGNAL_CHAR', 'pow_spec_type') # power spectrum type; see associated function below

	def essentials(self):

		self.fhz = np.fft.fftfreq(self.nsam,self.dt)
		self.n_nn_fsam = len(self.fhz[self.fhz>=0])
		# number_of_non-negative_frequency_samples. Remember, when nsam is even, self.n_nn_fsam  is
		# smaller by one sample: the positive Nyquist is missing.
		# print("From SignalParameters in config: ")
		# print(self.nsam, self.dt, self.n_nn_fsam)

		self.omega_rad = 2*np.pi*self.fhz 		# angular frequency, radians
		self.domega = self.omega_rad[1] - self.omega_rad[0]	# d_self.omega_rad

		# time series corresponding to cross-correlation lags
		if self.nsam%2==0:
			self.tt=np.arange(-self.n_nn_fsam,self.n_nn_fsam) * self.dt
			# NB: the advantage of building a time series like this, rather than doing np.arange(tstart,tend,deltat),
        	# is that the above formulation ensures that you always get the time sample zero - regardless of deltat.
		else:
			self.tt=np.arange(-(self.n_nn_fsam-1),self.n_nn_fsam) * self.dt

		assert self.tt.size == self.nsam

	#-------------------------------------------------------------------------------
	def power_spectrum_synthetic(self):

		pss_type = {1: self.pspec_nb,		# narrow band
						2: self.pspec_bb}	# broad band

		if self.pst==1:
			self.pow_spec_sources = pss_type[self.pst](self.cf, self.fsigma)
		elif self.pst==2:
			self.pow_spec_sources = pss_type[self.pst](self.lf, self.hf, self.altukey)

	#-------------------------------------------------------------------------------
	def pspec_nb(self, f0, fsigma):
		# stf=np.exp(-tt**2/256) * np.cos(2*np.pi*sig_char.cf*tt)
		# pow_spec_sources=np.abs(np.fft.fft(stf))**2

		# return np.exp(-((self.fhz-f0)/self.fsigma)**2) + np.exp(-((self.fhz+f0)/self.fsigma)**2)
		return np.exp(-((self.fhz-f0)/fsigma)**2) + np.exp(-((self.fhz+f0)/fsigma)**2)

	#-------------------------------------------------------------------------------

	def pspec_bb(self, lowf, highf, alp):

		fwin = np.zeros(self.fhz.size)
		sfreqs=np.fft.fftshift(self.fhz)
		#print "sfreqs is: ", sfreqs[:5]
		#print "self.fhz is: ", self.fhz[:5]

		ilf_pos=np.searchsorted(sfreqs,lowf)
		ihf_pos=np.searchsorted(sfreqs,highf)
		ilf_neg=np.searchsorted(sfreqs,-lowf)
		ihf_neg=np.searchsorted(sfreqs,-highf)
		# we first obtain the relevant indices on the sorted frequencies -- because np.searchsorted
		# requires a sorted array -- and then convert to the corresponding indices on the frequencies
		# as ordered by np.fft.

		#print "ilf_pos and ihf_pos: ", ilf_pos, ihf_pos
		if sig_char.nsam%2==0:
			#print "Even number of samples: ", nsam
			ilf_pos -= int(sig_char.nsam/2)
			ihf_pos -= int(sig_char.nsam/2)
			ilf_neg += int(sig_char.nsam/2)
			ihf_neg += int(sig_char.nsam/2)
		else:
			#print "Odd number of samples: ", nsam
			ilf_pos -= int((sig_char.nsam-1)/2)
			ihf_pos -= int((sig_char.nsam-1)/2)
			ilf_neg += int((sig_char.nsam-1)/2) + 1
			ihf_neg += int((sig_char.nsam-1)/2) + 1

		#print "WHITENING low cut-off self.fhz pos and neg: ", self.fhz[ilf_pos], self.fhz[ilf_neg]
		#print "WHITENING high cut-off self.fhz pos and neg: ",  self.fhz[ihf_pos], self.fhz[ihf_neg]

		# Deal with painful discrepancies that may arise due to the vagaries of
		# np.searchsorted when it encounters an "exact" match.
		# got to make sure the window is symmetric (same for positive ad negative frequencies).
		if (abs(self.fhz[ihf_neg]) != self.fhz[ihf_pos]) or (abs(self.fhz[ilf_neg]) != self.fhz[ilf_pos]):
			if abs(self.fhz[ihf_neg]) != self.fhz[ihf_pos]:
				ihf_neg-=1
			if abs(self.fhz[ilf_neg]) != self.fhz[ilf_pos]:
				ilf_neg-=1

		npassband = ihf_pos - ilf_pos + 1 # no. of samples in the passband of the window (filter)
		ntot = round(npassband/(1-alp))	  # total number of samples in window
		ntrans = int(round(ntot*alp/2))   # number of samples in the transition band on either side

		#print "WHITENING npassband, ntot, ntrans: ", npassband, ntot, ntrans

		# make sure the window is symmetric
		# (not in the sense of symmetric relative to zero frequency, but
		# in the sense that transition band is equal on both sides)
		if (ntot - npassband)%2 != 0:
			# either increase the passband by one sample
			if (ntot - npassband) > 2*ntrans:
				npassband+=1
				ihf_pos+=1
				ihf_neg-=1
			# OR decrease the passband by one sample
			elif (ntot - npassband) < 2*ntrans:
				npassband-=1
				ihf_pos-=1
				ihf_neg+=1

		wstart_pos = ilf_pos - ntrans
		wend_pos = ihf_pos + ntrans
		wstart_neg = ihf_neg - ntrans
		wend_neg = ilf_neg + ntrans

		if wstart_pos <=0:
			raise SystemExit("Cannot implement sources power spectrum with alpha parameter = %.3f.\
				 Either change the parameter or choose a higher low-end frequency." %(alp))

		fwin[wstart_pos:wend_pos+1]=ss.windows.tukey(int(ntot),alpha=alp)
		fwin[wstart_neg:wend_neg+1]=ss.windows.tukey(int(ntot),alpha=alp)

		return fwin

# ------------------------------------------------------------------------------

sig_char = SignalParameters()
sig_char.essentials()
sig_char.power_spectrum_synthetic()

#*******************************************************************************
# GLOBALLY USED FUNCTIONS
#*******************************************************************************

add_epsilon = lambda m: m + (dom_geom.dx * epsilon)
mat_to_vec = lambda m: np.tril(m).flatten(order='F')
main_box_from_omost = lambda m: m[...,mainstart_om:mainend_om,mainstart_om:mainend_om]

#-------------------------------------------------------------------------------

def sig_thresh_dB(x, y, dB_thresh=-40):

	sig_dB = 10 * np.log10(y/np.max(y))
	x_thresh = x[sig_dB>=dB_thresh]
	return x_thresh[x_thresh>0]
	
#-------------------------------------------------------------------------------

def box_indices_largerbox_src(slx, sly, ngp_mod=dom_geom.ngp_box):

	""" Function that allows us to use the 'larger-box' source trick (in case of laterally homogeneous models).
	Returns the indices of the 'actual box' stencil within the larger box, given any source location(s)
	inside the actual box. When sources are allowed ANYWHERE inside the actual box, the larger box must be
	TWICE the size of the actual box.
	"""

	assert len(slx)==len(sly)

	hlen=dom_geom.box_len/2

	if ngp_mod==dom_geom.ngp_box:
		lfac=2
	elif ngp_mod>dom_geom.ngp_box:
		# size of the model is greater than the inverse modelling domain
		assert syn_data.ofac>2
		lfac = syn_data.ofac

	ngp_hlen_mod = int((ngp_mod - 1)/2)

	slocsx=np.array(slx)
	slocsy=np.array(sly)

	Xtrick_larger=np.arange(-lfac*hlen,lfac*hlen+dom_geom.dx,dom_geom.dx)
	Ytrick_larger=np.arange(-lfac*hlen,lfac*hlen+dom_geom.dx,dom_geom.dx)

	boxmid_x=list(map(lambda m: np.searchsorted(Xtrick_larger,m), -1*slocsx))
	boxmid_y=list(map(lambda m: np.searchsorted(Ytrick_larger,m), -1*slocsy))

	bstartx = np.array([m-ngp_hlen_mod for m in boxmid_x])
	bstarty = np.array([m-ngp_hlen_mod for m in boxmid_y])
	bendx= bstartx + ngp_mod
	bendy= bstarty + ngp_mod

	# if len(slx)>1:
	return bstartx, bendx, bstarty, bendy
	# else:
	# 	return *bstartx, *bendx, *bstarty, *bendy
		# using starred expressions to unpack a list
		# returning and error on the cluster, need to check.
		# return bstartx[0], bendx[0], bstarty[0], bendy[0]

################################################################################
# For SOURCE INVERSION only
################################################################################

# source model type
somod_types = {-1: 'egp',	# every grid point
				0: 'mg',	# multiple gaussians
				1: 'rg',	# ring of gaussians
				2: 'rgr',	# radially gaussian ring
				3: 'gg'}	# gaussian grid


smt_tru = cp.getint('SOMOD', 'src_mod_type_testdata')
smt_inv = cp.getint('SOMOD', 'src_mod_type_inversion')

tru_mod_type = somod_types[smt_tru]
inv_mod_type = somod_types[smt_inv]


mg_specs_r0_x_string = cp.get('SOMOD', 'somod_mg_specs_r0_x')
mg_specs_r0_x_list = [int(i) for i in mg_specs_r0_x_string.split()]

mg_specs_r0_y_string = cp.get('SOMOD', 'somod_mg_specs_r0_y')
mg_specs_r0_y_list = [int(i) for i in mg_specs_r0_y_string.split()]

mg_specs_w_x_string = cp.get('SOMOD', 'somod_mg_specs_w_x')
mg_specs_w_x_list = [int(i) for i in mg_specs_w_x_string.split()]

mg_specs_w_y_string = cp.get('SOMOD', 'somod_mg_specs_w_y')
mg_specs_w_y_list = [int(i) for i in mg_specs_w_y_string.split()]

mg_specs_mag_string = cp.get('SOMOD', 'somod_mg_specs_mag')
mg_specs_mag_list = [int(i) for i in mg_specs_mag_string.split()]

somod_mg_specs = {'r0': list(zip(mg_specs_r0_x_list,mg_specs_r0_y_list)),       # (x,y) coordinates of 2-D gaussian centres
                    'w': list(zip(mg_specs_w_x_list,mg_specs_w_y_list)),          # (x,y) widths of 2-D gaussians (km)
					'mag': mg_specs_mag_list}                                                         # magnitude

rg_specs_t1_string = cp.get('SOMOD', 'somod_rg_specs_t1')
rg_specs_t1_list = [int(i) for i in rg_specs_t1_string.split()]

rg_specs_t2_string = cp.get('SOMOD', 'somod_rg_specs_t2')
rg_specs_t2_list = [int(i) for i in rg_specs_t2_string.split()]

rg_specs_pert_string = cp.get('SOMOD', 'somod_rg_specs_pert')
rg_specs_pert_list = [int(i) for i in rg_specs_pert_string.split()]

somod_rg_specs = {'r': cp.getint('SOMOD', 'somod_rg_specs_r'),                                                                                                                                                 # radius (km)
                    'w': cp.getint('SOMOD', 'somod_rg_specs_w'),                                                                                                                                               # width (km)
					'as': cp.getint('SOMOD', 'somod_rg_specs_as'),                                                                                                                                             # basis angular sampling (degrees)
					'np': cp.getint('SOMOD', 'somod_rg_specs_np'),                                                                                                                                             # number of perturbed segments
					't1': rg_specs_t1_list,           # segment start (degrees)
			        't2': rg_specs_t2_list,           # segment end (degrees)
					'pert': rg_specs_pert_list}       # segment perturbation (additive)


rgr_specs_r0_x_string = cp.get('SOMOD', 'somod_rgr_specs_r0_x')
rgr_specs_r0_x_list = [int(i) for i in rgr_specs_r0_x_string.split()]

rgr_specs_r0_y_string = cp.get('SOMOD', 'somod_rgr_specs_r0_y')
rgr_specs_r0_y_list = [int(i) for i in rgr_specs_r0_y_string.split()]

rgr_specs_r_string = cp.get('SOMOD', 'somod_rgr_specs_r')
rgr_specs_r_list = [int(i) for i in rgr_specs_r_string.split()]

rgr_specs_w_string = cp.get('SOMOD', 'somod_rgr_specs_w')
rgr_specs_w_list = [int(i) for i in rgr_specs_w_string.split()]

rgr_specs_mag_string = cp.get('SOMOD', 'somod_rgr_specs_mag')
rgr_specs_mag_list = [int(i) for i in rgr_specs_mag_string.split()]


somod_rgr_specs = {'r0': list(zip(rgr_specs_r0_x_list,rgr_specs_r0_y_list)),  # (x,y) coordinates of ring centres
                     'r': rgr_specs_r_list,				# ring radii (km)
					 'w': rgr_specs_w_list,					# ring widths (km)
					 'mag': rgr_specs_mag_list}				# ring magnitudes


gg_specs_pos_x_string = cp.get('SOMOD', 'somod_gg_specs_pos_x')
gg_specs_pos_x_list = [float(i) for i in gg_specs_pos_x_string.split()]

gg_specs_pos_y_string = cp.get('SOMOD', 'somod_gg_specs_pos_y')
gg_specs_pos_y_list = [float(i) for i in gg_specs_pos_y_string.split()]

gg_specs_x_l_string = cp.get('SOMOD', 'somod_gg_specs_x_l')
gg_specs_x_l_list = [int(i) for i in gg_specs_x_l_string.split()]

gg_specs_x_u_string = cp.get('SOMOD', 'somod_gg_specs_x_u')
gg_specs_x_u_list = [int(i) for i in gg_specs_x_u_string.split()]

gg_specs_y_l_string = cp.get('SOMOD', 'somod_gg_specs_y_l')
gg_specs_y_l_list = [int(i) for i in gg_specs_y_l_string.split()]

gg_specs_y_u_string = cp.get('SOMOD', 'somod_gg_specs_y_u')
gg_specs_y_u_list = [int(i) for i in gg_specs_y_u_string.split()]

gg_specs_mag_string = cp.get('SOMOD', 'somod_gg_specs_mag')
gg_specs_mag_list = [int(i) for i in gg_specs_mag_string.split()]

somod_gg_specs = {'w': cp.getfloat('SOMOD', 'somod_gg_specs_w'),						# width (km)
				  'ls': cp.getfloat('SOMOD', 'somod_gg_specs_ls'),						# basis linear sampling (km)
				  'pos': list(zip(gg_specs_pos_x_list, gg_specs_pos_y_list)), # perturbed coordinates [x,y]
				  'np': cp.getint('SOMOD', 'somod_gg_specs_np'),					    # number of perturbations
				  'x_l': gg_specs_x_l_list,				# lower limit of gaussian array along the row
				  'x_u': gg_specs_x_u_list,				# upper limit of gaussian array along the row
				  'y_l': gg_specs_y_l_list,              # lower limit of gaussian array along the column
				  'y_u': gg_specs_y_u_list,			# upper limit of gaussian array along the column
				  'mag': gg_specs_mag_list}           # magnitude

# built-in checks
try:
	assert len(somod_mg_specs['w'])>=len(somod_mg_specs['r0'])
	assert len(somod_mg_specs['mag'])>=len(somod_mg_specs['r0'])
except AssertionError:
	raise SystemExit('Problem with source model specs (\'mg\') - see config file')

try:
	assert len(somod_rg_specs['t1'])>=somod_rg_specs['np']
	assert len(somod_rg_specs['t2'])>=somod_rg_specs['np']
	assert len(somod_rg_specs['pert'])>=somod_rg_specs['np']
except AssertionError:
	raise SystemExit('Problem with source model specs (\'rg\') - see config file')


# structure model
tru_struc_lat_homo = cp.getboolean('SOMOD', 'struc_lat_homo_testdata')
inv_struc_lat_homo = cp.getboolean('SOMOD', 'struc_lat_homo_inversion')

# modelling type
anal = {0: 'anal_scal_0D',		# Scalar, 0-D model
 		1: 'anal_elas_1D'}		# Elastic, 1-D model

num = {0: 'num_scal_2D_FD',		# Frequency domain
	   1: 'num_scal_2D_TD'}		# Time domain

mdlng_type = {0: anal,			# Analytical
				1: num}			# Numerical

mtl1_tru = cp.getint('SOMOD', 'modelling_level1_testdata')
mtl1_inv = cp.getint('SOMOD', 'modelling_level1_inversion')

mtl2_tru = cp.getint('SOMOD', 'modelling_level2_testdata')
mtl2_inv = cp.getint('SOMOD', 'modelling_level2_inversion')

tru_mdlng_type = mdlng_type[mtl1_tru][mtl2_tru]
inv_mdlng_type = mdlng_type[mtl1_inv][mtl2_inv]

################################################################################
# For STRUCTURE INVERSION/KERNELS only
################################################################################

coord_map_trivial = {'R': 'x', 'T': 'y', 'Z': 'z'} # coordinate transformation valid in a simple geometry (receiver pair along x-axis)
measured_quantity = {0: 'CTT', 1: 'AMP'}

mainstart_om = (syn_data.ofac-1)*dom_geom.ngp_hlen
mainend_om = mainstart_om + 2*dom_geom.ngp_hlen + 1

# observable or measured quantity
MQ = cp.getint('STRUC_INV', 'kernel_measurement')
