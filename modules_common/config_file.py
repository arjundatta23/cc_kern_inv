#!/usr/bin/python

# Standard modules
import numpy as np
import scipy.signal as ss
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
	SYST_RAM = 6

################################################################################
# BASICS (for both SOURCE/STRUCTURE inversion)
################################################################################

reald=False
# whether working with real data or synthetics

# dictionaries
comp_dic_xyz={'x':0, 'y':1, 'z':2}
comp_dic_rtz={'R':0, 'T':1, 'Z':2}
green_tensor_rtz = {0:'RR', 1:'RT', 2:'RZ',
 					3:'TR', 4: 'TT', 5: 'TZ',
					6:'ZR', 7: 'ZT', 8: 'ZZ'}

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

		self.Xoutmost = np.arange(-invc.ofac*hlen, invc.ofac*hlen+self.dx, self.dx)
		self.Youtmost = np.arange(-invc.ofac*hlen, invc.ofac*hlen+self.dx, self.dx)

		self.gx, self.gy = np.meshgrid(self.X,self.Y)
		self.gx2, self.gy2 = np.meshgrid(self.X2,self.Y2)
		self.omgx, self.omgy = np.meshgrid(self.Xoutmost,self.Youtmost)

# ------------------------------------------------------------------------------

dom_geom = DomainGeometry()
dom_geom.dx = 1.0 #2.5
dom_geom.zmax = 0
dom_geom.box_len = 60.

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
ccmt.src_dir = 'z'
ccmt.GTC = [8]

################################################################################
# Parameters pertaining to inversion
################################################################################

class InversionChoices():
	""" sets user choices for inversion (including synthetic tests)
		User-defined attributes:
			gamma_inv: damping parameter for regularized least-squares inversion
			ofac: parameter relevant for synthetic test inversions only - allows the true/test model to be larger than the modelling domain; see h13 module for details

	"""

invc = InversionChoices()
invc.gamma_inv = 0.1
invc.ofac = 2

dom_geom.grid() # needs to be done AFTER "invc.ofac" is defined

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
scal_mod.rho_scal2D = 1.0
scal_mod.wavspeed_scal2D = 3.0

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

	def essentials(self):

		self.fsigma = self.cf/2

		self.fhz = np.fft.fftfreq(self.nsam,self.dt)
		self.n_nn_fsam = len(self.fhz[self.fhz>=0])
		# number_of_non-negative_frequency_samples. Remember, when nsam is even, self.n_nn_fsam  is
		# smaller by one sample: the positive Nyquist is missing.

		self.omega_rad = 2*np.pi*self.fhz 		# angular frequency, radians
		self.domega = self.omega_rad[1] - self.omega_rad[0]	# d_self.omega_rad

		# time series corresponding to cross-correlation lags
		if self.nsam%2==0:
			self.tt=np.arange(-self.n_nn_fsam,self.n_nn_fsam)*sig_char.dt
			# NB: the advantage of building a time series like this, rather than doing np.arange(tstart,tend,deltat),
        	# is that the above formulation ensures that you always get the time sample zero - regardless of deltat.
		else:
			self.tt=np.arange(-(self.n_nn_fsam-1),self.n_nn_fsam)*sig_char.dt

		assert self.tt.size == self.nsam

	#-------------------------------------------------------------------------------
	def power_spectrum(self):

		self.pst = 1

		pss_type = {1: self.pspec_nb,		# narrow band
						2: self.pspec_bb}	# broad band

		if self.pst==1:
			self.pow_spec_sources = pss_type[self.pst](self.cf)
		elif self.pst==2:
			self.pow_spec_sources = pss_type[self.pst](self.lf, self.hf, self.altukey)

	#-------------------------------------------------------------------------------
	def pspec_nb(self, f0):
		# stf=np.exp(-tt**2/256) * np.cos(2*np.pi*sig_char.cf*tt)
		# pow_spec_sources=np.abs(np.fft.fft(stf))**2

		return np.exp(-((self.fhz-f0)/self.fsigma)**2) + np.exp(-((self.fhz+f0)/self.fsigma)**2)

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
sig_char.dt = 0.2
sig_char.nsam = 250
sig_char.cf = 0.3
sig_char.lf = 0.04 # lower bound frequency
sig_char.hf = 0.4 # upper bound frequency
sig_char.altukey = 0.1 # alpha parameter for Tukey window in spectral domain from lf to hf

sig_char.essentials()
sig_char.power_spectrum()

#*******************************************************************************
# GLOBALLY USED FUNCTIONS
#*******************************************************************************

add_epsilon = lambda m: m + (dom_geom.dx*1e-6)
main_box_from_omost = lambda m: m[...,mainstart_om:mainend_om,mainstart_om:mainend_om]

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
		assert invc.ofac>2
		lfac = invc.ofac

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
somod_types = {0: 'mg',		# multiple gaussians
				1: 'rg',	# ring of gaussians
				2: 'rgr'}	# radially gaussian ring

smt_tru = 1
smt_inv = 1

tru_mod_type = somod_types[smt_tru]		# for synthetic tests only
inv_mod_type = somod_types[smt_inv]

somod_mg_specs = {'r0': [(0,-25), (-15,0)],		# (x,y) coordinates of 2-D gaussian centres
					'w': [(500,18), (18,500)],	# (x,y) widths of 2-D gaussians (grid-point units)
					'mag': [10,10]}				# magnitude

somod_rg_specs = {'r': 25,						# radius (km)
					'w': 10,					# width (grid-point units)
					'as': 10,					# basis angular sampling (degrees)
					'np': 3,					# number of perturbed segments
					't1': [130,220,345,30],		# segment start (degrees)
					't2': [150,240,15,50],		# segment end (degrees)
					'pert': [8,5,5]}			# segment perturbation (additive)

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
tru_struc_lat_homo = True
inv_struc_lat_homo = True

# modelling type
anal = {0: 'anal_scal_0D',		# Scalar, 0-D model
 		1: 'anal_elas_1D'}		# Elastic, 1-D model

num = {0: 'num_scal_2D_FD',		# Frequency domain
	   1: 'num_scal_2D_TD'}		# Time domain

mdlng_type = {0: anal,			# Analytical
				1: num}			# Numerical

mtl1_tru = 0
mtl1_inv = 0

mtl2_tru = 1
mtl2_inv = 1

tru_mdlng_type = mdlng_type[mtl1_tru][mtl2_tru]		# required for synthetic tests only
inv_mdlng_type = mdlng_type[mtl1_inv][mtl2_inv]

amp_scaling = True
# relevant for synthetics tests only, recommended setting: False
# - defaults to True in case of real data
# - defaults to False in case 'tru_mdlng_type' and 'inv_mdlng_type' are the SAME
# user must set this paramter judiciously. 'True' setting recommended ONLY when unable to reconcile
# (orders of magnitude) amplitude discrepancies between results of different modelling methods.

################################################################################
# For STRUCTURE INVERSION/KERNELS only
################################################################################

ngp_rloc=20
rloc=ngp_rloc*dom_geom.dx

coord_map_trivial = {'R': 'x', 'T': 'y', 'Z': 'z'} # coordinate transformation valid in a simple geometry (receiver pair along x-axis)
measured_quantity = {0: 'CTT', 1: 'AMP'}

# chosen dictionary keys
MQ = 1 # for 'measured_quantity'

# ot_xO = otgx #- dx/2
# ot_yO = otgy #- dx/2
om_xO = dom_geom.omgx #- dx/2
om_yO = dom_geom.omgy #- dx/2
# coordinates relative to source

# ot_rO = np.sqrt(ot_xO**2 + ot_yO**2)
# ot_rO = add_epsilon(ot_rO)
om_rO = np.sqrt(om_xO**2 + om_yO**2)
om_rO = add_epsilon(om_rO)
# the source is at the origin

mainstart_om = (invc.ofac-1)*dom_geom.ngp_hlen
mainend_om = mainstart_om + 2*dom_geom.ngp_hlen + 1
