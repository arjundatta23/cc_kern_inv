#!/usr/bin/python

"""
Code by Arjun Datta

Started 2018 at Tata Institute of Fundamental Research, Mumbai, India

SERIAL VERSION

"""

###########################################################################################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../modules_common')
# path to the modules which are common to the cc source- and structure-inversion codes
sys.path.append(os.path.expanduser('~/code_general/modules.python'))
# path to the "SW1D_earthsr" set of modules

# Custom modules: set 1
import utils_io as uio
import config_file as config

#*********************************************** Parameters from config file ****************************************************
use_reald = config.ext_data
nrecs_select = config.nrecs # no. of stations/receivers to use

dg = config.dom_geom
origx = dg.grid_origx
origy = dg.grid_origy
ngp_domain = dg.ngp_box

wspeed = config.scal_mod.wavspeed_scal2D

# Custom modules: set 2
import cctomo_utils1 as u1

#***************************************************** Get code input ************************************************************
uioo = uio.get_user_input()
uc_keys = ["scalar", "elastic", "nz", "elas_mod_1D", "acou_vel_mod_obs", "acou_vel_mod_syn"]
user_choices = {}
for k in uc_keys:
	try:
		user_choices[k] = getattr(uioo, k)
	except AttributeError:
		user_choices[k] = None

###################################### START OF PROGRAM: PART 1 (SETUP) ##################################################

#--------------------------------- Flags governing scope of code ---------------------------------------------
do_inv=True
iter_only1=False

#--------------------------------- Station/receiver network and modelling geometry ---------------------------------------------
coordfile="INPUT/receivers.csv"

stno, stid, stx, sty = uio.read_station_file(coordfile)

try:
	assert stno[-1] == len(stid)
	nrecs_total = stno[-1]
except AssertionError:
	raise SystemExit("Problem with input STATION/RECEIVER coordinates file.")
try:
	assert nrecs_select <= nrecs_total
except AssertionError:
	raise SystemExit("Trying to select more STATIONS/RECEIVERS (%d) than are present in the input file (%d)" %(nrecs_select, nrecs_total))

smdo = u1.setup_modelling_domain(stno, stid, stx, sty, origx, origy)

#-------------------------------- Data reading (if relevant) and temporal signal attributes of data/synthetics ----------------------------------

if not use_reald:
# SYNTHETIC TEST CASE
	sig_att = config.sig_char
	oam = None
	obsdata = None
	avail_data_dummy = 1 - np.eye(nrecs_select)
	obsdata_info = (avail_data_dummy, None, None, None)
else:
# EXTERNAL (REAL) DATA CASE
	import cctomo_utils2 as u2

	try:
		rdo=u2.cc_data.Read(smdo.chosen_st_id, smdo.chosen_st_no, uioo.data_loc, uioo.data_fmt, uioo.new_dt, uioo.fext)
		mdo=u2.cc_data.MatrixForm()
		pdo=u2.cc_data.Process(smdo.act_dist_rp, wspeed, mdo.nmissing)
		edo=u2.cc_data.Errors(pdo.snr)
	except Exception as e:
		print(e)
		raise SystemExit("Problem with data read/preparation/processing; program aborted.")

	obsdata = pdo.use_data
	obsdata_info = (mdo.mark_avail_data, pdo.snr, edo.esnrpd_ltpb, (pdo.lefw, pdo.rigw))

	# override signal parameters set in the config file
	sig_att = config.SignalParameters()
	sig_att.dt = pdo.dt
	sig_att.nsam = pdo.nsam
	sig_att.lf = rdo.fpb[0]
	sig_att.hf = rdo.fpb[1]
	sig_att.cf = (rdo.fpb[0]+rdo.fpb[1])/2
	sig_att.altukey = rdo.fpb[2]
	sig_att.pst = 3

	sig_att.essentials()
	sso = u2.source_spectrum(obsdata)
	try:
		oam, sig_att.pow_spec_sources = sso.match_obs_spectra(sig_att.nsam, sig_att.fhz, mdo.nmissing)
	except Exception as e:
	    print(e)
	    raise SystemExit("Problem building the source spectrum in u1, program aborted.")

	# sig_att = config.sig_char

	# check self-consistency
	try:
		tolerance=sig_att.dt/1e2
		assert np.allclose(sig_att.tt,pdo.ccl_used,atol=tolerance)
	except AssertionError:
		print(sig_att.tt, sig_att.tt.size)
		print(pdo.ccl_used, pdo.ccl_used.size)
		raise SystemExit("Aborted - problem with cross-correlation lag time series.")

#-------------------------------- Check suitability of parameter choices/code input ----------------------------------
# Custom modules: set 3
import validate_params as val

try:
	cso = val.check_settings(sig_att, smdo.dist_1Darr)
	cso.time_series_length()
	cso.dc_comp_spectrum()

	mro = val.memory_reqt(sig_att, user_choices['nz'])
	mro.source_kern()
except Exception as e:
	print(e)
	if do_inv:
		raise SystemExit("Problem with settings; program aborted.")
	else:
		print("INVERSION NOT POSSIBLE")

############################################## PART 2 (CORE CODE) #########################################################
# Custom modules: set 4
import hans2013_serial as h13

icao = h13.inv_cc_amp(smdo.rchosenx_igp, smdo.rchoseny_igp, sig_att, do_inv, iter_only1, obsdata, obsdata_info)

############################################## PART 3 (POST RUN) ##########################################################

try:
	icao
except NameError:
	# icao is not defined
	print("Core module(s) not used. Showing network plots...")
	plt.show()
	# for plots that are made BEFORE running the h13 module
else:
	if do_inv:
	# Inversion has been run (at least one iteration)
		print("\nStoring the result...")
		uio.post_run(0, sig_att, 0, oam, obsdata_info[0], osmd=smdo, oica=icao)
	else:
	# Inversion has NOT been run, but models have been set up
		if use_reald:
			edo = u2.egy_vs_dist(use_reald, icao.dist_rp_sorted, icao.c/icao.hf)
			icao.egy_obs, icao.oef = edo.fit_curve_1byr(sig_att.nsam, obsdata, 'TD', icao.dist_rp_grid, mdo.nmissing, edo.sig_dummy)
		uio.post_run(0, sig_att, 1, oam, obsdata_info[0], osmd=smdo, oica=icao)
		plt.show()
