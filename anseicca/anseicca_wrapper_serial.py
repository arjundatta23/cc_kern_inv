#!/usr/bin/python

"""
Code by Arjun Datta at Tata Institute of Fundamental Research, Mumbai, India

SERIAL CODE

"""

###########################################################################################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../modules_common')
# path to the modules which are common to the cc source and structure inversion codes
sys.path.append(os.path.expanduser('~/code_general/modules.python'))
# path to the "SW1D_earthsr" set of modules
sys.path.append(os.path.expanduser('~/code_general/Bharath_Shekar'))

# Custom modules: set 1
import utils_io as uio
import config_file as config
import anseicca_utils2 as u2
import SW1D_earthsr.utils_pre_code as u0

#*********************************************** Parameters from config file ****************************************************
dg = config.dom_geom
use_reald = config.reald
ngp_domain = dg.ngp_box
wspeed = config.scal_mod.wavspeed_scal2D

# Custom modules: set 2
import anseicca_utils1 as u1

#***************************************************** Get code input ************************************************************
uioo = uio.process_args(sys.argv)
try:
	mod1dfile = uioo.mod1dfile
	egn_ray = uioo.egn_ray
	disp_ray = uioo.disp_ray
	scalar=False
	elastic=True
	try:
		egn_lov = uioo.egn_lov
		disp_lov = uioo.disp_lov
	except AttributeError:
		# no Love wave input
		pass
except AttributeError:
	scalar=True
	elastic=False

###################################### START OF PROGRAM: PART 1 (SETUP) ##################################################

#--------------------------------- Flags/user choices governing scope of code ---------------------------------------------
do_inv=True
map_plots=True
iter_only1=True

#--------------------------------- Station/receiver network and modelling geometry ---------------------------------------------
coordfile="INPUT/receivers.csv"
origx = 406; origy = 8504
# origx = 492; origy = 2751.0
# coordinate origin of the modelling domain (km units); ballpark values chosen to have the origin roughly at the network centre
nrecs=23 # total no. of stations/receivers in the network
nrecs_select=23 # no. of stations/recivers to use

stno, stid, stx, sty = u1.read_station_file(coordfile)
smdo = u2.setup_modelling_domain(stno, stid, stx, sty, origx, origy, nrecs_select, map_plots)

#-------------------------------- Model reading (if relevant) ----------------------------------
if elastic:
# read input depth-dependent model and fix/extract necessary parameters
    upreo = u0.model_1D(mod1dfile)
    dep_pts_mod = upreo.deps_all
    hif_mod = upreo.mod_hif
    upreo.fix_max_depth(dg.zmax)
    dep_pts_use = upreo.deps_tomax
    # hif_mod_use = hif_mod[hif_mod<=dg.zmax]
    print("Layer interfaces in model: ", hif_mod, hif_mod.size)
    print("Depth points to be used in code: ", dep_pts_use, dep_pts_use.size)
    nz = dep_pts_use.size
    wspeed = config.scal_mod.wavspeed_scal2D # NEED TO CHANGE, SHOULD BE BASED ON VELOCITIES FROM DISP FILE
else:
    nz = 1

#-------------------------------- Check suitability of parameter choices/code input ----------------------------------
# Custom modules: set 3
import validate_params as val
import hans2013_serial as h13

try:
	cso = val.check_settings(smdo.dist_1Darr)
	cso.time_series_length()
	mro = val.memory_reqt(nz)
	mro.source_kern(nrecs_select)
except Exception as e:
	print(e)
	raise SystemExit("Problem with settings, program aborted.")

#-------------------------------- Data reading (if relevant) and temporal signal attributes of data/synthetics ----------------------------------

if not use_reald:
# SYNTHETIC TEST CASE
	sig_att = config.sig_char
	obsdata=None
	obsdata_info=None
else:
# REAL DATA CASE
	data_format = {0: 'python_binary_archive'}
	infile="EXAMPLES/stack_manypairs_99_zz.npz"

	rdo=u2.cc_data.Read(infile,data_format[0])
	mdo=u2.cc_data.MatrixForm(stno,stx,sty,nrecs_select,smdo.chosen_st_no)
	pdo=u2.cc_data.Process(wspeed,nrecs_select)
	azstep=90
	edo=u2.cc_data.Errors(nrecs_select,azstep)

	# override signal parameters set in the config file
	sig_att = config.SignalParameters()
	sig_att.dt = pdo.dt
	sig_att.nsam = pdo.nsam
	sig_att.lf = rdo.fpb[0]
	sig_att.hf = rdo.fpb[1]
	sig_att.cf = (rdo.fpb[0]+rdo.fpb[1])/2
	sig_att.altukey = rdo.fpb[2]

	sig_att.essentials()

	obsdata = pdo.use_data
	obsdata_info = (mdo.act_dist_rp, pdo.snr, edo.DelE)

############################################## PART 2 (CORE CODE) #####################################################

icao = h13.inv_cc_amp(smdo.rchosenx_igp, smdo.rchoseny_igp, sig_att, do_inv, iter_only1, obsdata, obsdata_info)

############################################## PART 3 (POST RUN) ##########################################################

try:
	icao
except NameError:
	# icao is not defined
	print("h13 module not used. Showing plot(s)...")
	plt.show()
	# for plots that are made before running the h13 module
else:
	if do_inv:
	# Inversion has been run (at least one iteration)
		print("\nStoring the result...")
		u2.post_run(wspeed, sig_att, 0, osmd=smdo, oica=icao)
	else:
	# Models have been setup but inversion has NOT been run
		u2.post_run(wspeed, sig_att, 1, osmd=smdo, oica=icao)
		plt.show()
