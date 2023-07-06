#!/usr/bin/python

# Standard modules
import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
import config_file as config
import read_velocity_models as u0

##########################################################################################################################

# global variables
dg = config.dom_geom
dxy = config.dom_geom.dx
reald = config.ext_data

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

##########################################################################################################################

class get_user_input:

	def __init__(self):
		
		if config.ext_data:
			data_format = {1: 'binary_archive_python', 2: 'individual_files_ccpairs'}

			self.data_loc = input("Path to data (file or directory): ")
			df = int(input("Data format: \n1. Archive (Python Binary)\n 2. Individual files\n: "))
			self.data_fmt = data_format[df]

			usrc_resamp = float(input("Resample data? (0 if no, dew dt in seconds if yes): "))
			self.new_dt = None if usrc_resamp==0 else usrc_resamp

			self.fext = input("File extension: ")


		if config.inv_mdlng_type.find('anal_elas_1D') != -1 or ( not config.ext_data and config.tru_mdlng_type.find('anal_elas_1D') != -1 ):
			indir=input("Directory containing 1-D elastic model files: ")
			try:
				assert os.path.isdir(indir)
				self.scalar=False
				self.elastic=True
			except AssertionError:
				raise SystemExit("Problem with code input: must be a directory containing disp/eigen files")
			else:
				self.elas_mod_1D={}
				self.nz = self.input_elastic_1D(indir)
		else:
			self.scalar=True
			self.elastic=False
			self.nz=1

		if config.inv_mdlng_type.find('num_scal_2D') != -1:

			input_vel_mod_syn = input("2-D velocity model to be used in INVERSION (enter 0 to use a homogeneous velocity model): ")
			if input_vel_mod_syn.isdigit():
				print("No velocity model file provided, using homogeneous velocity")
			else:
				self.acou_vel_mod_syn = self.input_acoustic_2D(input_vel_mod_syn)

		if (not config.ext_data and config.tru_mdlng_type.find('num_scal_2D') != -1):
			input_vel_mod_obs = input("2-D velocity model to be used to simulate TEST DATA (enter 0 for homogeneous/same as in inversion): ")
			if input_vel_mod_obs.isdigit():
				try:
					self.acou_vel_mod_obs = self.acou_vel_mod_syn
				except AttributeError:
					pass
			else:
				self.acou_vel_mod_obs = self.input_acoustic_2D(input_vel_mod_obs)

    # --------------------------------------------------------------------------

	def input_elastic_1D(self, inpdir):

		fullpath = lambda m: os.path.join(inpdir,m)
		mod_files=[ n for n in os.listdir(inpdir) if n.startswith("mod") ]
		egn_files=[ n for n in os.listdir(inpdir) if n.startswith("eigen.") ]
		disp_files=[ n for n in os.listdir(inpdir) if n.startswith("disp.") ]
		try:
			assert len(mod_files)==1
			assert len(egn_files)==len(disp_files)
			assert len(egn_files)>0 and len(egn_files)<3
		except AssertionError:
			raise SystemExit("Problem with code input: must contain files corresponding to one Earth model only")

		self.elas_mod_1D["mod1dfile"]=fullpath(mod_files[0])
		self.elas_mod_1D["egn_ray"]=fullpath([ f for f in egn_files if f.endswith(".ray") ][0]) # eigenfunctions Rayleigh
		self.elas_mod_1D["disp_ray"]=fullpath([ f for f in disp_files if f.endswith(".ray") ][0]) # dispersion Rayleigh
		if len(egn_files)==2:
			self.elas_mod_1D["egn_lov"]=fullpath([ f for f in egn_files if f.endswith(".lov") ][0]) # eigenfunctions Love
			self.elas_mod_1D["disp_lov"]=fullpath([ f for f in disp_files if f.endswith(".lov") ][0]) # dispersion Love

		# read input depth-dependent model and fix/extract necessary parameters
		upreo = u0.model_1D(self.elas_mod_1D["mod1dfile"])
		upreo.fix_max_depth(dg.zmax)

		self.elas_mod_1D["hif_mod"] = upreo.mod_hif
		self.elas_mod_1D["dep_pts_use"] = upreo.deps_tomax
		self.elas_mod_1D["dep_pts_mod"] = upreo.deps_all
		print("Layer interfaces in model: ", self.elas_mod_1D["hif_mod"], self.elas_mod_1D["hif_mod"].size)
		print("Depth points to be used in code: ", self.elas_mod_1D["dep_pts_use"], self.elas_mod_1D["dep_pts_use"].size)
		return self.elas_mod_1D["dep_pts_use"].size

	# --------------------------------------------------------------------------

	def input_acoustic_2D(self, input_vel_mod):

		vel_mod_file = os.path.expanduser(input_vel_mod)
		up2d = u0.model_2D(dg.box_len, dg.dx)
		# up2d.grd_file(vel_mod_file)
		up2d.npz_file(vel_mod_file)
		return up2d.vel_acou

#######################################################################################################################################

class post_run():

	def __init__(self, calling_code, schar, dowhat, oam, dam, **classobjects):

		self.oam = oam
		self.dam = dam
		self.schar = schar

		self.oica = classobjects['oica']
		try:
			self.osmd = classobjects['osmd']
			self.pickling=True
		except KeyError:
			self.pickling=False

		pickle_func={0: self.save_pickle_anseicca, 1: self.save_pickle_strucinv}

		if dowhat==0:
			# save the results
			# self.save_pickle()
			pickle_func[calling_code]()
		elif dowhat==1:
			# produce plots showing inversion setup etc.
			self.make_plots()

	# --------------------------------------------------------------------------

	def save_pickle_anseicca(self):

		jarname="output_anseicca.pckl"
		# archname="output_anseicca"
		jarfile=os.path.join(os.getcwd(),jarname)
		# archfile=os.path.join(os.getcwd(),archname)
		win_lr_np = np.stack((self.oica.negl, self.oica.negr, self.oica.posl, self.oica.posr), axis=0)

		if self.pickling:
			jar=gzip.open(jarfile,'w')

			# STORE NECESSARY INPUT PARAMETERS/SETTINGS
			# from config file
			pickle.dump(config.ext_data,jar)
			pickle.dump(config.ccmt,jar)
			pickle.dump(config.invc,jar)
			pickle.dump(config.scal_mod,jar)
			pickle.dump(config.dom_geom,jar)
			pickle.dump(config.sig_char,jar)
			pickle.dump(config.somod_mg_specs,jar)
			pickle.dump(config.somod_rg_specs,jar)
			pickle.dump(config.somod_rgr_specs,jar)
			pickle.dump(config.somod_gg_specs,jar)
			pickle.dump(config.inv_mod_type,jar)
			pickle.dump(config.inv_mdlng_type,jar)
			if not reald:
				pickle.dump(config.tru_mod_type,jar)
				pickle.dump(config.tru_mdlng_type,jar)
				pickle.dump(config.syn_data,jar)
			# from the main code (parameters defined by data, in case of real data)
			pickle.dump(self.oica.data_availability,jar)
			pickle.dump(self.schar,jar)
			pickle.dump(self.oica.pss,jar)
			# from external sources (e.g. velocity models)
			try:
				pickle.dump((self.oica.vp_dvto_obs,self.oica.vp_dvto_syn),jar)
			except AttributeError:
				pickle.dump((None,None),jar)

			# STORE DESIRED OUTPUT (quantities computed by the code)
			pickle.dump(win_lr_np,jar)
			pickle.dump(self.oica.snr_val,jar)
			pickle.dump(self.oica.dist_rp_grid,jar)
			pickle.dump(list(zip(self.osmd.chosen_st_no,self.osmd.chosen_st_id)),jar)
			pickle.dump(self.osmd.rchosenx_igp,jar)
			pickle.dump(self.osmd.rchoseny_igp,jar)
			pickle.dump(self.oica.allit_mc,jar)
			pickle.dump(self.oica.allit_misfit,jar)
			pickle.dump(self.oica.mfit_kern_pos,jar)
			pickle.dump(self.oica.mfit_kern_neg,jar)
			pickle.dump(self.oica.flit_indmis_p,jar)
			pickle.dump(self.oica.flit_indmis_n,jar)
			pickle.dump(self.oica.obscross,jar)
			pickle.dump(self.oica.allit_syncross[0],jar)
			pickle.dump(self.oica.allit_syncross[-1],jar)
			pickle.dump(self.oica.distribs_inv,jar)
			if not reald:
				pickle.dump(self.oica.distribs_true,jar)
			jar.close()

	# --------------------------------------------------------------------------

	def save_pickle_strucinv(self):

		jarname="output_strucinv.pckl"
		jarfile=os.path.join(os.getcwd(),jarname)
		if self.pickling:
			jar=gzip.open(jarfile,'w')

			# STORE NECESSARY INPUT PARAMETERS/SETTINGS
			# from config file
			pickle.dump(config.ext_data,jar)
			pickle.dump(config.ccmt,jar)
			pickle.dump(config.dom_geom,jar)
			pickle.dump(config.scal_mod,jar)
			pickle.dump(config.invc,jar)
			# from the main code (parameters defined by data, in case of real data)
			pickle.dump(self.schar,jar)
			# pickle.dump(self.oica.pss,jar)

			# STORE DESIRED OUTPUT (all from the main code)
			# pickle.dump(win_lr_np,jar)
			pickle.dump(self.oica.dist_rp_grid,jar)
			pickle.dump(list(zip(self.osmd.chosen_st_no,self.osmd.chosen_st_id)),jar)
			pickle.dump(self.osmd.rchosenx_igp,jar)
			pickle.dump(self.osmd.rchoseny_igp,jar)
			pickle.dump(self.oica.syncross,jar)

		jar.close()

	# --------------------------------------------------------------------------

	def make_plots(self):
		#plt.tight_layout() # use this when axis labels are off the plot

		def plot_mod(inmod, ptitle, flexisize, model_min=None, model_max=None):
			fig=plt.figure()
			axm=fig.add_subplot(111)
			axm.set_title(ptitle)
			if flexisize and config.syn_data.ofac>2:
			    cax=axm.pcolor(config.dom_geom.gx2,config.dom_geom.gy2,inmod,cmap=plt.cm.jet,vmin=model_min,vmax=model_max)
			else:
			    cax=axm.pcolor(config.dom_geom.gx,config.dom_geom.gy,inmod,cmap=plt.cm.jet,vmin=model_min,vmax=model_max)
			    #cax=axm.pcolor(config.dom_geom.gx,config.dom_geom.gy,inmod,cmap=plt.cm.jet)
			axm.plot(dxy*self.osmd.rchosenx_igp, dxy*self.osmd.rchoseny_igp, 'wd', markerfacecolor="None")
			axm.set_xlabel("X [km]", fontsize=14)
			axm.set_ylabel("Y [km]", fontsize=14)
			axm.tick_params(axis='both', labelsize=14)
			for j in range(len(self.osmd.rchosenx_igp)):
				# axm.annotate(j, xy=(dxy*self.osmd.rchosenx_igp[j],dxy*self.osmd.rchoseny_igp[j]), color='white')
				axm.annotate(self.osmd.chosen_st_id[j], xy=(dxy*self.osmd.rchosenx_igp[j],dxy*self.osmd.rchoseny_igp[j]), color='white')
			plt.colorbar(cax,ax=axm)

		if reald:
			# ----------------- station map ----------------------
			# ind_all = np.arange(rx.size)
			fig_map=plt.figure()
			axmap=fig_map.add_subplot(111,aspect='equal')
			nrecs=self.osmd.act_recx_rel.size
			# plot stations
			axmap.scatter(self.osmd.act_recx_rel, self.osmd.act_recy_rel, marker='^', s=100, facecolor='r', edgecolor='k')
			# plot interstation paths for all RELEVANT station pairs
			pair_counter=0
			for k in range(nrecs-1):
				for j in range(k+1,nrecs):
					if self.dam[j,k]>0:
						# print("Relevant pair: ",j,k)
						x1=self.osmd.act_recx_rel[j]
						x2=self.osmd.act_recx_rel[k]
						y1=self.osmd.act_recy_rel[j]
						y2=self.osmd.act_recy_rel[k]
						axmap.plot([x1,x2],[y1,y2],color='grey')
						pair_counter+=1
			axmap.set_xlabel("X [km]")
			axmap.set_ylabel("Y [km]")
			# try:
			# 	assert pair_counter==len(data_list_form)
			# except AssertionError:
			# 	print(pair_counter, len(data_list_form))
			# 	raise SystemExit("From uio post_run: problem with pair count.")

			# -------------------- SNR plot ----------------------
			snr_2D = self.oica.snr_val
			snr_pb = snr_2D[np.nonzero(np.tril(snr_2D))]
			snr_nb = snr_2D[np.nonzero(np.triu(snr_2D))]
			showmax_nb = 100
			showmax_pb = 100
			hbe_snr_n = np.linspace(0,showmax_nb,10)
			hbe_snr_p = np.linspace(0,showmax_pb,10)
			n_nb=np.argwhere(snr_nb<=showmax_nb).size
			n_pb=np.argwhere(snr_pb<=showmax_pb).size
			fig_snr = plt.figure()
			fig_snr.suptitle('S/N ratio')
			ax_snr_nb=fig_snr.add_subplot(121)
			ax_snr_pb=fig_snr.add_subplot(122)
			ax_snr_nb.hist(snr_nb,bins=hbe_snr_n,edgecolor='black')
			# ax_snr_nb.text(0.7, 0.8, "%d/%d" %(n_nb, len(data_list_form)), transform=ax_snr_nb.transAxes)
			ax_snr_nb.set_xlabel('Negative branch')
			ax_snr_pb.hist(snr_pb,bins=hbe_snr_p,edgecolor='black')
			# ax_snr_pb.text(0.7, 0.7, "%d/%d" %(n_pb, len(data_list_form)), transform=ax_snr_pb.transAxes)
			ax_snr_pb.set_xlabel('Positive branch')

			# -------------- energy vs distance plot -------------
			fig_egy = plt.figure()
			ax_egy=fig_egy.add_subplot(111)
			x_use = self.oica.dist_rp_sorted[np.isfinite(self.oica.egy_obs)]
			y_obs = self.oica.egy_obs[np.isfinite(self.oica.egy_obs)]
			y_obs_norm = y_obs/np.amax(y_obs)
			y_fit = self.oica.oef[np.isfinite(self.oica.egy_obs)]
			y_fit_norm = y_fit/np.amax(y_obs)
			ax_egy.scatter(x_use,y_obs_norm)
			ax_egy.plot(x_use,y_fit_norm,'k',)
			ax_egy.set_xlabel("Distance [km]", fontsize=14)
			ax_egy.set_ylabel("Normalised energy", fontsize=14)

			# -------------- source spectrum plot -------------
			fig_ss = plt.figure()
			ax_ss=fig_ss.add_subplot(111)
			# ax_ss.plot(np.fft.fftshift(self.oica.fhz),np.fft.fftshift(self.oica.obs_aspec_mean), label='Observed')
			ax_ss.plot(np.fft.fftshift(self.oica.fhz),np.fft.fftshift(self.oam), label='Observed')
			ax_ss.plot(np.fft.fftshift(self.oica.fhz),np.fft.fftshift(self.oica.pss), label='Synthetic')
			# ax_ss.set_xlim(0,1/(2*self.deltat))
			ax_ss.set_xlabel('Frequency [Hz]')
			ax_ss.set_title('Source spectrum')
			ax_ss.legend()
		else:
			# ******** TEMPORARY HACK FOR SAVING THE SOURCE MODEL ********
			# archname="test_vel_model_2D"
			# archfile=os.path.join(os.getcwd(),archname)
			# np.savez_compressed(archfile, gx=config.dom_geom.gx, gy=config.dom_geom.gy, somod=self.oica.distribs_true)
			# ******** END: TEMPORARY HACK ********
			mod_min=min(np.amin(self.oica.distribs_true),np.amin(self.oica.distribs_start))
			mod_max=max(np.amax(self.oica.distribs_true),np.amax(self.oica.distribs_start))
			plot_mod(self.oica.distribs_true,"True Model",True, mod_min, mod_max)

		plot_mod(self.oica.distribs_start,"Starting model",False)

##########################################################################################
