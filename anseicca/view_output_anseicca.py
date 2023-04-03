#!/usr/bin/python

# General purpose modules
import os
import sys
import math
import itertools
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# Custom modules
sys.path.append('../modules_common')
sys.path.append(os.path.expanduser('~/Research/code_own/cc_kern_inv/modules_common'))
import new_read_pickle_output as rpo

################################################################################
# PLOTTING FUNCTIONS
################################################################################

def plot_kernels():

	fig1=plt.figure()
	ax1=fig1.add_subplot(111)
	ax1.set_title("Initial misfit kernel: positive branch")
	cax1=ax1.pcolor(kcao_gx,kcao_gy,rapo.kcao_mfkp,cmap=plt.cm.jet)
	ax1.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'wd', markerfacecolor="None")
	if len(rapo.rc_xp)<13:
	# plot the station numbers as seen by the h13 module
		for i in range(len(rapo.rc_xp)):
			ax1.annotate(i, xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]))
			#pass
	ax1.tick_params(axis='both', labelsize=14)
	fig1.colorbar(cax1)

	fig2=plt.figure()
	ax2=fig2.add_subplot(111)
	ax2.set_title("Initial misfit kernel: negative branch")
	cax2=ax2.pcolor(kcao_gx,kcao_gy,rapo.kcao_mfkn,cmap=plt.cm.jet)
	ax2.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'wd', markerfacecolor="None")
	ax2.tick_params(axis='both', labelsize=14)
	# plot the actual station numbers of the real data set
	if len(rapo.rc_xp)<13:
		for i in range(len(rapo.rc_xp)):
			ax2.annotate(rec_id[i], xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]))
	fig2.colorbar(cax2)

################################################################################

class model_info:

	def __init__(self):

		#************************* Compute the models from the stored model coefficients ******************************

		# NB: we store only the model coefficients. These need to be applied to the model basis-set to get the model.

		mod_specs = {'mg': rapo.cg_somod_mg_specs, 'rg': rapo.cg_somod_rg_specs, 'rgr': rapo.cg_somod_rgr_specs, 'gg': rapo.cg_somod_gg_specs}
		sdist_type = {'mg': u1.somod.mult_gauss, 'rg': u1.somod.ringg, 'rgr': u1.somod.rgring, 'gg': u1.somod.gcover}

		xpoints = kcao_gx[0,:]
		ypoints = kcao_gy[:,0]

		alltheta_deg=np.arange(0,360,rapo.cg_somod_rg_specs['as'])
		alltheta=alltheta_deg*np.pi/180
		allcen = lambda specs: specs['r0']
		gg_loc=[]
		astep = mod_specs['gg']['ls']
		row_val=np.arange(-rapo.cg_dom_geom.box_len/2+(astep/2),rapo.cg_dom_geom.box_len/2,astep)
		xv,yv=np.array(np.meshgrid(row_val,row_val))
		for i in range(len(xv)):
			for j in range(len(xv[0])):
				gg_loc.append([xv[i,j],yv[i,j]])
		gg_loc=np.array(gg_loc)

		nbasis = {'egp': num_points**2,
					'rg': alltheta.size,
		 			'mg': len(mod_specs['mg']['r0']),
					'rgr': len(mod_specs['rgr']['r0']),
					'gg': int((rapo.cg_dom_geom.box_len/mod_specs['gg']['ls'])**2)}

		basis_loc = {'rg': alltheta,
		 				'mg': allcen(mod_specs['mg']),
						'rgr': allcen(mod_specs['rgr']),
						'gg': gg_loc}

		if len(gg_loc)!=nbasis['gg']:
			nbasis['gg']=len(gg_loc)

		basis = np.zeros((nbasis[rapo.cg_inv_mod_type],num_points,num_points))
		self.kcao_sditer = np.zeros((numit,num_points,num_points))

		if rapo.cg_inv_mod_type != 'egp':
			for k,theta in enumerate(basis_loc[rapo.cg_inv_mod_type]):
				basis[k,:,:]=sdist_type[rapo.cg_inv_mod_type](theta, mod_specs[rapo.cg_inv_mod_type])

		for i in range(numit):
			mc_iter = rapo.kcao_allit_mc[i]
			if rapo.cg_inv_mod_type == 'egp':
				self.kcao_sditer[i,:,:] = mc_iter.reshape(num_points,num_points)
			else:
				for k,theta in enumerate(basis_loc[rapo.cg_inv_mod_type]):
					self.kcao_sditer[i,:,:] += (mc_iter[k]**2) * basis[k,:,:]

		self.kcao_sdstart = self.kcao_sditer[0,:,:]
		# self.kcao_sdinv = self.kcao_sditer[-1,:,:]

		self.basis_cen = basis_loc[rapo.cg_inv_mod_type]

		#************************ Determine range of values for appropriate colour scales *****************************

		if not rapo.extdata:
			self.mod_min=min(np.amin(rapo.kcao_sdtrue),np.amin(self.kcao_sdstart),np.amin(rapo.kcao_sdinv))
			self.mod_max=max(np.amax(rapo.kcao_sdtrue),np.amax(self.kcao_sdstart),np.amax(rapo.kcao_sdinv))
			# self.mod_min=0
			# self.mod_max=28
		else:
			self.mod_min=min(np.amin(self.kcao_sdstart),np.amin(rapo.kcao_sdinv))
			self.mod_max=max(np.amax(self.kcao_sdstart),np.amax(rapo.kcao_sdinv))

# --------------------------------------------------------------------------------------------------

	def plot_source_models(self):

		diff_sizes=False

		# normalize all together
		# sdinv_norm = rapo.kcao_sdinv/self.mod_max
		# sdstart_norm = self.kcao_sdstart/self.mod_max
		# if not rapo.extdata:
		# 	sdtrue_norm = rapo.kcao_sdtrue/self.mod_max

		# normalize each one independently
		sdinv_norm = rapo.kcao_sdinv/np.amax(rapo.kcao_sdinv)
		sdstart_norm = self.kcao_sdstart/np.amax(self.kcao_sdstart)
		if not rapo.extdata:
			sdtrue_norm = rapo.kcao_sdtrue/np.amax(rapo.kcao_sdtrue)

		if not rapo.extdata:
			fig4=plt.figure()
			ax4=fig4.add_subplot(111, aspect='equal')
			ax4.set_title("True model")

			if rapo.kcao_sdtrue.shape[0]==self.kcao_sdstart.shape[0]:
				xpts_true = kcao_gx
				ypts_true = kcao_gy
			else:
				hlbox_outer = kcao_dx*(kcao_gx.shape[0]-1)
				ngp_outer = 2*kcao_gx.shape[0] - 1
				xobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
				yobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
				xpts_true, ypts_true = np.meshgrid(xobox, yobox)
				diff_sizes = True

			# cax4=ax4.pcolor(xpts_true,ypts_true,rapo.kcao_sdtrue,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
			# cax4=ax4.pcolor(xpts_true,ypts_true,sdtrue_norm,cmap=plt.cm.jet,vmin=0,vmax=1)
			cax4=ax4.pcolor(xpts_true,ypts_true,sdtrue_norm,cmap=plt.cm.Greys,vmin=0,vmax=1)
			ax4.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'c^', markerfacecolor="None")
			ax4.tick_params(axis='both', labelsize=14)
			ax4.set_xlabel('X [km]', fontsize=14)
			ax4.set_ylabel('Y [km]', fontsize=14)
			# for i in range(len(rapo.rc_xp)):
			# 	ax4.annotate(i, xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]), color='green')
			# plt.colorbar(cax4,ax=ax4, pad=0.1)
			print("Min and max values in True model: ", np.amin(rapo.kcao_sdtrue), np.amax(rapo.kcao_sdtrue))

		fig3=plt.figure()
		ax3=fig3.add_subplot(111, aspect='equal')
		ax3.set_title("Starting model")
		if diff_sizes:
			ax3.set_xlim(-hlbox_outer,hlbox_outer)
			ax3.set_ylim(-hlbox_outer,hlbox_outer)
		# try:
		# 	cax3=ax3.pcolor(kcao_gx,kcao_gy,self.kcao_sdstart,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
		# except NameError:
		# 	cax3=ax3.pcolor(kcao_gx,kcao_gy,self.kcao_sdstart,cmap=plt.cm.jet)
		# cax3=ax3.pcolor(kcao_gx,kcao_gy,sdstart_norm,cmap=plt.cm.jet,vmin=0,vmax=1)
		cax3=ax3.pcolor(kcao_gx,kcao_gy,sdstart_norm,cmap=plt.cm.Greys,vmin=0,vmax=1)
		ax3.tick_params(axis='both', labelsize=14)
		ax3.set_xlabel('X [km]', fontsize=14)
		ax3.set_ylabel('Y [km]', fontsize=14)
		# ax3.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'c^', markerfacecolor="None")
		# ax3.plot(self.basis_cen[:,0], self.basis_cen[:,1], '.', markerfacecolor="gray")
		for i in range(len(rapo.rc_xp)):
			ax3.annotate(rec_id[i], xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]), color='white')
		# cbar3 = plt.colorbar(cax3,ax=ax3)
		# cbar3.set_label("Normalized source strength", fontsize=14)
		print("Min and max values in the starting model: ", np.amin(self.kcao_sdstart), np.amax(self.kcao_sdstart))
		print(self.basis_cen.shape)

		fig5=plt.figure()
		ax5=fig5.add_subplot(111, aspect='equal')
		# ax5.set_title("Inversion result")
		#if diff_sizes:
		#	ax5.set_xlim(-hlbox_outer,hlbox_outer)
		#	ax5.set_ylim(-hlbox_outer,hlbox_outer)
		print("Min and max values in inverted result: ", np.amin(rapo.kcao_sdinv), np.amax(rapo.kcao_sdinv))
		# try:
		# 	cax5=ax5.pcolor(kcao_gx,kcao_gy,rapo.kcao_sdinv,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
		# 	#cax5=ax5.pcolor(kcao_gx,kcao_gy,rapo.kcao_sdinv,cmap=plt.cm.jet)
		# except NameError:
		# 	cax5=ax5.pcolor(kcao_gx,kcao_gy,rapo.kcao_sdinv,cmap=plt.cm.jet)
		# cax5=ax5.pcolor(kcao_gx,kcao_gy,sdinv_norm,cmap=plt.cm.jet,vmin=0,vmax=1)
		cax5=ax5.pcolor(kcao_gx,kcao_gy,sdinv_norm,cmap=plt.cm.Greys,vmin=0,vmax=1)
		ax5.tick_params(axis='both', labelsize=14)
		ax5.set_xlabel('X [km]', fontsize=14)
		ax5.set_ylabel('Y [km]', fontsize=14)
		ax5.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'c^', markerfacecolor="None")
		# for i in range(len(rapo.rc_xp)):
		# 		ax5.annotate(i, xy=((kcao_dx*rapo.rc_xp[i])+1,kcao_dx*rapo.rc_yp[i]), color='green')
		# cbar5 = plt.colorbar(cax5,ax=ax5) #, orientation='horizontal',fraction=0.04)
		# cbar5.set_label("Normalized source strength", fontsize=14)

# --------------------------------------------------------------------------------------------------

	def plot_vel_models(self):

		if not rapo.kcao_vel_acou[0] is None:
			# synthetic data case
			fig_vm=plt.figure()
			axvm1=fig_vm.add_subplot(111, aspect='equal')
			try:
				caxvm1 = axvm1.pcolor(kcao_gx, kcao_gy, rapo.kcao_vel_acou[0],cmap=plt.cm.seismic.reversed(),vmin=1.6,vmax=2.4)
			except TypeError:
				# the velocity model is larger than usual (usual size as defined by kcao_gx and kcao_gy)
				caxvm1 = axvm1.pcolor(rapo.cg_dom_geom.gx2, rapo.cg_dom_geom.gy2,rapo.kcao_vel_acou[0],cmap=plt.cm.seismic.reversed(),vmin=1.6,vmax=2.4)
			axvm1.set_title("Velocity model for test data")
			axvm1.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'c^', markerfacecolor="None")
			# axvm.tick_params(axis='both', labelsize=14)
			# for i in range(len(rapo.rc_xp)):
			# 	axvm1.annotate(i, xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]), color='green')
			axvm1.tick_params(axis='both', labelsize=14)
			axvm1.set_xlabel('X [km]', fontsize=14)
			axvm1.set_ylabel('Y [km]', fontsize=14)
			# cbar = plt.colorbar(caxvm1,ax=axvm1)
			# cbar.set_label("km/s", fontsize=14)

		fig_vm=plt.figure()
		axvm2=fig_vm.add_subplot(111, aspect='equal')
		# axvm2.set_title("Velocity model for inversion")
		caxvm2 = axvm2.pcolor(kcao_gx, kcao_gy, rapo.kcao_vel_acou[1],cmap=plt.cm.seismic.reversed(),vmin=1.6,vmax=2.4)
		axvm2.plot(kcao_dx*rapo.rc_xp, kcao_dx*rapo.rc_yp, 'c^', markerfacecolor="None")
		# axvm.tick_params(axis='both', labelsize=14)
		# for i in range(len(rapo.rc_xp)):
		# 	axvm2.annotate(i, xy=(kcao_dx*rapo.rc_xp[i],kcao_dx*rapo.rc_yp[i]), color='green')
		axvm2.tick_params(axis='both', labelsize=14)
		# plt.colorbar(caxvm2,ax=axvm2)

################################################################################

class inversion_progress_plots:

	def __init__(self):

		#******** Determine the paths (receiver pairs) for which waveform fits have improved through inversion ********

		self.bs_deltad=0.5 #2.5 #0.5
		self.bs_deltat=0.5
		# this is the bin size for histogram of misfits (see functions "hist_deltad" or "hist_deltat")

	# --------------------------------------------------------------------------------------------------

	def hist_deltad(self):

		print("CHECK 04 Nov:")
		print(rapo.kcao_flit_indmis_p[0].size)
		print(rapo.kcao_flit_indmis_p[-1].size)

		bad_thresh = 1.3*self.bs_deltad
		good_thresh = self.bs_deltad/2 # DO NOT CHANGE THIS. CHANGE ONLY THE BINSIZE (self.bs_deltad)

		bad_before_p = np.where(abs(rapo.kcao_flit_indmis_p[0])>bad_thresh)[0]
		bad_before_n = np.where(abs(rapo.kcao_flit_indmis_n[0])>bad_thresh)[0]

		try:
			good_after_p = np.where(abs(rapo.kcao_flit_indmis_p[-1])<good_thresh)[0]
		except IndexError:
			good_after_p = np.array([])

		try:
			good_after_n = np.where(abs(rapo.kcao_flit_indmis_n[-1])<good_thresh)[0]
		except IndexError:
			good_after_n = np.array([])

		nga_total = good_after_p.size + good_after_n.size
		good_frac = float(nga_total)/(2*npairs)
		print("Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(nga_total,100*good_frac))

		irpi_pos = np.intersect1d(bad_before_p, good_after_p)
		irpi_neg = np.intersect1d(bad_before_n, good_after_n)
		# irpi_xxx -> "index-of-receiver-pair-improved_branch"
		irpi_both = np.intersect1d(irpi_pos,irpi_neg)

		try:
			really_good_p = np.where(abs(rapo.kcao_flit_indmis_p[1])<good_thresh/8)[0]
		except IndexError:
			really_good_p = np.array([])
		try:
			really_good_n = np.where(abs(rapo.kcao_flit_indmis_n[1])<good_thresh/8)[0]
		except IndexError:
			really_good_n = np.array([])
		really_good = np.intersect1d(really_good_p,really_good_n)

		#********************************** some extra calculations for info ********************************************

		# compute ACTUAL distances - based on the coordinates file, NOT effective distances from kcao

		adrp=np.zeros((nrecs,nrecs))
		# adrp -> actual_distance_receiver_pairs

		if input_files:
			for b,brec in enumerate(rec_no[:-1]):
				urecs=rec_no[b+1:]
				x1=xloc[np.searchsorted(rnum,brec)]
				y1=yloc[np.searchsorted(rnum,brec)]
				x2=xloc[np.searchsorted(rnum,urecs)]
				y2=yloc[np.searchsorted(rnum,urecs)]
				adrp[b+1:,b]=np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
				adrp[b,b+1:]=adrp[b+1:,b]

		rp_orig=list(range(npairs_max))
		rp_kcao=list(range(npairs_max))
		rp_dist=list(range(npairs_max))
		cp=0
		for x in itertools.combinations(range(nrecs),2):
			rp_kcao[cp]=x[::-1]
			# reversal of the pair ordering is required to match what is done in h13 (lower triangular matrix)
			rp_orig[cp]=(rec_no[x[1]],rec_no[x[0]])
			rp_dist[cp]=adrp[x]
			cp+=1

		if nrecs<10:
			#spl = np.where(rapo.kcao_flit_indmis_n[0]<-0.7)[0]
			print("List of GOOD fits after inversion (regardless of good or bad BEFORE):")
			print("RP (kcao)\t\tRP(actual)\t\tInterstation distance (km)")
			for k in really_good:
				print(rp_kcao[k], "\t\t", rp_orig[k], "\t\t", rp_dist[k])
			# print("List of fits IMPROVED by inversion (both branch fits were poor BEFORE inversion):")
			# print("RP (kcao)\tRP(actual)")
			# for k in irpi_both:
			# 	print(rp_kcao[k], "\t\t", rp_orig[k])
			# print("SPECIAL: ")

		# **************************************** End of extra calculations and info ********************************************

		# determine appropriate histogram bins such that "0" is a bin center
		mv1=max(max(rapo.kcao_flit_indmis_p[0]),max(rapo.kcao_flit_indmis_n[0]))
		mv2=min(min(rapo.kcao_flit_indmis_p[0]),min(rapo.kcao_flit_indmis_n[0]))
		maxval=np.round(max(abs(mv1),abs(mv2)))
		print("Max val for DELTA-D histograms is: ", maxval)
		hbe_p=np.arange(self.bs_deltad/2,maxval+self.bs_deltad,self.bs_deltad)
		hbe_n=-1*hbe_p[::-1]
		hbe = np.hstack((hbe_n,hbe_p))
		# hbe -> histogram_bin_edges

		ntot="$N_{tot}$\n= %d" %(npairs)

		#fig = plt.figure(figsize=(5.5,10))
		#axh_p, axh_n = fig.subplots(2,1,sharex=True,sharey=True)
		fig = plt.figure()
		ncen="$N_{cen}$\n= %d" %(good_after_p.size)
		axh_p, axh_n = fig.subplots(1,2,sharex=True,sharey=True)
		axh_p.hist(rapo.kcao_flit_indmis_p[0],bins=hbe,edgecolor='black')
		axh_p.hist(rapo.kcao_flit_indmis_p[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_p.set_xlabel(r'$\Delta d$')
		axh_p.set_xlabel(r'$\ln \left( E^{obs}/E^{syn} \right)$', fontsize=12)
		#axh_p.set_ylabel("No. of pairs")#, fontsize=14)
		axh_p.set_title("Positive branch")#, fontsize=18)
		axh_p.tick_params(labelsize=14)
		axh_p.text(0.7, 0.8, ntot, transform=axh_p.transAxes)
		axh_p.text(0.7, 0.7, ncen, transform=axh_p.transAxes)

		ncen="$N_{cen}$\n= %d" %(good_after_n.size)
		axh_n.hist(rapo.kcao_flit_indmis_n[0],bins=hbe,edgecolor='black')
		axh_n.hist(rapo.kcao_flit_indmis_n[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_n.set_xlabel(r'$\Delta d$')#, fontsize=18)
		axh_n.set_xlabel(r'$\ln \left( E^{obs}/E^{syn} \right)$', fontsize=12)
		#axh_n.set_ylabel("No. of pairs")#, fontsize=14)
		axh_n.set_title("Negative branch")#, fontsize=18)
		axh_n.tick_params(labelsize=14)
		axh_n.text(0.1, 0.8, ntot, transform=axh_n.transAxes)
		axh_n.text(0.1, 0.7, ncen, transform=axh_n.transAxes)

	# --------------------------------------------------------------------------------------------------

	def hist_deltat(self):

		# compute waveform envelopes
		kcao_obsenv=np.abs(ss.hilbert(rapo.kcao_obscross, axis=0))
		kcao_synenv_init=np.abs(ss.hilbert(rapo.kcao_syncross_init, axis=0))
		kcao_synenv_final=np.abs(ss.hilbert(rapo.kcao_syncross_final, axis=0))

		# indices of waveform windows (each branch)
		kcao_negl = rapo.kcao_win_ind[0,:,:]
		kcao_negr = rapo.kcao_win_ind[1,:,:]
		kcao_posl = rapo.kcao_win_ind[2,:,:]
		kcao_posr = rapo.kcao_win_ind[3,:,:]

		dt_float = kcao_t[1] - kcao_t[0]
		pow10=np.abs(math.floor(np.log10(dt_float)))
		dt = np.round(dt_float,pow10)
		print("dt: ", dt)

		deltat_pos_before=np.zeros(npairs)
		deltat_neg_before=np.zeros(npairs)
		deltat_pos_after=np.zeros(npairs)
		deltat_neg_after=np.zeros(npairs)

		cp=0 # cp stands for count_pair

		# get the traveltime discrepancies
		for j in range(nrecs-1):
			for i in range(j+1,nrecs):
				if kcao_used_pairs[i,j]==0:
					pass
				else:
					# do positive branch
					ind_peak_obs = np.argmax(kcao_obsenv[kcao_posl[i,j]:kcao_posr[i,j],i,j])
					ind_peak_isyn = np.argmax(kcao_synenv_init[kcao_posl[i,j]:kcao_posr[i,j],i,j])
					ind_peak_fsyn = np.argmax(kcao_synenv_final[kcao_posl[i,j]:kcao_posr[i,j],i,j])
					deltat_pos_before[cp] = (ind_peak_obs - ind_peak_isyn)*dt
					deltat_pos_after[cp] = (ind_peak_obs - ind_peak_fsyn)*dt
					# do negative branch
					ind_peak_obs = np.argmax(kcao_obsenv[kcao_negl[i,j]:kcao_negr[i,j],i,j])
					ind_peak_isyn = np.argmax(kcao_synenv_init[kcao_negl[i,j]:kcao_negr[i,j],i,j])
					ind_peak_fsyn = np.argmax(kcao_synenv_final[kcao_negl[i,j]:kcao_negr[i,j],i,j])
					deltat_neg_before[cp] = (ind_peak_obs - ind_peak_isyn)*dt
					deltat_neg_after[cp] = (ind_peak_obs - ind_peak_fsyn)*dt

					cp+=1

		assert cp==npairs

		good_thresh = self.bs_deltat/2 # DO NOT CHANGE THIS. CHANGE ONLY THE BINSIZE

		cenbin_p=np.where(abs(deltat_pos_after)<good_thresh)[0]
		cenbin_n=np.where(abs(deltat_neg_after)<good_thresh)[0]
		cenbin_tot = cenbin_p.size + cenbin_n.size
		cenbin_frac = float(cenbin_tot)/(2*npairs)

		# determine appropriate histogram bins such that "0" is a bin center
		mv1=max(max(deltat_pos_before),max(deltat_neg_before))
		mv2=min(min(deltat_pos_before),min(deltat_neg_before))
		maxval=np.round(max(abs(mv1),abs(mv2)))
		print(mv1,mv2)
		print("Max val for DELTA-T histograms is: ", maxval)
		hbe_p=np.arange(self.bs_deltat/2,maxval+self.bs_deltat,self.bs_deltat)
		hbe_n=-1*hbe_p[::-1]
		hbe = np.hstack((hbe_n,hbe_p))
		# hbe -> histogram_bin_edges

		print("Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(cenbin_tot,100*cenbin_frac))
		ntot="$N_{tot}$\n= %d" %(npairs)
		fig = plt.figure()
		try:
			axh_p, axh_n = fig.subplots(1,2,sharex=True,sharey=True)
		except AttributeError:
			fig, axh = plt.subplots(1,2,sharex=True,sharey=True)
			axh_p = axh[0]
			axh_n = axh[1]

		ncen="$N_{cen}$\n= %d" %(cenbin_p.size)
		axh_p.hist(deltat_pos_before,bins=hbe,edgecolor='black')
		axh_p.hist(deltat_pos_after,bins=hbe,histtype='step',linewidth='1.5')
		axh_p.set_xlabel(r'$\Delta t$')
		axh_p.set_ylabel("No. of pairs")#, fontsize=14)
		axh_p.set_title("Positive branch")#, fontsize=18)
		axh_p.text(0.7, 0.8, ntot, transform=axh_p.transAxes)
		axh_p.text(0.7, 0.7, ncen, transform=axh_p.transAxes)

		ncen="$N_{cen}$\n= %d" %(cenbin_n.size)
		axh_n.hist(deltat_neg_before,bins=hbe,edgecolor='black')
		axh_n.hist(deltat_neg_after,bins=hbe,histtype='step',linewidth='1.5')
		axh_n.set_xlabel(r'$\Delta t$')#, fontsize=18)
		#axh_n.set_ylabel("No. of pairs")#, fontsize=14)
		axh_n.set_title("Negative branch")#, fontsize=18)
		#axh_n.tick_params(axis='x')#, labelsize=14)
		axh_n.text(0.1, 0.8, ntot, transform=axh_n.transAxes)
		axh_n.text(0.1, 0.7, ncen, transform=axh_n.transAxes)

	# --------------------------------------------------------------------------------------------------

	def chi_iter(self):

		print("Misfits: ", rapo.kcao_allit_misfit)
		print("Final misfit: ", rapo.kcao_allit_misfit[-1])

		nchid = rapo.kcao_allit_misfit/np.amax(rapo.kcao_allit_misfit)

		its=range(numit)
		fig=plt.figure()
		ax=fig.add_subplot(111)
		#ax.set_title("Inversion progress: total misfit", fontsize=18)
		try:
			ax.plot(its,nchid,'-o')
		except ValueError:
			ax.plot(its[:-1],nchid,'-o')
		ax.xaxis.set_ticks(its)
		ax.ticklabel_format(axis='y',style='scientific',scilimits=(-2,2))
		ax.set_ylabel(r"$\chi_d(m_k)$", fontsize=14)
		ax.set_xlabel("k, iteration number", fontsize=14)
		ax.tick_params(labelsize=14)
		#plt.xticks(fontsize=14)

	# --------------------------------------------------------------------------------------------------

	def mod_iter(self, kcao_sditer, mod_min, mod_max):

		fig=plt.figure()
		#pub = numit-1
		#for p in range(pub):
		for p in range(numit):
			#it=p+1
			#spname = "k=%d" %(it)
			spname = "k=%d" %(p)
			try:
				axsp=fig.add_subplot(4,4,p+1) #,aspect='equal')
			except ValueError:
				print("Problem plotting inversion result for iteration >= %d. Mismatch between number of iterations and number of subplots." %(p))
				return
			cax=axsp.pcolor(kcao_gx,kcao_gy,kcao_sditer[p,:,:],cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
			axsp.text(0.8,0.85,spname,transform=axsp.transAxes,color='white')
			#axsp.set_title(spname)
			#if p==pub-1:
			#	plt.colorbar(cax,ax=axsp,orientation="horizontal")

###############################################################################################################

class plot_misc_info:

	def __init__(self):

		print(rapo.kcao_pss[0])
		fig=plt.figure()
		ax=fig.add_subplot(111)
		# ax.plot(rapo.kcao_pss)
		ax.plot(np.fft.fftshift(rapo.kcao_sig_char.fhz), np.fft.fftshift(rapo.kcao_pss))

###############################################################################################################

def plot_Lcurve():

	ind_min_misfit=np.argmin(gam_sorted_misfit)
	ind_opt=np.argmin(gam_sorted_misfit_modnorm)

	print("Model norm: ", gam_sorted_modnorm)
	print("Misfit: ", gam_sorted_misfit)
	print("Norm + Misfit: ", gam_sorted_misfit_modnorm)
	print("Elbow of L-curve at gamma = %f" %(sorted_gamma[ind_opt]))
	print("Minimum misfit at gamma = %f" %(sorted_gamma[ind_min_misfit]))

	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(gam_sorted_modnorm,gam_sorted_misfit)
	for p in range(len(filelist)):
		gval = np.sort(gamma_files)[p]
		if gval>=1 or gval==0:
			plabel=r"$\gamma = %d$" %(gval)
		else:
			# dp=abs(int(np.floor(np.log10(gval))))
			dp=4
			plabel=r"$\gamma = %.*f$" %(dp,gval)
		ax.plot(gam_sorted_modnorm[p],gam_sorted_misfit[p],'o',label=plabel)
	ax.legend()
	ax.set_xlabel("Model norm (relative)")
	ax.set_ylabel("Misfit")

###############################################################################################################

def info_code_run():

	print("\nCROSSS-CORRELATION MODELLING THEORY\n")
	print("Source direction: ", rapo.cg_ccmt.src_dir)
	print("Green tensor component: ", rapo.cg_ccmt.GTC)

	print("\nMODELLING TYPE\n")
	try:
		print("Modelling type for generating data (synthetic tests ONLY): ", rapo.cg_tru_mdlng_type)
	except AttributeError:
		pass
	print("Modelling type for inversion: ", rapo.cg_inv_mdlng_type)

	print("\nMODELLING DOMAIN GEOMETRY\n")
	print("dx (km): ", rapo.cg_dom_geom.dx)
	print("zmax (km): ", rapo.cg_dom_geom.zmax)
	print("size of box (km): ", rapo.cg_dom_geom.box_len)

	print("\nMODELLING PARAMETERIZATION\n")
	print("MG:")
	print("Width: ", rapo.cg_somod_mg_specs['w'])
	print("Centre(s): ", rapo.cg_somod_mg_specs['r0'])

	print("\nRG:")
	print("Radius: ", rapo.cg_somod_rg_specs['r'])
	print("Width: ", rapo.cg_somod_rg_specs['w'])
	print("Angular sampling: ", rapo.cg_somod_rg_specs['as'])

	print("\nRGR:")
	print("Radius: ", rapo.cg_somod_rgr_specs['r'])
	print("Width: ", rapo.cg_somod_rgr_specs['w'])

	print("\nGG:")
	print("Width: ", rapo.cg_somod_gg_specs['w'])
	print("Linear separation: ", rapo.cg_somod_gg_specs['ls'])

	print("\nSIGNAL PARAMETERS\n")
	print("dt (s): ", rapo.kcao_sig_char.dt)
	print("nsam: ", rapo.kcao_sig_char.nsam)
	print("cf (Hz): ", rapo.kcao_sig_char.cf)
	print("fsigma (Hz): ", rapo.kcao_sig_char.fsigma)
	print("lf (Hz): ", rapo.kcao_sig_char.lf)
	print("hf (Hz): ", rapo.kcao_sig_char.hf)
	print("alpha parameter (Tukey window): ", rapo.kcao_sig_char.altukey)
	print("Power-spectrum type: ", rapo.kcao_sig_char.pst)

	print("\nSCALAR MODELLING PARAMETERS\n")
	print("Density (gm/cc): ", rapo.cg_scal_mod.rho_scal2D)
	print("Wavespeed (km/s): ", rapo.cg_scal_mod.wavspeed_scal2D)

	print("\nINVERSION RELATED CHOICES\n")
	print("Gamma for damping: ", kcao_gamma)

	try:
		print("\nCHOICES PERTAINING TO GENERATION OF INTERNAL (SYNTHETIC) DATA\n")
		print("Amplitude of added noise (%): ", rapo.cg_syn_data.noise_level)
		print("Frequency band of added noise (Hz): ", rapo.cg_syn_data.noise_band)
	except AttributeError:
		pass

	print("\nRUN-TIME INFORMATION\n")
	print("No. of iterations: ", numit)

###############################################################################################################

def usage():
	print("What do you want to do ?")
	print("Choose from the following options")
	print("		 1 - View models")
	print("		 2 - View kernels")
	print("		 3 - View inversion progress")
	print("		 4 - View individual waveforms")
	print("		 5 - Write SAC output (all waveforms)")
	print("		 6 - Miscellaneous")
	print("          7 - Exit")

################################################################################
# MAIN PROGRAM
################################################################################

inarg1=sys.argv[1]
try:
	coordfile=sys.argv[2] #"EXAMPLES/coordinates_receivers_h13format.csv"
	input_files = True
except IndexError:
	input_files = False

if os.path.isdir(inarg1):
	# used, for example, to plot an L-curve: when inversion is done with several different values of the damping parameter, gamma
	filelist=[os.path.join(inarg1,n) for n in os.listdir(inarg1) if n.endswith('.pckl')]
	nrecs_files=np.zeros(len(filelist))
	gamma_files=np.zeros(len(filelist))
	misfit_files=np.zeros(len(filelist))
	modnorm_files=np.zeros(len(filelist))
elif os.path.isfile(inarg1):
	filelist=[inarg1]

#********************************* read coordinates file if provided *********************************

if input_files:
	rnum, rid, xloc, yloc = u1.read_station_file(coordfile)

#********************************* Read the pickle file(s) **************************************************
for p, pfile in enumerate(filelist):

	print("Reading ", pfile)

	rapo = rpo.read_anseicca_pickle(pfile)

	cg_dom_geom=rapo.cg_dom_geom
	# variable required for use in u1

	if p==0:
		# Custom modules
		import cctomo_utils1 as u1

	#************************************** Extract quantities required in this code **********************************************

	rec_no = [csi[0] for csi in rapo.chosen_st_info]
	rec_id = [csi[1] for csi in rapo.chosen_st_info]

	num_points = rapo.cg_dom_geom.gx.shape[0]
	assert num_points == rapo.cg_dom_geom.ngp_box

	kcao_dx = rapo.cg_dom_geom.dx
	kcao_gx = rapo.cg_dom_geom.gx
	kcao_gy = rapo.cg_dom_geom.gy
	kcao_gamma = rapo.cg_invc.gamma_inv

	# print(rapo.kcao_sig_char['r0'])
	kcao_t = rapo.kcao_sig_char.tt
	kcao_sig_char = rapo.kcao_sig_char
	kcao_used_pairs = rapo.kcao_used_pairs

	nrecs = len(rapo.rc_xp)
	npairs = rapo.kcao_flit_indmis_p[0].size
	npairs_max = int(nrecs*(nrecs-1)/2)
	numit = len(rapo.kcao_allit_mc)

	mpo = model_info()

	if len(filelist)>1:
		nrecs_files[p]=len(rapo.rc_xp)
		gamma_files[p]=kcao_gamma
		misfit_files[p]=rapo.kcao_allit_misfit[-1]
		delta_m_coeff = rapo.kcao_allit_mc[-1] - rapo.kcao_allit_mc[0]
		mod_norm = np.sqrt(np.sum(np.square(delta_m_coeff)))
		# delta_m_act = rapo.kcao_sdinv - mpo.kcao_sdstart
		# mod_norm = np.sqrt(np.sum(np.square(delta_m_act.flatten())))
		modnorm_files[p]=mod_norm
	else:
		#  Print info about code run
		info_code_run()

	#************************************** Perform necessary checks **********************************************

	if rapo.extdata:
		assert npairs <= npairs_max
	else:
		print(npairs, npairs_max)
		assert npairs == npairs_max

	try:
		assert rapo.kcao_obscross.shape[1]==rapo.kcao_obscross.shape[2]
		assert rapo.kcao_syncross_init.shape[1]==rapo.kcao_syncross_init.shape[2]
		assert rapo.kcao_syncross_final.shape[1]==rapo.kcao_syncross_final.shape[2]
	except AssertionError:
		raise SystemExit("Problem with stored waveform matrices")

	assert nrecs == rapo.kcao_obscross.shape[1]
	assert npairs == rapo.kcao_flit_indmis_n[0].size
	# assert npairs == rapo.kcao_flit_indmis_p[1].size
	# assert len(rapo.kcao_allit_misfit) == len(rapo.kcao_allit_mc)

#********************************* End of FOR loop over pickle file(s) **************************************************

# Custom modules
# import cctomo_utils1 as u1

if len(filelist)>1:
	# usually used to compare inversions with different settings (e.g. L-curve plotting)

	if len(np.unique(nrecs_files))>1:
		sys.exit("The different pickles have different number of receivers - script terminated.")

	sortind = np.argsort(gamma_files)
	sorted_gamma = np.sort(gamma_files)
	gam_sorted_misfit = misfit_files[sortind]
	gam_sorted_modnorm = modnorm_files[sortind]
	gam_sorted_misfit_modnorm = np.sqrt(gam_sorted_misfit**2 + gam_sorted_modnorm**2)
	print("Sorted gamma values: ", sorted_gamma)
	plot_Lcurve()
	plt.show()

else:
	# misfit_start = 0.5*np.sum(np.square(rapo.kcao_flit_indmis_p[0]) + np.square(rapo.kcao_flit_indmis_n[0]))
	# misfit_end = 0.5*np.sum(np.square(rapo.kcao_flit_indmis_p[-1]) + np.square(rapo.kcao_flit_indmis_n[-1]))
	# if np.round(misfit_start,6) != np.round(rapo.kcao_allit_misfit[0],6) or np.round(misfit_end,6) != np.round(rapo.kcao_allit_misfit[-1],6):
	# 	print("Initial misfit with and without errors: ", rapo.kcao_allit_misfit[0], misfit_start)
	# 	print("Final misfit with and without errors: ", rapo.kcao_allit_misfit[-1], misfit_end)

	desired_output = 0
	while desired_output != 7:
		usage()
		desired_output = int(input("Enter your choice (number) here: "))
		if desired_output>=7:
			sys.exit('Thank you')
		elif desired_output==1:
			mpo.plot_source_models()
			if not (rapo.kcao_vel_acou[0] is None and rapo.kcao_vel_acou[1] is None):
				mpo.plot_vel_models()
		elif desired_output==2:
			plot_kernels()
		elif desired_output==3:
			ippo = inversion_progress_plots()
			ippo.mod_iter(mpo.kcao_sditer, mpo.mod_min, mpo.mod_max)
			ippo.chi_iter()
			ippo.hist_deltad()
			ippo.hist_deltat()
		elif desired_output==4:
			wpo = rpo.waveform_plots(rapo)
			usrc_ww=int(input("See actual waveforms (1) or waveform envelopes (2) ? "))
			usr_int=True
			while usr_int:
				usrc_nn=input("Enter receiver/station numbers (any other key for back to main): ")
				if len(usrc_nn.split())==2:
					m=int(usrc_nn.split()[0])
					n=int(usrc_nn.split()[1])
					if usrc_ww==1:
						# wpo.plot_waveforms_flit(m,n)
						wpo.plot_waveforms_oneit(m,n,-1)
					elif usrc_ww==2:
						wpo.plot_envelopes_flit(m,n)
						wpo.plot_envelopes_oneit(m,n,-1)
					plt.show()
				else:
					usr_int=False
		elif desired_output==5:
			try:
				import obspy.core as oc
			except ModuleNotFoundError:
				raise SystemExit("Requires 'obspy' module. Please run in an environment containing the obspy module.")
			wwoo = rpo.write_waveform_output(rapo, nrecs)
			wwoo.sac_output(oc, rec_id, kcao_sig_char)
		elif desired_output==6:
			pmioo = plot_misc_info()
		plt.show()
