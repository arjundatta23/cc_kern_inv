#!/usr/bin/python

# General purpose modules
import os
import sys
import gzip
import pickle
import itertools
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

sys.path.append('../modules_common')

################################################################################
# PLOTTING FUNCTIONS
################################################################################

def plot_kernels():

	fig1=plt.figure()
	ax1=fig1.add_subplot(111)
	ax1.set_title("Initial misfit kernel: positive branch")
	cax1=ax1.pcolor(kcao_gx,kcao_gy,kcao_mfkp,cmap=plt.cm.jet)
	ax1.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	if len(rc_xp)<13:
	# plot the station numbers as seen by the h13 module
		for i in range(len(rc_xp)):
			ax1.annotate(i, xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]))
			#pass
	ax1.tick_params(axis='both', labelsize=14)
	fig1.colorbar(cax1)

	fig2=plt.figure()
	ax2=fig2.add_subplot(111)
	ax2.set_title("Initial misfit kernel: negative branch")
	cax2=ax2.pcolor(kcao_gx,kcao_gy,kcao_mfkn,cmap=plt.cm.jet)
	ax2.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	ax2.tick_params(axis='both', labelsize=14)
	# plot the actual station numbers of the real data set
	if len(rc_xp)<13:
		for i in range(len(rc_xp)):
			ax2.annotate(rchosen[i], xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]))
	fig2.colorbar(cax2)

################################################################################

class model_info:

	def __init__(self):

		#************************* Compute the models from the stored model coefficients ******************************

		# NB: we store only the model coefficients. These need to be applied to the model basis-set to get the model.

		xpoints = kcao_gx[0,:]
		ypoints = kcao_gy[:,0]

		alltheta_deg=np.arange(0,360,10)
		alltheta=alltheta_deg*np.pi/180

		basis = np.zeros((alltheta.size,num_points,num_points))
		self.kcao_sditer = np.zeros((numit,num_points,num_points))

		for k,theta in enumerate(alltheta):
			basis[k,:,:]=u1.somod.ringg(theta, cg_somod_rg_specs)
			# if (not reald) and (len(trumod.shape)==1):
			# 	kcao_sdtrue += mc_true[k]*basis[k,:,:]

		for i in range(numit):
			mc_iter = kcao_allit_mc[i]
			for k,theta in enumerate(alltheta):
				self.kcao_sditer[i,:,:] += mc_iter[k]*basis[k,:,:]

		self.kcao_sdstart = self.kcao_sditer[0,:,:]
		self.kcao_sdinv = self.kcao_sditer[-1,:,:]

		#************************ Determine range of values for appropriate colour scales *****************************

		if not reald:
			self.mod_min=min(np.amin(kcao_sdtrue),np.amin(self.kcao_sdstart),np.amin(self.kcao_sdinv))
			self.mod_max=max(np.amax(kcao_sdtrue),np.amax(self.kcao_sdstart),np.amax(self.kcao_sdinv))
		else:
			self.mod_min=min(np.amin(self.kcao_sdstart),np.amin(self.kcao_sdinv))
			self.mod_max=max(np.amax(self.kcao_sdstart),np.amax(self.kcao_sdinv))

# --------------------------------------------------------------------------------------------------

	def plot_models(self):

		diff_sizes=False
		if not reald:
			fig4=plt.figure()
			ax4=fig4.add_subplot(111)
			ax4.set_title("True model")

			if kcao_sdtrue.shape[0]==self.kcao_sdstart.shape[0]:
				xpts_true = kcao_gx
				ypts_true = kcao_gy
			else:
				hlbox_outer = kcao_dx*(kcao_gx.shape[0]-1)
				ngp_outer = 2*kcao_gx.shape[0] - 1
				xobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
				yobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
				xpts_true, ypts_true = np.meshgrid(xobox, yobox)
				diff_sizes = True

			cax4=ax4.pcolor(xpts_true,ypts_true,kcao_sdtrue,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
			ax4.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
			ax4.tick_params(axis='both', labelsize=14)
			for i in range(len(rc_xp)):
				ax4.annotate(i, xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]), color='green')
			plt.colorbar(cax4,ax=ax4)
			print("Min and max values in True model: ", np.amin(kcao_sdtrue), np.amax(kcao_sdtrue))

		fig3=plt.figure()
		ax3=fig3.add_subplot(111)
		ax3.set_title("Starting model")
		if diff_sizes:
			ax3.set_xlim(-hlbox_outer,hlbox_outer)
			ax3.set_ylim(-hlbox_outer,hlbox_outer)
		try:
			cax3=ax3.pcolor(kcao_gx,kcao_gy,self.kcao_sdstart,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
		except NameError:
			cax3=ax3.pcolor(kcao_gx,kcao_gy,self.kcao_sdstart,cmap=plt.cm.jet)
		ax3.tick_params(axis='both', labelsize=14)
		ax3.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
		for i in range(len(rc_xp)):
			ax3.annotate(rchosen[i], xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]), color='white')
		plt.colorbar(cax3,ax=ax3)

		fig5=plt.figure()
		ax5=fig5.add_subplot(111) #, aspect='equal')
		ax5.set_title("Inversion result")
		#if diff_sizes:
		#	ax5.set_xlim(-hlbox_outer,hlbox_outer)
		#	ax5.set_ylim(-hlbox_outer,hlbox_outer)
		print("Min and max values in inverted result: ", np.amin(self.kcao_sdinv), np.amax(self.kcao_sdinv))
		try:
			cax5=ax5.pcolor(kcao_gx,kcao_gy,self.kcao_sdinv,cmap=plt.cm.jet,vmin=self.mod_min,vmax=self.mod_max)
			#cax5=ax5.pcolor(kcao_gx,kcao_gy,self.kcao_sdinv,cmap=plt.cm.jet,vmin=0.0,vmax=1.25)
			#cax5=ax5.pcolor(kcao_gx,kcao_gy,self.kcao_sdinv,cmap=plt.cm.jet)
		except NameError:
			cax5=ax5.pcolor(kcao_gx,kcao_gy,self.kcao_sdinv,cmap=plt.cm.jet)
		ax5.tick_params(axis='both', labelsize=14)
		ax5.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
		plt.colorbar(cax5,ax=ax5) #, orientation='horizontal',fraction=0.04)

################################################################################

class inversion_progress_plots:

	def __init__(self):

		#******** Determine the paths (receiver pairs) for which waveform fits have improved through inversion ********

		self.bs=0.5 #2.5 #0.5
		# this is the bin size for histogram of misfits (see functions "hist_deltad" or "hist_deltat")

		self.npairs=int(nrecs*(nrecs-1)/2)

	# --------------------------------------------------------------------------------------------------

	def hist_deltad(self):

		bad_thresh = 1.3*self.bs
		good_thresh = self.bs/2 # DO NOT CHANGE THIS. CHANGE ONLY THE self.bs

		bad_before_p = np.where(abs(kcao_flit_indmis_p[0])>bad_thresh)[0]
		bad_before_n = np.where(abs(kcao_flit_indmis_n[0])>bad_thresh)[0]

		try:
			good_after_p = np.where(abs(kcao_flit_indmis_p[-1])<good_thresh)[0]
		except IndexError:
			good_after_p = np.array([])

		try:
			good_after_n = np.where(abs(kcao_flit_indmis_n[-1])<good_thresh)[0]
		except IndexError:
			good_after_n = np.array([])

		nga_total = good_after_p.size + good_after_n.size
		good_frac = float(nga_total)/(2*self.npairs)
		print("Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(nga_total,100*good_frac))

		irpi_pos = np.intersect1d(bad_before_p, good_after_p)
		irpi_neg = np.intersect1d(bad_before_n, good_after_n)
		# irpi_xxx -> "index-of-receiver-pair-improved_branch"
		irpi_both = np.intersect1d(irpi_pos,irpi_neg)

		try:
			really_good_p = np.where(abs(kcao_flit_indmis_p[1])<good_thresh/8)[0]
		except IndexError:
			really_good_p = np.array([])
		try:
			really_good_n = np.where(abs(kcao_flit_indmis_n[1])<good_thresh/8)[0]
		except IndexError:
			really_good_n = np.array([])
		really_good = np.intersect1d(really_good_p,really_good_n)

		#********************************** some extra calculations for info ********************************************

		# compute ACTUAL distances - based on the coordinates file, NOT effective distances from kcao

		adrp=np.zeros((nrecs,nrecs))
		# adrp -> actual_distance_receiver_pairs

		if input_files:
			for b,brec in enumerate(rchosen[:-1]):
				urecs=rchosen[b+1:]
				x1=xloc[np.searchsorted(rnum,brec)]
				y1=yloc[np.searchsorted(rnum,brec)]
				x2=xloc[np.searchsorted(rnum,urecs)]
				y2=yloc[np.searchsorted(rnum,urecs)]
				adrp[b+1:,b]=np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
				adrp[b,b+1:]=adrp[b+1:,b]

		rp_orig=list(range(self.npairs))
		rp_kcao=list(range(self.npairs))
		rp_dist=list(range(self.npairs))
		cp=0
		for x in itertools.combinations(range(nrecs),2):
			rp_kcao[cp]=x[::-1]
			# reversal of the pair ordering is required to match what is done in h13 (lower triangular matrix)
			rp_orig[cp]=(rchosen[x[1]],rchosen[x[0]])
			rp_dist[cp]=adrp[x]
			cp+=1

		if nrecs<30:
			#spl = np.where(kcao_flit_indmis_n[0]<-0.7)[0]
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
		mv1=max(max(kcao_flit_indmis_p[0]),max(kcao_flit_indmis_n[0]))
		mv2=min(min(kcao_flit_indmis_p[0]),min(kcao_flit_indmis_n[0]))
		maxval=np.round(max(abs(mv1),abs(mv2)))
		print("Max val for histograms is: ", maxval)
		hbe_p=np.arange(self.bs/2,maxval+self.bs,self.bs)
		hbe_n=-1*hbe_p[::-1]
		hbe = np.hstack((hbe_n,hbe_p))
		# hbe -> histogram_bin_edges

		ntot="$N_{tot}$\n= %d" %(self.npairs)

		#fig = plt.figure(figsize=(5.5,10))
		#axh_p, axh_n = fig.subplots(2,1,sharex=True,sharey=True)
		fig = plt.figure()
		ncen="$N_{cen}$\n= %d" %(good_after_p.size)
		axh_p, axh_n = fig.subplots(1,2,sharex=True,sharey=True)
		axh_p.hist(kcao_flit_indmis_p[0],bins=hbe,edgecolor='black')
		axh_p.hist(kcao_flit_indmis_p[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_p.set_xlabel(r'$\Delta d$')
		axh_p.set_xlabel(r'$\ln \left( A^{obs}/A^{syn} \right)$', fontsize=16)
		#axh_p.set_ylabel("No. of pairs")#, fontsize=14)
		axh_p.set_title("Positive branch")#, fontsize=18)
		axh_p.tick_params(labelsize=14)
		axh_p.text(0.7, 0.8, ntot, transform=axh_p.transAxes)
		axh_p.text(0.7, 0.7, ncen, transform=axh_p.transAxes)

		ncen="$N_{cen}$\n= %d" %(good_after_n.size)
		axh_n.hist(kcao_flit_indmis_n[0],bins=hbe,edgecolor='black')
		axh_n.hist(kcao_flit_indmis_n[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_n.set_xlabel(r'$\Delta d$')#, fontsize=18)
		axh_n.set_xlabel(r'$\ln \left( A^{obs}/A^{syn} \right)$', fontsize=16)
		#axh_n.set_ylabel("No. of pairs")#, fontsize=14)
		axh_n.set_title("Negative branch")#, fontsize=18)
		axh_n.tick_params(labelsize=14)
		axh_n.text(0.1, 0.8, ntot, transform=axh_n.transAxes)
		axh_n.text(0.1, 0.7, ncen, transform=axh_n.transAxes)

	# --------------------------------------------------------------------------------------------------

	def hist_deltat(self):

		# compute waveform envelopes
		kcao_obsenv=np.abs(ss.hilbert(kcao_obscross, axis=0))
		kcao_synenv_init=np.abs(ss.hilbert(kcao_syncross_init, axis=0))
		kcao_synenv_final=np.abs(ss.hilbert(kcao_syncross_final, axis=0))

		# indices of waveform windows (each branch)
		kcao_negl = kcao_win_ind[0,:,:]
		kcao_negr = kcao_win_ind[1,:,:]
		kcao_posl = kcao_win_ind[2,:,:]
		kcao_posr = kcao_win_ind[3,:,:]

		# get the traveltime discrepancies
		dt = np.round(kcao_t[1] - kcao_t[0],1)
		deltat_pos_before=np.zeros(self.npairs)
		deltat_neg_before=np.zeros(self.npairs)
		deltat_pos_after=np.zeros(self.npairs)
		deltat_neg_after=np.zeros(self.npairs)

		cp=0 # cp stands for count_pair
		for j in range(nrecs-1):
			for i in range(j+1,nrecs):
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

		good_thresh = self.bs/2 # DO NOT CHANGE THIS. CHANGE ONLY THE BINSIZE

		cenbin_p=np.where(abs(deltat_pos_after)<good_thresh)[0]
		cenbin_n=np.where(abs(deltat_neg_after)<good_thresh)[0]
		cenbin_tot = cenbin_p.size + cenbin_n.size
		cenbin_frac = float(cenbin_tot)/(2*self.npairs)

		# determine appropriate histogram bins such that "0" is a bin center
		mv1=max(max(deltat_pos_before),max(deltat_neg_before))
		mv2=min(min(deltat_pos_before),min(deltat_neg_before))
		maxval=np.round(max(abs(mv1),abs(mv2)))
		print("Max val for histograms is: ", maxval)
		hbe_p=np.arange(self.bs/2,maxval+self.bs,self.bs)
		hbe_n=-1*hbe_p[::-1]
		hbe = np.hstack((hbe_n,hbe_p))
		# hbe -> histogram_bin_edges

		print("Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(cenbin_tot,100*cenbin_frac))
		ntot="$N_{tot}$\n= %d" %(self.npairs)
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

		nchid = kcao_allit_misfit/np.amax(kcao_allit_misfit)

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
				axsp=fig.add_subplot(3,3,p+1) #,aspect='equal')
			except ValueError:
				print("Problem plotting inversion result for iteration >= %d. Mismatch between number of iterations and number of subplots." %(p))
				return
			cax=axsp.pcolor(kcao_gx,kcao_gy,kcao_sditer[p,:,:],cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
			axsp.text(0.8,0.85,spname,transform=axsp.transAxes,color='white')
			#axsp.set_title(spname)
			#if p==pub-1:
			#	plt.colorbar(cax,ax=axsp,orientation="horizontal")

################################################################################

class waveform_plots:

	def __init__(self):

		# compute waveform envelopes
		self.kcao_obsenv=np.abs(ss.hilbert(kcao_obscross, axis=0))
		kcao_synenv_init=np.abs(ss.hilbert(kcao_syncross_init, axis=0))
		kcao_synenv_final=np.abs(ss.hilbert(kcao_syncross_final, axis=0))

		self.kcao_flit_synenv=(kcao_synenv_init, kcao_synenv_final)
		self.kcao_flit_syncross=(kcao_syncross_init, kcao_syncross_final)

		self.kcao_negl = kcao_win_ind[0,:,:]
		self.kcao_negr = kcao_win_ind[1,:,:]
		self.kcao_posl = kcao_win_ind[2,:,:]
		self.kcao_posr = kcao_win_ind[3,:,:]

	# --------------------------------------------------------------------------------------------------

	def plot_waveforms_oneit(self,a,b,z):

		"""
		a,b -> receiver indices
		z -> 0 for first iteration, 1 for last iteration
		"""

		fig=plt.figure(figsize=(7.5,2.5))
		ax=fig.add_subplot(111)
		# ax.spines['top'].set_visible(False)
		# ax.spines['left'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.yaxis.set_ticks([])
		ax.tick_params(axis='both', labelsize=14)
		print("Max value obscross: ", np.amax(kcao_obscross[:,a,b]))
		print("Max value syncross (first iteration): ", np.amax(self.kcao_flit_syncross[0][:,a,b]))
		# ax.plot(kcao_t,kcao_obscross[:,a,b],label='Observation')
		ax.plot(kcao_t,self.kcao_flit_syncross[z][:,a,b],label='Synthetic')
		try:
			ax.axvline(x=kcao_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
		except IndexError:
			# this happens when the window is the entire branch (usually with synthetic inversions)
			pass
		plt.legend()

	# --------------------------------------------------------------------------------------------------

	def plot_waveforms_flit(self,a,b):

		"""
		a,b -> receiver indices
		"""

		fig = plt.figure(figsize=(14,2))
		axf, axl = fig.subplots(1,2,sharey=True)
		axes=[axf,axl]
		ptitle={0: "Before", 1: "After"}
		for i,ax in enumerate(axes):
			z=0 if i==0 else -1
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.yaxis.set_ticks([])
			ax.plot(kcao_t,kcao_obscross[:,a,b],label='Observation')
			ax.plot(kcao_t,self.kcao_flit_syncross[z][:,a,b],label='Synthetic')
			try:
				ax.axvline(x=kcao_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
			except IndexError:
				# this happens when the window is the entire branch (usually with synthetic inversions)
				print("No lines for window..")
				pass
			ax.set_title(ptitle[i])
			if z!=0:
				plt.legend()

	# --------------------------------------------------------------------------------------------------

	def plot_envelopes_oneit(self,a,b,z):

		"""
		a,b -> receiver indices
		z -> 0 for first iteration, 1 for last iteration
		"""

		fig=plt.figure(figsize=(7,2))
		ax=fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.tick_params(axis='both', labelsize=14)
		ax.yaxis.set_ticks([])
		ax.plot(kcao_t,self.kcao_obsenv[:,a,b],label='Observation')
		ax.plot(kcao_t,kcao_flit_synenv[z][:,a,b],label='Synthetic')
		try:
			ax.axvline(x=kcao_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
		except IndexError:
			# this happens when the window is the entire branch (usually with synthetic inversions)
			pass
		if z!=0:
			plt.legend()

	# --------------------------------------------------------------------------------------------------

	def plot_envelopes_flit(self,a,b):

		"""
		a,b -> receiver indices
		"""

		fig = plt.figure(figsize=(14,2))
		axf, axl = fig.subplots(1,2,sharey=True)
		axes=[axf,axl]
		ptitle={0: "Before", 1: "After"}
		for i,ax in enumerate(axes):
			z=0 if i==0 else -1
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.yaxis.set_ticks([])
			ax.plot(kcao_t,self.kcao_obsenv[:,a,b],label='Observation')
			ax.plot(kcao_t,kcao_flit_synenv[z][:,a,b],label='Synthetic')
			try:
				ax.axvline(x=kcao_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
				ax.axvline(x=kcao_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
			except IndexError:
				# this happens when the window is the entire branch (usually with synthetic inversions)
				pass
			#ax.set_title(ptitle[i])
			if z!=0:
				plt.legend()

###############################################################################################################

class write_waveform_output:

	def __init__(self):
		self.rcoord_x = rc_xp * cg_dom_geom.dx
		self.rcoord_y = rc_yp * cg_dom_geom.dx
		print(self.rcoord_x)
		print(self.rcoord_y)
		self.pair_az = np.zeros((nrecs,nrecs))
		self.pair_baz = np.zeros((nrecs,nrecs))
		remove_sign = lambda m: m if m>0 else m+360
		for j in range(nrecs-1):
			for k in range(j+1,nrecs):
				x2_m_x1 = self.rcoord_x[k] - self.rcoord_x[j]
				y2_m_y1 = self.rcoord_y[k] - self.rcoord_y[j]
				az = np.arctan2(x2_m_x1,y2_m_y1)*180/np.pi
				baz = np.arctan2(-x2_m_x1,-y2_m_y1)*180/np.pi
				self.pair_az[j,k] = remove_sign(az)
				self.pair_baz[j,k] = remove_sign(baz)

		# print(kcao_drp)
		# print(self.pair_az)
		# print(self.pair_baz)

	def sac_output(self):

		# def create_trace(in_data, evname, stname, in_dist, br_x, br_y, ur_x, ur_y):
			# tr=oc.trace.Trace(data=in_data, header={ 'delta': kcao_dt, 'station': stname,\
			#  'sac': {'kevnm': evname, 'kstnm': stname, 'dist': in_dist, 'b': kcao_schar.tt[0], 'e': kcao_schar.tt[-1],\
			#   'az': 45, 'baz': 225, 'evlo': br_x, 'evla': br_y, 'stlo': ur_x, 'stla': ur_y}})
		def create_trace(in_data, evname, stname, in_dist, in_az, in_baz):
			tr=oc.trace.Trace(data=in_data, header={ 'delta': kcao_dt, 'station': stname,\
			 'sac': {'kevnm': evname, 'kstnm': stname, 'dist': in_dist, 'b': kcao_schar.tt[0], 'e': kcao_schar.tt[-1],\
			  'az': in_az, 'baz': in_baz}})
			return tr

		for br in range(kcao_obscross.shape[1]):
			urecs = list(range(kcao_obscross.shape[1]))[br+1:]
			trname = list(map(lambda x: '%d-%d' %(br,x), urecs))
			# common_args = lambda l,m,n: (trname[l][0], trname[l][2], kcao_drp[m,n], self.rcoord_x[m], self.rcoord_y[m], self.rcoord_x[n], self.rcoord_y[n])
			common_args = lambda l,m,n: (trname[l].split('-')[0], trname[l].split('-')[1], kcao_drp[m,n], self.pair_az[m,n], self.pair_baz[m,n])
			if len(urecs)>0:
				obs_cc=kcao_obscross
				syn_cc=kcao_syncross_init
				tr_obs = [create_trace(obs_cc[:,br,ur], *common_args(e,br,ur)) for e,ur in enumerate(urecs)]
				tr_syn = [create_trace(syn_cc[:,br,ur], *common_args(e,br,ur)) for e,ur in enumerate(urecs)]
				final_obs = oc.stream.Stream(traces=tr_obs)
				final_syn = oc.stream.Stream(traces=tr_syn)
				for tn,tr in enumerate(final_obs.traces):
					sacfname='cc_obs_' + trname[tn] +'.sac'
					tr.write(sacfname, format='SAC')
				for tn,tr in enumerate(final_syn.traces):
					sacfname='cc_syn_' + trname[tn] +'.sac'
					tr.write(sacfname, format='SAC')

###############################################################################################################

def plot_Lcurve():
	print("Model norm: ", sorted_modnorm)
	print("Misfit: ", sorted_misfit)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(sorted_modnorm,sorted_misfit)
	for p in range(len(filelist)):
		gval = np.sort(gamma_files)[p]
		if gval>=1 or gval==0:
			plabel=r"$\gamma = %d$" %(gval)
		else:
			dp=abs(int(np.floor(np.log10(gval))))
			plabel=r"$\gamma = %.*f$" %(dp,gval)
		ax.plot(sorted_modnorm[p],sorted_misfit[p],'o',label=plabel)
	ax.legend()
	ax.set_xlabel("Model norm (relative)")
	ax.set_ylabel("Misfit")

###############################################################################################################

def info_code_run():

	print("\nCROSSS-CORRELATION MODELLING THEORY\n")
	print("Source direction: ", cg_ccmt.src_dir)
	print("Green tensor component: ", cg_ccmt.GTC)

	print("\nMODELLING DOMAIN GEOMETRY\n")
	print("dx (km): ", cg_dom_geom.dx)
	print("zmax (km): ", cg_dom_geom.zmax)
	print("size of box (km): ", cg_dom_geom.box_len)

	print("\nMODELLING PARAMEERIZATION\n")
	print("Radius: ", cg_somod_rg_specs['r'])
	print("Width: ", cg_somod_rg_specs['w'])
	print("Angular sampling: ", cg_somod_rg_specs['as'])

	print("\nSIGNAL PARAMETERS\n")
	print("dt (s): ", kcao_schar.dt)
	print("nsam: ", kcao_schar.nsam)
	print("cf (Hz): ", kcao_schar.cf)
	print("lf (Hz): ", kcao_schar.lf)
	print("hf (Hz): ", kcao_schar.hf)
	print("alpha parameter (Tukey window): ", kcao_schar.altukey)
	print("fsigma (Hz): ", kcao_schar.fsigma)
	print("Power-spectrum type: ", kcao_schar.pst)

	print("\nSCALAR MODELLING PARAMETERS\n")
	print("Density (gm/cc): ", cg_scal_mod.rho_scal2D)
	print("Wavespeed (km/s): ", cg_scal_mod.wavspeed_scal2D)

	print("\nINVERSION RELATED CHOICES\n")
	print("Gamma for damping: ", kcao_gamma)

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
	print("          6 - Exit")

##################################################################################################################

################################################################################
# MAIN PROGRAM
################################################################################

# if __name__ == '__main__':

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

	jar=gzip.open(pfile)
	print("Reading ", pfile)

	# load CODE-INPUT quantities stored
	reald=pickle.load(jar)
	cg_ccmt=pickle.load(jar)
	cg_invc=pickle.load(jar)
	cg_scal_mod=pickle.load(jar)
	cg_dom_geom=pickle.load(jar)
	cg_sig_char=pickle.load(jar)
	cg_somod_mg_specs=pickle.load(jar)
	cg_somod_rg_specs=pickle.load(jar)
	kcao_schar=pickle.load(jar)
	kcao_pss=pickle.load(jar)

	# load CODE-OUTPUT quantities stored
	kcao_win_ind=pickle.load(jar)
	kcao_drp=pickle.load(jar)
	rchosen=pickle.load(jar)
	rc_xp=pickle.load(jar)
	rc_yp=pickle.load(jar)
	kcao_allit_mc=pickle.load(jar)
	kcao_allit_misfit=pickle.load(jar)
	kcao_mfkp=pickle.load(jar)
	kcao_mfkn=pickle.load(jar)
	kcao_flit_indmis_p=pickle.load(jar)
	kcao_flit_indmis_n=pickle.load(jar)
	kcao_obscross=pickle.load(jar)
	kcao_syncross_init=pickle.load(jar)
	kcao_syncross_final=pickle.load(jar)
	if not reald:
		kcao_sdtrue = pickle.load(jar)
		# trumod = pickle.load(jar)
		# if len(trumod.shape)==1:
		# # implies basis coefficients are stored
		# 	mc_true=trumod
		# 	kcao_sdtrue = np.zeros((num_points,num_points))
		# elif len(trumod.shape)==2:
		# implies the complete model itself is stored (not specified in terms of basis)
		# 	kcao_sdtrue = trumod
	jar.close()

	#************************************** Extract quantities required in this code **********************************************

	num_points = cg_dom_geom.gx.shape[0]

	kcao_dx = cg_dom_geom.dx
	kcao_gx = cg_dom_geom.gx
	kcao_gy = cg_dom_geom.gy
	kcao_gamma = cg_invc.gamma_inv

	kcao_t = kcao_schar.tt
	kcao_dt = kcao_schar.dt

	nrecs = len(rc_xp)
	numit = len(kcao_allit_mc)

	if len(filelist)>1:
		nrecs_files[p]=len(rc_xp)
		gamma_files[p]=kcao_gamma
		misfit_files[p]=kcao_allit_misfit[-1]
		# mod_norm = np.sum(np.square(kcao_allit_mc[-1]))
		mod_norm = np.sum(np.square(kcao_allit_mc[-1]))/np.sum(np.square(kcao_allit_mc[0]))
		modnorm_files[p]=mod_norm
	else:
		#  Print info about code run
		info_code_run()

	#************************************** Perform necessary checks **********************************************

	try:
		assert kcao_obscross.shape[1]==kcao_obscross.shape[2]
		assert kcao_syncross_init.shape[1]==kcao_syncross_init.shape[2]
		assert kcao_syncross_final.shape[1]==kcao_syncross_final.shape[2]
	except AssertionError:
		raise SystemExit("Problem with stored waveform matrices")

	assert nrecs == kcao_obscross.shape[1]
	assert len(kcao_allit_misfit) == len(kcao_allit_mc)

#********************************* End of FOR loop over pickle file(s) **************************************************

# Custom modules
import anseicca_utils1 as u1

if len(filelist)>1:
	# usually used to compare inversions with different settings (e.g. L-curve plotting)

	if len(np.unique(nrecs_files))>1:
		sys.exit("The different pickles have different number of receivers - script terminated.")

	sortind = np.argsort(gamma_files)
	sorted_misfit = misfit_files[sortind]
	sorted_modnorm = modnorm_files[sortind]
	print("Sorted gamma values: ", np.sort(gamma_files))
	plot_Lcurve()

else:
	# misfit_start = 0.5*np.sum(np.square(kcao_flit_indmis_p[0]) + np.square(kcao_flit_indmis_n[0]))
	# misfit_end = 0.5*np.sum(np.square(kcao_flit_indmis_p[-1]) + np.square(kcao_flit_indmis_n[-1]))
	# if np.round(misfit_start,6) != np.round(kcao_allit_misfit[0],6) or np.round(misfit_end,6) != np.round(kcao_allit_misfit[-1],6):
	# 	print("Initial misfit with and without errors: ", kcao_allit_misfit[0], misfit_start)
	# 	print("Final misfit with and without errors: ", kcao_allit_misfit[-1], misfit_end)

	mpo = model_info()
	desired_output = 0
	while desired_output != 6:
		usage()
		desired_output = int(input("Enter your choice (number) here: "))
		if desired_output>=6:
			sys.exit('Thank you')
		elif desired_output==1:
			mpo.plot_models()
		elif desired_output==2:
			plot_kernels()
		elif desired_output==3:
			ippo = inversion_progress_plots()
			ippo.mod_iter(mpo.kcao_sditer, mpo.mod_min, mpo.mod_max)
			ippo.chi_iter()
			ippo.hist_deltad()
			ippo.hist_deltat()
		elif desired_output==4:
			wpo = waveform_plots()
			usrc_ww=int(input("See actual waveforms (1) or waveform envelopes (2) ? "))
			usr_int=True
			while usr_int:
				usrc_nn=input("Enter receiver/station numbers (any other key for back to main): ")
				if len(usrc_nn.split())==2:
					m=int(usrc_nn.split()[0])
					n=int(usrc_nn.split()[1])
					if usrc_ww==1:
						wpo.plot_waveforms_flit(m,n)
						wpo.plot_waveforms_oneit(m,n,0)
					elif usrc_ww==2:
						wpo.plot_envelopes_flit(m,n)
						wpo.plot_envelopes_oneit(m,n,0)
					plt.show()
				else:
					usr_int=False
		elif desired_output==5:
			try:
				import obspy.core as oc
			except ModuleNotFoundError:
				raise SystemExit("Requires 'obspy' module. Please run in an environment containing the obspy module.")
			wwoo = write_waveform_output()
			wwoo.sac_output()
		plt.show()
