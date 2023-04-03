#!/usr/bin/python

# General purpose modules
import os
import sys
import gzip
import pickle
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

##########################################################################################

class read_anseicca_pickle:

    def __init__(self, pfile):

        jar=gzip.open(pfile)

        # load 'INPUT/USER-DETERMINED' quantities stored
        self.extdata=pickle.load(jar)
        self.cg_ccmt=pickle.load(jar)
        self.cg_invc=pickle.load(jar)
        self.cg_scal_mod=pickle.load(jar)
        self.cg_dom_geom=pickle.load(jar)
        self.cg_sig_char=pickle.load(jar)
        self.cg_somod_mg_specs=pickle.load(jar)
        self.cg_somod_rg_specs=pickle.load(jar)
        self.cg_somod_rgr_specs=pickle.load(jar)
        self.cg_somod_gg_specs=pickle.load(jar)
        self.cg_inv_mod_type=pickle.load(jar)
        self.cg_inv_mdlng_type=pickle.load(jar)
        if not self.extdata:
            self.cg_tru_mod_type=pickle.load(jar)
            self.cg_tru_mdlng_type=pickle.load(jar)
            self.cg_syn_data=pickle.load(jar)
        self.kcao_used_pairs=pickle.load(jar)
        self.kcao_sig_char=pickle.load(jar)
        self.kcao_pss=pickle.load(jar)
        self.kcao_vel_acou=pickle.load(jar)

        # load 'OUTPUT/CODE-COMPUTED' quantities stored
        self.kcao_win_ind=pickle.load(jar)
        self.kcao_snr=pickle.load(jar)
        self.kcao_drp=pickle.load(jar)
        self.chosen_st_info=pickle.load(jar)
        self.rc_xp=pickle.load(jar)
        self.rc_yp=pickle.load(jar)
        self.kcao_allit_mc=pickle.load(jar)
        self.kcao_allit_misfit=pickle.load(jar)
        self.kcao_mfkp=pickle.load(jar)
        self.kcao_mfkn=pickle.load(jar)
        self.kcao_flit_indmis_p=pickle.load(jar)
        self.kcao_flit_indmis_n=pickle.load(jar)
        self.kcao_obscross=pickle.load(jar)
        self.kcao_syncross_init=pickle.load(jar)
        self.kcao_syncross_final=pickle.load(jar)
        self.kcao_sdinv = pickle.load(jar)
        if not self.extdata:
        	self.kcao_sdtrue = pickle.load(jar)
        jar.close()

##########################################################################################

class read_strucinv_pickle:

    def __init__(self, pfile):

        jar=gzip.open(pfile)
        self.extdata=pickle.load(jar)
        self.cg_ccmt=pickle.load(jar)
        self.cg_dom_geom=pickle.load(jar)
        self.cg_scal_mod=pickle.load(jar)
        self.cg_invc=pickle.load(jar)
        self.kcao_sig_char=pickle.load(jar)
        # self.kcao_pss=pickle.load(jar)

        # load 'CODE-OUTPUT' quantities stored
        # self.kcao_win_ind=pickle.load(jar)
        self.kcao_drp=pickle.load(jar)
        self.chosen_st_info=pickle.load(jar)
        self.rc_xp=pickle.load(jar)
        self.rc_yp=pickle.load(jar)
        self.kcao_syncross_init=pickle.load(jar)

        jar.close()

##########################################################################################

class waveform_plots:

    def __init__(self, ipo):

        self.ipo = ipo

        self.ipo_t = ipo.kcao_sig_char.tt

        try:
            ipo.kcao_obscross
        except AttributeError:
            self.obs_present=False
        else:
            self.obs_present=True

        # compute waveform envelopes
        kcao_synenv_init=np.abs(ss.hilbert(self.ipo.kcao_syncross_init, axis=0))

        try:
            self.kcao_obsenv=np.abs(ss.hilbert(self.ipo.kcao_obscross, axis=0))
            kcao_synenv_final=np.abs(ss.hilbert(self.ipo.kcao_syncross_final, axis=0))

            self.kcao_flit_synenv=(kcao_synenv_init, kcao_synenv_final)
            self.kcao_flit_syncross=(self.ipo.kcao_syncross_init, self.ipo.kcao_syncross_final)

            self.kcao_negl = self.ipo.kcao_win_ind[0,:,:]
            self.kcao_negr = self.ipo.kcao_win_ind[1,:,:]
            self.kcao_posl = self.ipo.kcao_win_ind[2,:,:]
            self.kcao_posr = self.ipo.kcao_win_ind[3,:,:]
        except AttributeError:
            self.kcao_flit_syncross=(self.ipo.kcao_syncross_init, self.ipo.kcao_syncross_init)

    # --------------------------------------------------------------------------------------------------

    def plot_waveforms_oneit(self,a,b,z):

        """
        a,b -> receiver indices
        z -> 0 for first iteration, 1 for last iteration
        """

        fig=plt.figure(figsize=(7.5,2.5))
        ax=fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.tick_params(axis='both', labelsize=14)
        # ax.set_xlabel('[s]', size=14)
        print("Max value syncross (first iteration): ", np.amax(self.kcao_flit_syncross[0][:,a,b]))
        if self.obs_present:
        	print("Max value obscross: ", np.amax(self.ipo.kcao_obscross[:,a,b]))
        ax.plot(self.ipo_t,self.ipo.kcao_obscross[:,a,b],label='Observed')
        ax.plot(self.ipo_t,self.kcao_flit_syncross[z][:,a,b],label='Predicted')
        try:
            ax.text(0.7, 0.8, self.ipo.kcao_snr[a,b], transform=ax.transAxes)
            ax.text(0.1, 0.8, self.ipo.kcao_snr[b,a], transform=ax.transAxes)
        except (TypeError, AttributeError):
            # happens when kcao_snr does not exist or is =None, i.e. for synthetic data
            pass
        try:
        	ax.axvline(x=self.ipo_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
        	ax.axvline(x=self.ipo_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
        	ax.axvline(x=self.ipo_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
        	ax.axvline(x=self.ipo_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
        except (AttributeError, IndexError):
            # AttributeError happens when no window is defined
        	# IndexError happens when the window is the entire branch (usually with synthetic inversions)
        	pass
        plt.legend(loc=2)

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
    		# ax.spines['top'].set_visible(False)
    		# ax.spines['left'].set_visible(False)
    		# ax.spines['right'].set_visible(False)
    		# ax.yaxis.set_ticks([])
    		ax.plot(self.ipo_t,self.ipo.kcao_obscross[:,a,b],label='Observed')
    		ax.plot(self.ipo_t,self.kcao_flit_syncross[z][:,a,b],label='Predicted')
    		try:
    			ax.axvline(x=self.ipo_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
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
    	ax.plot(self.ipo_t,self.kcao_obsenv[:,a,b],label='Observed')
    	ax.plot(self.ipo_t,self.kcao_flit_synenv[z][:,a,b],label='Predicted')
    	try:
    		ax.axvline(x=self.ipo_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
    		ax.axvline(x=self.ipo_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
    		ax.axvline(x=self.ipo_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
    		ax.axvline(x=self.ipo_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
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
    		ax.plot(self.ipo_t,self.kcao_obsenv[:,a,b],label='Observed')
    		ax.plot(self.ipo_t,self.kcao_flit_synenv[z][:,a,b],label='Predicted')
    		try:
    			ax.axvline(x=self.ipo_t[self.kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
    			ax.axvline(x=self.ipo_t[self.kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
    		except IndexError:
    			# this happens when the window is the entire branch (usually with synthetic inversions)
    			pass
    		#ax.set_title(ptitle[i])
    		if z!=0:
    			plt.legend()

##########################################################################################

class write_waveform_output:

    def __init__(self, ipo, nrecs):

        self.ipo = ipo
        self.nrecs = nrecs

        self.rcoord_x = self.ipo.rc_xp * self.ipo.cg_dom_geom.dx
        self.rcoord_y = self.ipo.rc_yp * self.ipo.cg_dom_geom.dx
        # print(self.rcoord_x)
        # print(self.rcoord_y)
        self.pair_az = np.zeros((self.nrecs,self.nrecs))
        self.pair_baz = np.zeros((self.nrecs,self.nrecs))
        remove_sign = lambda m: m if m>0 else m+360
        for j in range(self.nrecs-1):
        	for k in range(j+1,self.nrecs):
        		x2_m_x1 = self.rcoord_x[k] - self.rcoord_x[j]
        		y2_m_y1 = self.rcoord_y[k] - self.rcoord_y[j]
        		az = np.arctan2(x2_m_x1,y2_m_y1)*180/np.pi
        		baz = np.arctan2(-x2_m_x1,-y2_m_y1)*180/np.pi
        		self.pair_az[j,k] = remove_sign(az)
        		self.pair_baz[j,k] = remove_sign(baz)

        # print(self.ipo.kcao_drp)
        # print(self.pair_az)
        # print(self.pair_baz)

    # --------------------------------------------------------------------------------------------------

    def sac_output(self, oc, rchosen_id, kcao_sigchar):

        def create_trace(in_data, evname, stname, in_dist, in_az, in_baz):
            tr=oc.trace.Trace(data=in_data, header={ 'delta': kcao_dt, 'station': stname,\
             'sac': {'kevnm': evname, 'kstnm': stname, 'dist': in_dist, 'b': self.ipo.kcao_sig_char.tt[0], 'e': self.ipo.kcao_sig_char.tt[-1],\
              'az': in_az, 'baz': in_baz}})
            return tr

        kcao_dt = kcao_sigchar.dt

        # for br in range(self.ipo.kcao_obscross.shape[1]):
        for br in range(self.nrecs):
            # urecs = list(range(self.ipo.kcao_obscross.shape[1]))[br+1:]
            urecs = list(range(self.nrecs))[br+1:]
            trnme_stno_code = list(map(lambda x: '%d-%d' %(br,x), urecs))
            trhdr_stid_orig = list(map(lambda x: '%s-%s' %(rchosen_id[br],rchosen_id[x]), urecs))
            # common_args = lambda l,m,n: (trnme_stno_code[l][0], trnme_stno_code[l][2], self.ipo.kcao_drp[m,n], self.rcoord_x[m], self.rcoord_y[m], self.rcoord_x[n], self.rcoord_y[n])
            # common_args = lambda l,m,n: (trnme_stno_code[l].split('-')[0], trnme_stno_code[l].split('-')[1], self.ipo.kcao_drp[m,n], self.pair_az[m,n], self.pair_baz[m,n])
            common_args = lambda l,m,n: (trhdr_stid_orig[l].split('-')[0], trhdr_stid_orig[l].split('-')[1], self.ipo.kcao_drp[m,n], self.pair_az[m,n], self.pair_baz[m,n])
            if len(urecs)>0:

                syn_cc=self.ipo.kcao_syncross_init
                tr_syn = [create_trace(syn_cc[:,br,ur], *common_args(e,br,ur)) for e,ur in enumerate(urecs)]
                final_syn = oc.stream.Stream(traces=tr_syn)
                for tn,tr in enumerate(final_syn.traces):
                    sacfname='cc_syn_' + trnme_stno_code[tn] +'.sac'
                    tr.write(sacfname, format='SAC')

                try:
                    obs_cc=self.ipo.kcao_obscross
                    obs_present=True
                except AttributeError:
                    pass
                else:
                    tr_obs = [create_trace(obs_cc[:,br,ur], *common_args(e,br,ur)) for e,ur in enumerate(urecs)]
                    final_obs = oc.stream.Stream(traces=tr_obs)
                    for tn,tr in enumerate(final_obs.traces):
                        sacfname='cc_obs_' + trnme_stno_code[tn] +'.sac'
                        tr.write(sacfname, format='SAC')

###############################################################################################################
