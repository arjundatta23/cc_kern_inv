import sys
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt


import config_file as config

#------------------------------------------ global variables -------------------------------------------------
nrecs_select = config.nrecs
reald = config.ext_data
dom_geom = config.dom_geom
try:
    # overwrite some of these data types if relevant
    dom_geom = sys.modules['__main__'].dom_geom_pckl
except AttributeError:
    pass

dxy = dom_geom.dx
ngpmb = dom_geom.ngp_box
gamma = config.invc.gamma_inv

ngp = {2: ngpmb, 3: 2*ngpmb-1}
xall = {2: dom_geom.X, 3: dom_geom.X2}
yall = {2: dom_geom.Y, 3: dom_geom.Y2}

###############################################################################################################
def compute_cc_1pair_ptsrc(uA, uB, sig_par, src_distrib, powspec=None):

    """ Algorithm using point sources (invokes reciprocity at both receivers) """

    # get necessary signal parameters
    fhz = sig_par.fhz
    nom = sig_par.nsam
    dom = sig_par.domega
    nom_nn = sig_par.n_nn_fsam

    # consider possible asymmetry in frequency samples (happens when the number of samples is even)
    fhzp = len(fhz[fhz>0])
    fhzn = len(fhz[fhz<0])
    ssna = abs(fhzn-fhzp)
    # ssna stands for samples_to_skip_due_to_nyquist_asymmetry
    print("SSNA: ", ssna)

    ans = np.zeros(nom, dtype='complex')

    # compute the cross-correlations for positive frequencies
    fsyn = uA[1:nom_nn,:,:] * np.conj(uB[1:nom_nn,:,:]) * src_distrib
    spa_int = np.sum(fsyn, axis=(1,2)) * (dxy**2)
    ans[1:nom_nn] = spa_int if powspec is None else spa_int * powspec[1:nom_nn]

    # Frequency-domain symmetry: negative frequency coefficients are complex conjugates of flipped positive coefficients
    ans[nom_nn+ssna:] = np.flipud(np.conj(ans[1:nom_nn]))
    # 22 June 2018: BEWARE, the negative Nyquist term gets left out in case ssna>0, i.e. in case self.nom is even.
    # this does matter of course, but it appears to make a very minor difference to the event kernels
    # so I am leaving it for the time being.

    ft_fac = nom*(dom/(2*np.pi))
    ans *= ft_fac

    return ans

###############################################################################################################

class setup_modelling_domain:

	def __init__(self, st_no, st_id, xrec, yrec, ox, oy, make_plots=True):

		self.dxy=dxy

		assert xrec.size==yrec.size
		nrecs_total = xrec.size

		#************** redefine receiver locations for chosen grid and coordinate origin ************
		recx_ro = xrec - ox
		recy_ro = yrec - oy
		# ro -> relative_to_origin
		recx_gp = recx_ro/dxy
		recy_gp = recy_ro/dxy
		# receiver locations in integer grid points away from coordinate origin
		recx_igp = np.asarray(np.rint(recx_gp), dtype=int)
		recy_igp = np.asarray(np.rint(recy_gp), dtype=int)

		if nrecs_select < nrecs_total:
			# sta_subset_file = input("File containing receiver subset (enter 0 for automatic selection): ")
			sta_subset_file = "0"
			if sta_subset_file.isdigit():
				#************** Automated selection criterion for selecting a subset of entire array *************
				ro_act = np.sqrt(recx_ro**2 + recy_ro**2)
				ro_grid = np.sqrt((recx_igp*dxy)**2 + (recy_igp*dxy)**2)
				grd_err = np.abs(ro_act-ro_grid)
				grd_err_chosen = np.sort(grd_err)[:nrecs_select]
				ichosen = np.argwhere(np.in1d(grd_err,grd_err_chosen))
			else:
				ssf=open(sta_subset_file,'r')
				entire=ssf.readlines()
				ssf.close()
				del ssf
				sel_st_id=list(map(lambda p: p.split()[0], entire))
				try:
					ichosen=np.array([st_id.index(j) for j in sel_st_id])
				except ValueError:
					print(sel_st_id)
					raise SystemExit("Problem with station selection file - listed stations not present in master list.")

		elif nrecs_select == nrecs_total:
			ichosen = np.arange(nrecs_total)

		self.act_recx_rel=recx_ro[ichosen]
		self.act_recy_rel=recy_ro[ichosen]

		try:
			assert len(ichosen)==nrecs_select
		except AssertionError:
			print(len(ichosen))
			raise SystemExit("Problem with number of receivers selected.")

		self.rchosenx_igp = recx_igp[ichosen.flatten()]
		self.rchoseny_igp = recy_igp[ichosen.flatten()]
		chosen_st_id = [st_id[s] for s in ichosen.flatten()]
		chosen_st_no = [st_no[s] for s in ichosen.flatten()]
		rnchosen = [j+1 for j in ichosen.flatten()]
		try:
			assert rnchosen==chosen_st_no
		except AssertionError:
			print(rnchosen)
			print(chosen_st_no)
			raise SystemExit("Problem with station/receiver numbers")

		print("\nChosen %d out of %d receivers: " %(len(ichosen),nrecs_total))
		print(chosen_st_no)
		print(chosen_st_id)

		self.chosen_st_no = chosen_st_no
		self.chosen_st_id = chosen_st_id

		#***************************** compute pairwise distances ************************************
		self.act_dist_rp=np.zeros((nrecs_select,nrecs_select))
		# self.act_dist_rp -> actual_distance_receiver_pairs. The word "actual" is used in the name so as to distinguish
		# these distances from the "effective" ones which are computed, in the h13 module, using the approximate
		# reciever locations on the uniform grid of h13.
		act_rnum_rp=np.zeros((nrecs_select,nrecs_select), dtype=object)
		# act_rnum_rp -> similar to self.act_dist_rp but for receiver-pair numbers/names, rather than distances

		for b,brec in enumerate(rnchosen[:-1]):
			urecs=rnchosen[b+1:]
			x1=xrec[np.searchsorted(st_no,brec)]
			y1=yrec[np.searchsorted(st_no,brec)]
			x2=xrec[np.searchsorted(st_no,urecs)]
			y2=yrec[np.searchsorted(st_no,urecs)]
			self.act_dist_rp[b+1:,b]=np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
			self.act_dist_rp[b,b+1:]=self.act_dist_rp[b+1:,b]
			act_rnum_rp[b:,b]=list(map(lambda x: '%d-%d' %(x,brec), rnchosen[b:]))
			act_rnum_rp[b,b:]=list(map(lambda x: '%d-%d' %(brec,x), rnchosen[b:]))

		self.dist_1Darr = self.act_dist_rp[np.nonzero(np.tril(self.act_dist_rp))]
		try:
			assert self.dist_1Darr.size == int(nrecs_select*(nrecs_select-1)/2)
		except AssertionError:
			raise SystemExit("Problem with matrix of receiver pair distances: found at least one zero-value for interstation distance")

		# print(self.act_dist_rp)
		# print("Receiver-pair distances: ")
		# print(self.dist_1Darr)

		#********************************** make plots if desired ************************************
		if make_plots:
			self.plot_absolute_actual(xrec,yrec,ichosen)
			self.plot_relative_gridded()
			self.plot_raypaths(xrec-ox,yrec-oy,ichosen)

		#*********************************************************************************************

	def plot_absolute_actual(self,rx,ry,ind_in):
		ind_all = np.arange(rx.size)
		ind_out = np.setdiff1d(ind_all,ind_in)
		xry_act=plt.figure()
		axmap=xry_act.add_subplot(111,aspect='equal')
		# selected stations
		axmap.scatter(rx[ind_in],ry[ind_in],marker='^',s=100,facecolor='r',edgecolor='k')
		# excluded stations
		axmap.scatter(rx[ind_out],ry[ind_out],marker='^',s=100,facecolor='r',alpha=0.1,edgecolor='k')
		# interstation paths for all RELEVANT station pairs
		for j in ind_in:
			for k in ind_in:
				if k>j:
					axmap.plot([rx[j],rx[k]],[ry[j],ry[k]],color='grey')
		axmap.set_xlabel("Easting [km]")
		axmap.set_ylabel("Northing [km]")
		axmap.set_title("Absolute coordinates")

	def plot_relative_gridded(self):
		xpos_grid=self.rchosenx_igp*self.dxy
		ypos_grid=self.rchoseny_igp*self.dxy
		fig=plt.figure()
		axgrid=fig.add_subplot(111, aspect='equal')
		axgrid.scatter(self.act_recx_rel,self.act_recy_rel,marker='^',facecolor='b',label="Actual")
		axgrid.scatter(xpos_grid,ypos_grid,marker='o',facecolor='r',label="On uniform grid")
		axgrid.set_xlabel('Km')
		axgrid.set_ylabel('Km')
		axgrid.set_title("Relative coordinates")
		axgrid.legend()

	def plot_raypaths(self,all_rx_rel,all_ry_rel,ind_sel):
		ind_all = np.arange(all_rx_rel.size)
		ind_out = np.setdiff1d(ind_all,ind_sel)

		nrecs_sel = self.rchosenx_igp.size

		xpos_grid=self.rchosenx_igp*self.dxy
		ypos_grid=self.rchoseny_igp*self.dxy
		fig=plt.figure()
		axrp=fig.add_subplot(111, aspect='equal')
		axrp.scatter(xpos_grid,ypos_grid,marker='^',s=100,facecolor='r',edgecolor='k')
		axrp.scatter(all_rx_rel[ind_out],all_ry_rel[ind_out],marker='^',s=100,facecolor='r',alpha=0.1,edgecolor='k')

		pair_counter=0
		for j in range(nrecs_sel-1):
			for k in range(j+1,nrecs_sel):
				# print("Selected pair: ",j,k)
				x1=xpos_grid[j]
				x2=xpos_grid[k]
				y1=ypos_grid[j]
				y2=ypos_grid[k]
				axrp.plot([x1,x2],[y1,y2],color='grey')
				pair_counter+=1
		assert pair_counter == nrecs_sel*(nrecs_sel-1)/2

		axrp.set_xlabel('X [km]', fontsize=14)
		axrp.set_ylabel('Y [km]', fontsize=14)
		axrp.tick_params(axis='both', labelsize=14)
		# axrp.set_title("Relative coordinates")

########################################################################################################################################

class use_modelling_domain:

	def __init__(self, rlocsx, rlocsy):

		nrecs = rlocsx.size

		self.dist_rp=np.zeros((nrecs,nrecs))
		for j in range(nrecs):
		    for i in range(nrecs):
		        self.dist_rp[i,j] = np.sqrt( (rlocsx[i]-rlocsx[j])**2 + (rlocsy[i]-rlocsy[j])**2 )

		# check matrix of distances computed after gridding
		izd=np.argwhere(self.dist_rp==0)
		try:
		    assert np.all(izd[:,0]==izd[:,1])
		    # only diagonal elements of the distance matrix should be 0
		except AssertionError:
		    print(izd[izd[:,0] != izd[:,1]])
		    raise SystemExit("One or more pair of stations (code ids printed above), has the same location on grid -\
		     please rectify by either changing the grid, or eliminating one of the stations in each pair.")

		dist_all = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]
		self.alldist_1D = dist_all[np.argsort(dist_all)]
		# a sorted 1-D array of receiver-pair distances

		print("Receiver pair indices sorted by distance: ")
		# for sd in self.alldist_1D:
		# 	rpi=np.argwhere(self.dist_rp==sd)
		# 	rpi_lt = rpi[rpi[:,0]>rpi[:,1]][0]
		# 	print(rpi_lt, "%s-%s" %(chosen_st_id[rpi_lt[0]], chosen_st_id[rpi_lt[1]]))

########################################################################################################################################

class add_noise_all_cc:

    def __init__(self, cc_matrix_clean, sig_att, noise_pcent, noise_fband):

        def bandlimited_noise(sig_shape):

            freq_samp=1.0/deltat
            # np.random.seed(seed=42)
            noise_full = noise_amp * np.random.randn(*sig_shape)
            Wn = noise_fband
            b, a = ss.butter(4, Wn, btype='bandpass',  fs=freq_samp)
            noise_banded = ss.filtfilt(b, a, noise_full, axis=0)
            return noise_banded

        deltat = sig_att.dt

        rms_mat = np.sqrt(np.mean(cc_matrix_clean**2, axis=0))
        assert np.all(np.tril(rms_mat) == rms_mat)
        rms_nz = rms_mat[np.nonzero(np.tril(rms_mat))]
        avg_rms_all_cc = np.mean(rms_nz)

        noise_amp = (noise_pcent * avg_rms_all_cc)/100

        self.noisy_signal = cc_matrix_clean + bandlimited_noise(cc_matrix_clean.shape)

###############################################################################################################

class somod:

    # ringg -> ring_of_gaussians (used by Datta_et_al, 2019)
    # rgring -> radially_gaussian_ring (used by Hanasoge, 2013)

    """ All gaussians are constructed such that 'half-width = 2-sigma' """

    @staticmethod
    def mult_gauss(rcen, specs, fac_rel=2):

        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        xp=rcen[0]
        yp=rcen[1]
        ind = specs['r0'].index((xp,yp))
        ampl = specs['mag'][ind]
        sigma_x = specs['w'][ind][0]/4
        sigma_y = specs['w'][ind][1]/4

        for j in range(ngp[fac_rel]):
            ans[:,j] = np.exp( -( (xall[fac_rel][j] - xp)**2/(sigma_x**2) + (yall[fac_rel] - yp)**2/(sigma_y**2) ) )

        return ans

    #******************************************************************************************

    @staticmethod
    def ringg(theta, specs, fac_rel=2):
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        rad=specs['r']
        sigma=specs['w']/4
        rad_use = {2: rad, 3: 2*rad}

        for j in range(ngp[fac_rel]):
            x0 = rad_use[fac_rel]*np.cos(theta)
            y0 = rad_use[fac_rel]*np.sin(theta)
            ans[:,j] = np.exp( -((xall[fac_rel][j] - x0)**2 + (yall[fac_rel] - y0)**2)/(sigma**2) )

        return ans

    #******************************************************************************************

    @staticmethod
    def rgring(rcen, specs, fac_rel=2):
        #         if mag2 is None:
        #         	ampl = mag1
        #         else:
        #         	#if abs(xall[j])<10:
        #         	if xall[j]>-22 and xall[j]<-15:
        #         		ampl = mag2
        #         	else:
        #         		ampl = mag1
        mag2=None
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        xp = rcen[0]
        yp = rcen[1]
        ind = specs['r0'].index((xp,yp))
        rad = specs['r'][ind]
        ampl = specs['mag'][ind]
        sigma=specs['w'][ind]/4

        for j in range(ngp[fac_rel]):
            r_ib = np.sqrt((xall[fac_rel][j]-xp)**2 + (yall[fac_rel]-yp)**2)
            ans[:,j] = ampl * ( np.exp( -(r_ib-rad)**2/(sigma**2)) )

        return ans

    #******************************************************************************************

    @staticmethod
    def gcover(coord, specs, fac_rel=2):
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        sigma = specs['w']/4
        x0=coord[0] 
        y0= coord[1]
        for j in range(ngp[fac_rel]):
            ans[:,j] = np.exp(-((xall[fac_rel][j] - x0)**2 + (yall[fac_rel] - y0)**2)/(sigma**2) )
        return ans

    #******************************************************************************************

###############################################################################################################

class inversion:

	def __init__(self, nm):

	    #---------------------------- fix the damping (model covarance matrix) ------------------------------
	    if reald:
		# in case of real data, we use a banded model covariance matrix (non-zero covariances)
	    	self.Cm = np.zeros((nm,nm))
	    	cord = 3
	    	# cord -> correlation_distance
	    	for r in range(self.Cm.shape[0]):
	    		col = np.arange(float(self.Cm.shape[1]))
	    		self.Cm[r,:] = ((1./gamma)**2)*np.exp(-0.5*(((r-col)/cord)**2))
	    	self.CmInv = np.linalg.inv(self.Cm)

	    else:
		# in case of synthetic data, we can get away with a diagonal model covariance matrix (covariances = 0)
	    	Dmat=np.identity(nm)
	    	self.CmInv = (gamma**2)*Dmat

    	# to view the model covariance matrix (with iPython), use:
    	# x=np.arange(kcao.Cm.shape[0]); y=np.arange(kcao.Cm.shape[1])
    	# gx,gy=np.meshgrid(x,y)
    	# plt.pcolor(gx,gy,kcao.Cm)

        #----------------------------------------- End of damping --------------------------------------------

	def invert(self, nm, basis, m_prior, m_iter, mfk_pos, mfk_neg, Gmat_pos, Gmat_neg, deltad_pos, deltad_neg, dvar_pos, dvar_neg, COMPLETE=True):

		#----------------------------------------- CORE OPTIMIZATION ROUTINES --------------------------------------------

		def GAUSS_NEWTON():

			# compute basic (unweighted, undamped) gradient
			if np.sum(basis)!=0:
				# METHOD 1 for basic gradient
				kb_prod2 = mfk[br] * basis * 2*m_iter[:,None,None]
				ng1[br][:] = np.sum(kb_prod2, axis=(1,2)) * dxy**2
			else:
				ng1[br] = mfk[br].flatten()
			# METHOD 2 for basic gradient
			ng2[br] = np.matmul(G[br].T,dd[br])
			try:
				assert np.allclose(ng1[br],ng2[br],rtol=1e-03)
			except AssertionError:
				print("Gradient computed by approch 1: ", ng1[br])
				print("Gradient computed by approch 2: ", ng2[br])
				print(dd[br])
				raise SystemExit("Quitting. Problem computing gradient.")

			# effect of weighting
			Gt_CdInv = (G[br].T)*dvi[br]
			ngrad = np.matmul(Gt_CdInv,dd[br])

			# effect of damping
			ngrad_use = ngrad - np.matmul(self.CmInv,(m_iter - m_prior))

			if COMPLETE:
				# Hessian with weighting and damping
				hess_apx = np.matmul(Gt_CdInv,G[br])
				self.hess_use = hess_apx + self.CmInv

				#********** solve the linear system for the model update
				deltam[b,:] = np.linalg.solve(self.hess_use,ngrad_use)
			else:
				#********** simply save desired quantities
				grad_undamped[b,:] = -1 * ngrad

		#----------------------------------------- CORE INVERSION ROUTINES --------------------------------------------

		ngrad1_pos=np.empty(nm); ngrad2_pos=np.empty(nm)
		ngrad1_neg=np.empty(nm); ngrad2_neg=np.empty(nm)
		# there are two ways of computing the gradient of chi: with and without explicit use of
		# the G-matrix. In other words: using individual kernels or using the total misfit kernel.
		# We compute the gradient in both ways (hence subscripts 1, 2 on the variables) and ensure
		# they are equal, for confidence in the calculations.

		# dictionaries involving variables received as input
		mfk = {'p': mfk_pos, 'n': mfk_neg}
		G = {'p': Gmat_pos, 'n': Gmat_neg}
		dd = {'p': deltad_pos, 'n': deltad_neg}
		dvi = {'p': 1./dvar_pos, 'n': 1./dvar_neg}

		# dictionaries involving variables assigned here
		ng1 = {'p': ngrad1_pos, 'n': ngrad1_neg}
		ng2 = {'p': ngrad2_pos, 'n': ngrad2_neg}

		deltam = np.zeros((2,nm))
		if not COMPLETE:
			grad_undamped = np.zeros((2,nm))

		for b,br in enumerate(G):
		# br -> branch (positive or negative)
		    GAUSS_NEWTON()

		if COMPLETE:
			# combine the results from the positive and negative branches
			deltam_use = np.mean(deltam,axis=0)
			m_new = m_iter + deltam_use
			return m_new
		else:
			return grad_undamped


    #***************************************************************************
