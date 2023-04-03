import os
import sys
import math
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import config_file as config

# global variables
reald = config.ext_data
dxy = config.dom_geom.dx
gamma = config.invc.gamma_inv
ngpmb = config.dom_geom.ngp_box


if reald:
	import obspy.core as oc
# 	try:
# 		# Custom module
# 		import azimuthal_anal_cc as azan
# 		use_azan=True
# 	except ImportError:
# 		use_azan=False
# 	pass


###############################################################################################################
class setup_modelling_domain:

	def __init__(self, st_no, st_id, xrec, yrec, ox, oy, num_select, make_plots=True):

		global actdrp, actnrp, chosen_st_id, chosen_st_no

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

		if num_select < nrecs_total:
			sta_subset_file = input("File containing receiver subset (enter 0 for automatic selection): ")
			if sta_subset_file.isdigit():
				#************** Automated selection criterion for selecting a subset of entire array *************
				ro_act = np.sqrt(recx_ro**2 + recy_ro**2)
				ro_grid = np.sqrt((recx_igp*dxy)**2 + (recy_igp*dxy)**2)
				grd_err = np.abs(ro_act-ro_grid)
				grd_err_chosen = np.sort(grd_err)[:num_select]
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

		elif num_select == nrecs_total:
			ichosen = np.arange(nrecs_total)

		self.act_recx_rel=recx_ro[ichosen]
		self.act_recy_rel=recy_ro[ichosen]

		try:
			assert len(ichosen)==num_select
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

		#***************************** compute pairwise distances ************************************
		self.act_dist_rp=np.zeros((num_select,num_select))
		# self.act_dist_rp -> actual_distance_receiver_pairs. The word "actual" is used in the name so as to distinguish
		# these distances from the "effective" ones which are computed, in the h13 module, using the approximate
		# reciever locations on the uniform grid of h13.
		act_rnum_rp=np.zeros((num_select,num_select), dtype=object)
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

		actdrp = self.act_dist_rp
		actnrp = act_rnum_rp
		self.dist_1Darr = self.act_dist_rp[np.nonzero(np.tril(self.act_dist_rp))]
		try:
			assert self.dist_1Darr.size == int(num_select*(num_select-1)/2)
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

		# axrp.set_xlim(-13,12)
		# axrp.set_ylim(-15,15)
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

class cc_data:

	class Read:

		def __init__(cc_data, inp_main, data_format, new_deltat):

			FuncDic={'binary_archive_python': cc_data.archive, 'individual_files_ccpairs': cc_data.dir_ifccp}

			global si, reclen, fpband, ccl, data_list_form, d_form
			global master, slave, allstored
			# global ctyp

			d_form = data_format
			cc_data.nsi = new_deltat if (not new_deltat is None) else None

			# cc_data.inpfile = inp_main
			FuncDic[data_format](inp_main)

			si=cc_data.sami
			fpband=cc_data.fpb
			ccl=cc_data.cclags
			data_list_form=cc_data.cookie

			try:
				master = cc_data.MASTER
				slave = cc_data.SLAVE
			except AttributeError:
				assert data_format=='binary_archive_python'
				master=None
				slave=None

			try:
				allstored=cc_data.recarray
			except AttributeError:
				assert data_format=='individual_files_ccpairs'
				allstored=None

			reclen=2*cc_data.cclags[-1]

			print("\n\nSignal parameters of read data- ")
			print("Number of samples: ", cc_data.cclags.size)
			print("Sample spacing (time domain): ", si)

		#---------------------------------------------------------------------------------------------------------

		def dir_ifccp(cc_data, inp_dir):

			f_exten = input("File extension: ")

			names_list=os.listdir(inp_dir)
			filenames=[n for n in names_list if n.endswith(f_exten)]
			# print(filenames)
			cc_data.SLAVE = []
			cc_data.MASTER = []
			Dat = []
			b_time = []
			e_time = []
			si_list = []
			np_list = []

			for i,f in enumerate(filenames):
				file = os.path.join(inp_dir,f)

				if (oc.read(file)[0].stats.sac.kstnm in chosen_st_id) and (oc.read(file)[0].stats.sac.kevnm in chosen_st_id):
				# read only those files which correspond to selected stations

					if cc_data.nsi is None:
						trace = oc.read(file)
						data = trace[0].data
						SI = trace[0].stats.delta
						NP = trace[0].stats.npts
					else:
						trace = oc.read(file)
						NP_calc = (trace[0].stats.sac.e - trace[0].stats.sac.b) / cc_data.nsi
						## take care of float errors
						if np.abs(NP_calc - np.round(NP_calc)) < trace[0].stats.delta:
							NP = int(np.round(NP_calc))
						else:
							NP = int(NP_calc)
						## ensure new number of points stays even or odd
						if trace[0].stats.npts % 2 !=0 :
							# print("HELLO", NP)
							# print(trace[0].stats.sac.e, trace[0].stats.sac.b, trace[0].stats.sac.e - trace[0].stats.sac.b)
							NP += 1

						data = ss.resample_poly(trace[0].data, up=NP, down=trace[0].stats.npts)
						# SI = trace[0].stats.delta*trace[0].stats.npts/NP
						SI = cc_data.nsi

					# print("CHECKING!! ")
					# print(trace[0].stats.npts, NP)

					#data = trace[0].data
					#SI = trace[0].stats.delta
					#NP = trace[0].stats.npts
					# NB: npts and delta from SAC headers must be read as above, NOT with ".stats.sac" as below
					s = trace[0].stats.sac.kstnm
					e = trace[0].stats.sac.kevnm
					b_t = trace[0].stats.sac.b
					e_t = trace[0].stats.sac.e

					cc_data.MASTER.append(e)
					cc_data.SLAVE.append(s)
					Dat.append(data)
					b_time.append(b_t)
					e_time.append(e_t)
					si_list.append(SI)
					np_list.append(NP)

			if np.unique(si_list).size == np.unique(b_time).size == np.unique(e_time).size == 1:
				cc_data.sami = np.unique(si_list)[0]
				cc_data.npts = np.unique(np_list)[0]
				# ************ Use either one of the following, whichever works for the data in question ********
				# cc_data.cclags=np.arange(np.unique(b_time)[0]+cc_data.sami,np.unique(e_time)[0]+cc_data.sami,cc_data.sami)
				cc_data.cclags=np.linspace(np.unique(b_time)[0],np.unique(e_time)[0],cc_data.npts)
				# ***********************************************************************************************
				cc_data.cookie=Dat
				freq_info = input("Frequency band of the processed cross-correlations (lf, hf, alpha (Tukey)): ")
				cc_data.fpb = [float(j) for j in freq_info.split()]
			else:
			    raise SystemExit("Could not find data files OR, could not find unique values in the data, for one or more of the following quantities: si, b_time, e_time")

		#---------------------------------------------------------------------------------------------------------

		def archive(cc_data, inp_file):

			# NB: only DECIMATION (downsampling) of data implemented in this case.

			loaded = np.load(inp_file)
			print("Reading ", inp_file)

			try:
			    cc_data.recarray=loaded['reclist']
			    cc_data.fpb=loaded['fpband']
			    try:
			        sami_orig=loaded['si'][0]
			    except IndexError:
			        sami_orig=loaded['si']
			    cclags_orig=loaded['cclags']
			    cookie_entire=loaded['cookie']
			except KeyError:
			    raise SystemExit("Problem reading %s" %(inp_file))

			cclags_orig=np.array([float("%.2f" %x) for x in cclags_orig])
			# making sure that ccl does not contain elements with more significant digits than are allowed by the sampling interval

			if len(cookie_entire.shape)==2:
			    # same-component cross-correlations
			    ctype='sc'
			elif len(cookie_entire.shape)==3:
			    # inter-component cross-correlations
			    ctype='ic'

			if cc_data.nsi is None:
				cc_data.sami = sami_orig
				cc_data.cclags = cclags_orig
				cookie_resamp = cookie_entire
			else:
				cc_data.sami = cc_data.nsi
				cookie_resamp = ss.decimate(cookie_entire, int(cc_data.nsi/sami_orig), axis=0)
				cc_data.cclags = np.linspace(cclags_orig[0], cclags_orig[-1], cookie_resamp.shape[0])

			# print("cookie shape: ")
			# print(cookie_entire.shape)
			# print(cookie_resamp.shape)

			# read data corresponding to selected stations
			cc_data.cookie = []
			nrtotal=cc_data.recarray.size

			for b,brec in enumerate(chosen_st_no[:-1]):
				rbefore=list(cc_data.recarray).index(brec)
				pbefore=int(rbefore*(2*(nrtotal-1)-(rbefore-1))/2)	# sum of terms of an AP with d=-1
				urecs=chosen_st_no[b+1:]
				crbr = list(map(lambda x: pbefore - 1 + (x-brec), urecs))
				# crbr is Columns_Relevant_to_Base_Receiver
				if b>0:
					assert len(crbr)==prev-1

				# print("Receiver ", brec)
				# print("rbefore and pbefore: ", rbefore, pbefore)
				# print("Selecting columns: ", crbr)

				for c in crbr:
					cc_data.cookie.append(cookie_resamp[:,c])
				prev=len(crbr)

			nc2 = int((len(chosen_st_no)*(len(chosen_st_no)-1))/2)
			assert len(cc_data.cookie)==nc2

	#*******************************************************************************************************

	class MatrixForm:

		# def __init__(self, rn_all, rx_act, ry_act, nrchosen, rnchosen):
		def __init__(MF, nrecs_chosen):

			# global MF.selec_data, actdrp, actnrp
			global data_matrix_pairs

			try:
				assert nrecs_chosen==len(chosen_st_no)
				assert nrecs_chosen==len(chosen_st_id)
			except AssertionError:
				raise SystemExit("Problem identifying chosen station subset")


			FuncDic={'binary_archive_python': MF.archive, 'individual_files_ccpairs': MF.dir_ifccp}

			MF.selec_data=np.zeros((ccl.size, nrecs_chosen, nrecs_chosen))

			FuncDic[d_form](nrecs_chosen)

			data_matrix_pairs = MF.selec_data

		#---------------------------------------------------------------------------------------------------------

		def dir_ifccp(MF, n):

			nc2 = int(n*(n-1)/2)

			try:
				assert len(data_list_form) == nc2
				MF.nmissing = 0
			except AssertionError:
				if len(data_list_form) > nc2:
					raise SystemExit("\nFound more cc-files (%d) than are allowed (%d) by no. of station selected! Please check.\n" %(len(data_list_form), nc2))
				else:
					print("\nWARNING: Some cc-pairs are missing. Only %d available out of %d\n" %(len(data_list_form), nc2))
					mdud=input("Confirm 'y' to continue, any other key to exit: ")
					if not (mdud == 'y' or mdud == 'Y'):
						print(mdud, len(mdud))
						raise SystemExit("Aborted")
					else:
						MF.nmissing = nc2 - len(data_list_form)

			cc_pairs=zip(master,slave)

			file_counter=0
			for p,ccp in enumerate(cc_pairs):
				file_counter+=1
				row=chosen_st_id.index(ccp[0])
				col=chosen_st_id.index(ccp[1])
				print("cc-file number %d, for stations: %s %s (%d %d)" %(file_counter, ccp[0],ccp[1],row,col))
				# for consistency with h13, the LOWER TRIANGULAR part of the cc-matrix MUST be filled in.
				# for completeness however, we fill in the entire matrix (except the main diagonal, which is 0).
				if row>col:
					# lower-triangular part
					MF.selec_data[:,row,col]=data_list_form[p]
					# upper-triangular part
					MF.selec_data[:,col,row]=np.flip(data_list_form[p])
				elif row<col:
					# upper-triangular part
					MF.selec_data[:,row,col]=data_list_form[p]
					# lower-triangular part
					MF.selec_data[:,col,row]=np.flip(data_list_form[p])
				elif row==col:
					raise SystemExit("\nProblem building cc-matrix: encountered same master and slave station.\n")

			MF.mark_avail_data = np.sum(np.abs(MF.selec_data), axis=0)
			# print("DATA AVAILABILITY CHECK!")
			# print(MF.mark_avail_data)
			present = np.nonzero(np.tril(MF.mark_avail_data))
			assert present[0].size == len(data_list_form)

		#---------------------------------------------------------------------------------------------------------

		def archive(MF, n):

			MF.nmissing = 0
			# NO MISSING DATA in this case, it is assumed.

			nrtotal=allstored.size

			# NB: in h13, the synthetic ccs computed directly are those corresponding to the LOWER TRIANGULAR part
			# of the cross-correlation matrix, i.e. Cjk is computed, where j>k. In case of the archived data, what is
			# stored is the UPPER TRIANGULAR part, i.e. Ckj is stored. These two are flipped versions of each other,
			# so we need to flip the stored data when filling in the lower triangular part of the matrix.

			counter=0

			for j in range(MF.selec_data.shape[1]):
				for k in range(j+1,MF.selec_data.shape[1]):
					MF.selec_data[:,j,k] = data_list_form[counter]
					# upper triangular part, filled in for completeness
					MF.selec_data[:,k,j] = np.flipud(data_list_form[counter])
					# lower triangular part, necessary for compatibility with h13
					counter+=1

			MF.mark_avail_data = np.sum(np.abs(MF.selec_data), axis=0)

		#********************************************************************************************************

	class Process:

		def __init__(self, wsp, num_chosen, num_missing):

			nyq = 1/(2*si)
			nmul = 5
			new_nyq = min(nyq,nmul*fpband[1]) # lower bound for the Nyquist frequency we now seek

			self.dt = si
			self.ccl_used = ccl
			self.nsam = data_list_form[0].shape[0]
			self.use_data = data_matrix_pairs

			self.SNR_and_taper(wsp, num_chosen, num_missing)

		#---------------------------------------------------------------------------------------------------------

		def SNR_and_taper(self, wspeed, nrchosen, nmissing):

			# cslow=1.5
			# cfast=6.0
			# tstart=actdrp/cfast
			# tend=actdrp/cslow

			self.lefw = -4.0
			self.rigw = +4.0
			tstart=actdrp/wspeed + self.lefw
			tstart[tstart<0] = 0.
			tend=actdrp/wspeed + self.rigw

			tap_mid=False
			# If True, the complete cc is tapered not only at the ends but also in the middle. This is
			# achieved by individually tapering the positive and negative branches on the "start" side.
			# In DHG_2019, we kept this parameter set to False, meaning that the entire cc is
			# treated as one waveform, not separated into positive and negative branches.

			pb_iws=np.searchsorted(self.ccl_used,tstart)
			pb_iwe=np.searchsorted(self.ccl_used,tend)
			nb_iws=np.searchsorted(self.ccl_used,-tend)-1
			nb_iwe=np.searchsorted(self.ccl_used,-tstart)-1
			# p/nb_iws/e stands for positive/negative_branch_index_of_window_start/end

			# self.snr=1e3*np.ones((nrchosen,nrchosen))
			self.snr = np.eye(nrchosen) - 1.0 # so that diagonal is 0, other terms are non-zero

			hnsam = int((self.nsam)/2) if (self.nsam)%2==0 else int(((self.nsam)-1)/2)
			# hnsam -> half_the_number_of_samples
			cchlen = reclen/2
			# cchlen -> cross-correlation_half_length
			if tap_mid:
				tap_portion=np.round(tstart/cchlen,2)
			else:
				tap_portion=np.round((cchlen - tend)/reclen,2)

			cmissing=0

			for i in range(nrchosen-1):
				for j in range(i+1,nrchosen):
					#****************** first compute the S/N ratio ******************
					ud=self.use_data[:,j,i]
					out_win=np.copy(ud)
					ns_ow = hnsam - (pb_iwe[j,i] + 1 - pb_iws[j,i])
					# ns_ow -> number-of_samples_outside_window

					# positive branch window
					pbwin=ud[pb_iws[j,i]:pb_iwe[j,i]+1]
					out_win[pb_iws[j,i]:pb_iwe[j,i]+1]=0.0
					egy_win_pb=np.mean(np.square(pbwin))
					egy_owin_pb=np.sum(np.square(out_win[hnsam+1:]))/(ns_ow)
					# print(egy_win_pb,egy_owin_pb)
					with np.errstate(invalid='raise'):
						try:
							self.snr[j,i] = egy_win_pb/egy_owin_pb
						except FloatingPointError as e:
							# this should happen only in case of MISSING DATA
							self.snr[j,i] = np.nan
							cmissing+=1

					# negative branch window
					nbwin=ud[nb_iws[j,i]:nb_iwe[j,i]+1]
					out_win[nb_iws[j,i]:nb_iwe[j,i]+1]=0.0
					egy_win_nb=np.mean(np.square(nbwin))
					egy_owin_nb=np.sum(np.square(out_win[:hnsam]))/(ns_ow)
					with np.errstate(invalid='raise'):
					# negative branch of [p,q] is positive branch of [q,p] cross-correlation
						try:
							self.snr[i,j] = egy_win_nb/egy_owin_nb
						except FloatingPointError as e:
							# this should happen only in case of MISSING DATA
							self.snr[i,j] = np.nan
							cmissing+=1

					#**************** then do the tapering ***************************
					if reald:
						# NB: obspy operations are performed in-place. So when you taper a
						# trace, even the NumPy array which forms the tr.data gets tapered!!!
						if tap_mid:
						# tapering the positive and negative branches individually
							branches = np.array_split(ud,2)
							tr_nb=oc.trace.Trace(data=branches[0], header={'delta': self.dt})
							tr_pb=oc.trace.Trace(data=branches[1], header={'delta': self.dt})
							tr_nb.taper(max_percentage=tap_portion[j,i],side='right')
							tr_pb.taper(max_percentage=tap_portion[j,i],side='left')

						tr=oc.trace.Trace(data=ud, header={'delta': self.dt})
						tr.taper(max_percentage=tap_portion[j,i])

			try:
				assert cmissing == 2*nmissing
			except AssertionError:
				print("From u2.Process:")
				print(cmissing, nmissing)
				raise SystemExit("Problem with missing data when computing SNR values.")

	#***************************************************************************************************************

	class Errors:

	    def __init__(self,num_chosen,stepaz):

	        self.DelE = np.zeros((num_chosen,num_chosen))
	        # in h13 we only need one (triangular) half of this matrix to cover ALL receiver pairs
	        # However, we fill both halves here to account for the positive and negative branches
	        # lower triangular half of DelE -> \Delta E for positive branch of h13-relevant waveform
	        # upper triangular half of DelE -> \Delta E for negative branch of h13-relevant waveform

	        swpmax=180

	        if len(data_list_form.shape)==2:
	        	# same-component cross-correlations
	        	ctype='sc'
	        elif len(data_list_form.shape)==3:
	        	# inter-component cross-correlations
	        	ctype='ic'

	        if use_azan:

	        	for az in np.arange(0,swpmax,stepaz):
	        		saefo = azan.do_single_azimuth(data_list_form,allstored,si,ccl,ctype,fpband,az,stepaz,(False,False,False,True))
	        		# saefo -> single_azimuth_energy_fitting_object
	        		for k,actrp in enumerate(saefo.azrp):
	        			h13_ind = np.where(actnrp==actrp)
	        			if len(h13_ind[0])>0:
	        				# print(k, actrp, h13_ind)
	        				self.DelE[h13_ind] = saefo.res_ep_pb[k]
	        				self.DelE[h13_ind[::-1]] = saefo.res_ep_nb[k]
	        			else:
	        				pass

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
		# try:
		# 	np.savez_compressed(archfile, t=self.oica.t, dt=self.oica.deltat, drp=self.oica.dist_rp_grid, wsp=self.wspeed, wobs=self.oica.obscross,\
		# 	 wsyn_i=self.oica.allit_syncross[0], wsyn_f=self.oica.allit_syncross[-1], win_ind=win_lr_np)
		# except AttributeError:
		# 	# use "flit" variables instead of "allit"
		# 	np.savez_compressed(archfile, t=self.oica.t, dt=self.oica.deltat, drp=self.oica.dist_rp_grid, wsp=self.wspeed, wobs=self.oica.obscross,\
		# 	wsyn_i=self.oica.flit_syncross[0], wsyn_f=self.oica.flit_syncross[-1], win_ind=win_lr_np)

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
			pickle.dump(list(zip(chosen_st_no,chosen_st_id)),jar)
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
			pickle.dump(list(zip(chosen_st_no,chosen_st_id)),jar)
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
				axm.annotate(chosen_st_id[j], xy=(dxy*self.osmd.rchosenx_igp[j],dxy*self.osmd.rchoseny_igp[j]), color='white')
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
			try:
				assert pair_counter==len(data_list_form)
			except AssertionError:
				print(pair_counter, len(data_list_form))
				raise SystemExit("From u2 post_run: problem with pair count.")

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
			ax_snr_nb.text(0.7, 0.8, "%d/%d" %(n_nb, len(data_list_form)), transform=ax_snr_nb.transAxes)
			ax_snr_nb.set_xlabel('Negative branch')
			ax_snr_pb.hist(snr_pb,bins=hbe_snr_p,edgecolor='black')
			ax_snr_pb.text(0.7, 0.7, "%d/%d" %(n_pb, len(data_list_form)), transform=ax_snr_pb.transAxes)
			ax_snr_pb.set_xlabel('Positive branch')

			# -------------- energy vs distance plot -------------
			fig_egy = plt.figure()
			ax_egy=fig_egy.add_subplot(111)
			ax_egy.scatter(self.oica.dist_rp_sorted,self.oica.egy_obs)
			ax_egy.plot(self.oica.dist_rp_sorted,self.oica.oef)
			ax_egy.set_xlabel("Distance [km]", fontsize=14)
			ax_egy.set_ylabel("Energy", fontsize=14)

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

	# def invert(self, nm, basis, m_prior, mc_iter, mfk_pos, mfk_neg, Gmat_pos, Gmat_neg, deltad_pos, deltad_neg, dvar_pos, dvar_neg, COMPLETE=True):
	def invert(self, nm, basis, m_prior, m_iter, mfk_pos, mfk_neg, Gmat_pos, Gmat_neg, deltad_pos, deltad_neg, dvar_pos, dvar_neg, COMPLETE=True):

		# print("CHECK U2-1: ")
		# print(np.sum(mfk_pos))
		# print(np.sum(mfk_neg))

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
		if COMPLETE:
			mod_update = np.zeros((ngpmb, ngpmb))
		else:
			grad_undamped = np.zeros((2,nm))

		#----------------------------------------- CORE INVERSION ROUTINES --------------------------------------------

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
				print(ng1[br])
				print(ng2[br])
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

		for b,br in enumerate(G):
		# br -> branch (positive or negative)
		    GAUSS_NEWTON()

		if COMPLETE:
			# combine the results from the positive and negative branches
			deltam_use = np.mean(deltam,axis=0)
			m_new = m_iter + deltam_use

			# perform model update
			# mc_iter.append(m_new)
			# mod_update = np.einsum('k,klm',deltam_use,basis)

			# return mc_iter, mod_update
			return m_new

		else:
			return grad_undamped


    #***************************************************************************
