#!/usr/bin/python

# Standard modules
import os
import sys
import utm

import numpy as np
import scipy.stats as sst
import scipy.interpolate as spi

sys.path.append(os.path.expanduser('~/code_general/modules.python'))
# path to the "SW1D_earthsr" set of modules

import SW1D_earthsr.read_earthsr_io as reo

##########################################################################################################################

class model_1D:

	def __init__(self, mod1d):
		self.oreo = reo.read_modfile([mod1d])
		# print self.oreo.struc[0], len(self.oreo.struc[0])

		vp = self.oreo.alpha
		vs = self.oreo.beta
		rho = self.oreo.rho

		uvals_vp, ind_vp = np.unique(vp, return_index=True)
		uvals_vs, ind_vs = np.unique(vs, return_index=True)
		uvals_rho, ind_rho = np.unique(rho, return_index=True)

		""" NB: numpy.unique returns SORTED unique VALUES by default. This can mess things up if
		parameter values are not all increasing with depth (e.g. low velocity layer at depth). Hence
		it is important to SORT the INDICES obtained above. """
		# uvals_vp=[b for a,b in sorted(zip(ind_vp,uvals_vp))]
		# uvals_vs=[b for a,b in sorted(zip(ind_vs,uvals_vs))]
		# uvals_rho=[b for a,b in sorted(zip(ind_rho,uvals_rho))]

		vp_ind = sorted(ind_vp)
		vs_ind = sorted(ind_vs)
		rho_ind = sorted(ind_rho)
		v_ind= np.union1d(vp_ind, vs_ind)
		hif_ind = np.union1d(v_ind, rho_ind)

		self.mod_hif = self.oreo.deps[hif_ind][1:]
		# first element of the array is ignored because it corresponds to the surface; z=0

		self.deps_all = self.oreo.deps[:-1]

	#***************************************************************************

	def fix_max_depth(self, indepth):

		max_ind = np.searchsorted(self.deps_all,indepth) + 1

		self.deps_tomax = self.deps_all[:max_ind]
		self.rho_tomax = self.oreo.rho[:max_ind]
		alpha_tomax = self.oreo.alpha[:max_ind]
		beta_tomax = self.oreo.beta[:max_ind]

		self.mu_tomax = self.rho_tomax * np.square(beta_tomax)
		self.lamda_tomax = self.rho_tomax * ( np.square(alpha_tomax) - (2*np.square(beta_tomax)) )

####################################################################################################################

class model_2D:

	def __init__(self, L, dxy):

		self.L = L
		self.dxy = dxy

	def npz_file(self, infile):

		loaded = np.load(infile)
		self.gx = np.unique(loaded['gx'])
		self.gy = np.unique(loaded['gy'])
		self.vel_acou = loaded['somod']
		print("Min and max values: ")
		print(np.amin(self.vel_acou), np.amax(self.vel_acou))

	def grd_file(self, infile):

		loaded = np.loadtxt(infile)
		x, y = utm.from_latlon(loaded[:,1], loaded[:,0])[0:2]
		x /= 1000; y /= 1000
		nx = (np.ceil(x[-1] - x[0])/self.dxy).astype(int)
		ny = (np.ceil(y[0] - y[-1])/self.dxy).astype(int)
		ggx = np.linspace(x[0],x[-1],nx)
		ggy = np.linspace(y[0],y[-1],ny)

		# interpolation to uniform grid
		[xx_reg, yy_reg] = np.meshgrid(ggx, ggy)
		points = np.asarray([x, y]).T
		vel_reg_grid = spi.griddata(points, loaded[:,2], (xx_reg, yy_reg), method='nearest')
		nx_inp = vel_reg_grid.shape[1]
		ny_inp = vel_reg_grid.shape[0]

		# create domain compatibe with cc code
		# vv = np.full((int(self.L/self.dxy +1),int(self.L/self.dxy +1)), sst.mode(vel_reg_grid,keepdims=False,axis=None)[0])
		vv = np.full((int(self.L/self.dxy +1),int(self.L/self.dxy +1)), sst.mode(vel_reg_grid,axis=None)[0])
		nx_out = vv.shape[1]
		ny_out = vv.shape[0]
		vv[(int(ny_out/2)-int(ny_inp/2)):(ny_inp+(int(ny_out/2)-int(ny_inp/2))),
		(int(nx_out/2)-int(nx_inp/2)):(nx_inp+(int(nx_out/2)-int(nx_inp/2)))] = vel_reg_grid
		self.vel_acou = vv

		xx = np.arange(((np.max(ggx)+np.min(ggx))/2 - self.dxy*(ny_out/2)),((np.max(ggx)+np.min(ggx))/2 + self.dxy*(ny_out/2)),self.dxy)
		yy = np.arange(((np.max(ggy)+np.min(ggy))/2 - self.dxy*(ny_out/2)),((np.max(ggy)+np.min(ggy))/2 + self.dxy*(ny_out/2)),self.dxy)
		xx[int((len(xx)-len(ggx))/2):(len(ggx)+int((len(xx)-len(ggx))/2))] = ggx
		yy[int((len(yy)-len(ggy))/2):(len(ggy)+int((len(yy)-len(ggy))/2))] = ggy

		self.gx = xx[0:vv.shape[0]]
		self.gy = yy[0:vv.shape[0]]

####################################################################################################################

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	vel_mod_file = os.path.expanduser(input("File containing 2-D velocity model: "))
	m2do = model_2D(75, 0.2)
	m2do.npz_file(vel_mod_file)
	# m2do.grd_file(vel_mod_file)
	plt.figure()
	plt.pcolor(m2do.gy, m2do.gx, m2do.vel_acou, cmap=plt.cm.jet.reversed())
	# plt.imshow(m2do.vel_acou, extent=(m2do.gx[0], m2do.gx[-1], m2do.gy[-1], m2do.gy[0]), cmap='jet')
	plt.colorbar()
	plt.show()
