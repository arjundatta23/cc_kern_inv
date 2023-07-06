#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/code_general/devito'))
if not __name__ == '__main__':
    sys.path.append('devito_solvers_TD/Helmholtz_2D')

from devito import configuration
# import devito.configuration
configuration['log-level'] = 'WARNING'

# NBVAL_IGNORE_OUTPUT
from examples.seismic import demo_model, Model, Receiver
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.source import PointSource, TimeAxis
# from examples.seismic import plot_shotrecord, plot_velocity

##########################################################################################################################

class sim_int_sources:

    def __init__(self, orig, dxdy, vp0, dt, pow_spec):

        self.ngp_bdry=80  # boundary thickness (in number of grid points)
        self.vel_model = Model(vp=vp0, origin=orig, shape=vp0.shape, spacing=dxdy, space_order=4, nbl=self.ngp_bdry, bcs="damp")
        self.use_dt = dt
        self.use_nt = len(pow_spec)
        ### Obtain wavelet from its autocorrelation (power spectrum)
        src_wavelet_powspec = np.fft.fftshift(np.fft.ifft(np.sqrt(pow_spec))).real

        ### Force devito solver to work with input dt

        try:
            assert self.use_dt < self.vel_model.critical_dt
        except AssertionError:
            print("Input dt is %f ms, should be < %f ms" %(self.use_dt, self.vel_model.critical_dt))
            raise SystemExit("From devito acoustic solver: problem with input signal characteristics")

        self.vel_model.dt_scale = self.use_dt / self.vel_model.critical_dt

        self.src_wavelet = src_wavelet_powspec

        #### Maximum frequency that can be propagated by a 2nd order-time, 4th order-space scheme
        freqs = np.fft.rfftfreq(self.use_nt, self.use_dt*1e-3)
        pow_spec_dB = 10 * np.log10(pow_spec[0:len(freqs)].real/np.max(pow_spec.real))
        freq_20dB = freqs[pow_spec_dB>-20.0][-1]
        dx_freq_20dB = 1e3*np.min(vp0)/(5*freq_20dB)
        fmax = 1e3*np.min(vp0)/(5*np.max(dxdy))

        try:
            assert freq_20dB<fmax
        except AssertionError:
            print("Max freq is %f Hz, should be < %f Hz" %(freq_20dB, fmax))
            raise SystemExit("From devito acoustic solver: problem with input power spectrum")

        try:
            assert np.max(dxdy)<dx_freq_20dB
        except AssertionError:
            print("Grid spacing is % m, should be < %f m" %(np.max(dxdy), dx_freq_20dB))
            raise SystemExit("From devito acoustic solver: problem with input grid parameters")

    # --------------------------------------------------------------------------

    # def solve(self, param, src_data, src_coord, rec_coord, save_wfld):
    def solve(self, src_coord):

        ### rec_coord is NOT actually used in this function
        ### save_wfld=True ## this option should always be set to True

        solver_origin = np.asarray(self.vel_model.origin)*1e3
        solver_src_coord = src_coord - solver_origin
        #solver_rec_coord = rec_coord - solver_origin
        solver_rec_coord = solver_origin   ### Dummy receiver coordinate

        # Incorporate the source
        trange = TimeAxis(start=0, step=self.vel_model.critical_dt, num=self.src_wavelet.shape[0])

        # Geometry
        geometry = AcquisitionGeometry(self.vel_model, solver_rec_coord, solver_src_coord,
                                       trange.start, trange.stop)

        src = PointSource(name='spat_src', grid=self.vel_model.grid, time_range=trange, space_order=4, coordinates=solver_src_coord,
                          data=self.src_wavelet.reshape((self.src_wavelet.shape[0],1))/np.prod(self.vel_model.spacing))
        AcquisitionGeometry.src = src

        # print("From solve: src_data.shape, src_coord.shape: ", src_data.shape, src_coord.shape)

        self.stf = geometry.src
        # print("From solve: self.stf.shape: ", self.stf.shape)

        # Set up solver.
        solver = AcousticWaveSolver(self.vel_model, geometry, space_order=4)

        # Generate synthetic receiver data and wave field from  model.
        #_, self.wav_fld_bound_include, _ = solver.forward(vp=self.vel_model.vp, save=True)
        _, self.wav_fld_bound_include, _ = solver.forward(save=True)
        # saving the wavefield for all timesteps

        self.wav_fld = self.wav_fld_bound_include.data[:,self.ngp_bdry:-self.ngp_bdry,self.ngp_bdry:-self.ngp_bdry]

        #wav_fld = wav_fld_bound_include.data[:, ngp_bdry:-ngp_bdry, ngp_bdry:-ngp_bdry]

        # plt.figure()
        # pd.plot_velocity(self.vel_model, source=src_coord, receiver=rec_coord)

    # --------------------------------------------------------------------------

    def resample(self):

        ### Resample data using devito method, specify new dt in ms
        #self.stf_resamp = ss.resample_poly(self.stf.data, up=self.use_nt, down=self.stf.shape[0], axis=0).ravel()
        self.stf_resamp = self.stf.data
        ### devito method not available for wavefield, so use scipy. Note that scipy requires
        ### number of samples as input argument
        #self.wav_fld_resamp = ss.resample_poly(self.wav_fld, up=self.use_nt, down=self.stf.shape[0], axis=0)
        self.wav_fld_resamp = self.wav_fld

    def get_FD(self):

        self.wav_fld_resamp_FD = np.fft.rfft(self.wav_fld_resamp, axis=0)

        if np.isnan(np.sum(self.wav_fld_resamp_FD)):
            raise Exception("From devito acoustic solver: problem computing the wavefield.")
        else:
            return np.conj(np.transpose(self.wav_fld_resamp_FD, [0,2,1]))


##########################################################################################################################

if __name__ == "__main__":

    import scipy.special as ssp
    import plotting_devito as pd

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # import wiggle.wiggle as wigplot

    def Ricker_wav(t, f0, phase_shift):
        t0 = 1.0 /f0
        tmp = (np.pi * f0 * (t - t0)) ** 2
        wt = (1.0 - 2.0 * tmp) * np.exp(-tmp)
        phase = phase_shift * np.pi / 180
        hlb = ss.hilbert(wt)
        wt = np.cos(phase) * np.real(hlb) - np.sin(phase) * np.imag(hlb)
        return wt


# ******************************************************************************
# Set simulation parameters
# ******************************************************************************

    sim_par = {'t0': 0.,
               'tn': 49600.,              # Simulation length (ms)
               'f0': 0.2*1e-3             # Source peak frequency (kHz)
                }

    dom_orig = [-60.0, -60.0]           # starting values of domain (km)
    # dxy = [150, 150] # grid spacing [x,y] (m)
    # ngp = [501, 201] # number of grid points [x,y]
    dxy = [1000, 1000] # grid spacing [x,y] (m)
    ngp = [121, 121] # number of grid points [x,y]
    wavespeed = 3.0*np.ones(ngp)


# ******************************************************************************
# Run a simulation
# ******************************************************************************

    anseicca_dt = 150
    t = np.arange(sim_par['t0'], sim_par['tn']+anseicca_dt, anseicca_dt)
    wt = Ricker_wav(t, sim_par['f0'], 0)
    pow_spec = np.multiply(np.conj(np.fft.fft(wt)),np.fft.fft(wt))
    rso = sim_int_sources(dom_orig, dxy, wavespeed, anseicca_dt, pow_spec)

    # Simulate a point source at the center of the model:

    # Source location(s)
    src_coordinates = np.empty((1, 2), dtype=np.float32)
    src_coordinates[:, 0] = (ngp[0] - 1) * 0.5 * dxy[0] + dom_orig[0] * 1e3
    src_coordinates[:, 1] = (ngp[1] - 1) * 0.5 * dxy[1] + dom_orig[1] * 1e3

    rso.solve(src_coordinates)
    rso.resample()
    rso.get_FD()


# ******************************************************************************
# Post-run
# ******************************************************************************


    model_extent = [rso.vel_model.origin[0], rso.vel_model.origin[0] + rso.vel_model.domain_size[0]* 1e-3,
                    rso.vel_model.origin[1] + rso.vel_model.domain_size[1]* 1e-3, rso.vel_model.origin[1]]

    # ******************************************************************************
    # Compute analytical wavefield
    # ******************************************************************************

    fft_src = (np.fft.rfft(rso.stf_resamp*np.prod(dxy)))
    freqs = (np.fft.rfftfreq(rso.use_nt,anseicca_dt*1e-3))

    an_wav_fld_FD = np.zeros((len(freqs), ngp[0], ngp[1]), dtype='complex_')

    x  = (np.arange(0,ngp[0])*dxy[0]*1e-3 + dom_orig[0])
    z  = (np.arange(0,ngp[1])*dxy[1]*1e-3 + dom_orig[1])
    [zz, xx] = np.meshgrid(z,x)
    xs = src_coordinates[0,0]
    zs = src_coordinates[0,1]

    for fidx in np.arange(0,len(freqs)):
        k = (2.0*freqs[fidx]*np.pi/(wavespeed[0,0]))
        an_wav_fld_FD[fidx,:,:] = -1j*fft_src[fidx]*0.25 * ssp.hankel2(0, k * np.power((np.power((zz - zs), 2) + np.power((xx - xs), 2)),
                                                                0.5) + np.finfo(np.float32).eps)
    an_wav_fld = (np.fft.irfft(an_wav_fld_FD, axis=0).real)


    ### plot wavefield at an intermediate time step
    nt = len(pow_spec)
    tstep = round(200*anseicca_dt/1000,2)
    fig, ax = plt.subplots(1,3, figsize=(18,6), sharey=True)
    plot=ax[0].imshow(rso.wav_fld_resamp[200,:,:].T, extent=model_extent, cmap='seismic')
    ax[0].set_xlabel('x (km)', fontsize=20)
    ax[0].set_ylabel('z (km)', fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    plt.colorbar(plot, cax=cax, orientation='vertical')
    ax[0].set_title('Devito')
    plot=ax[1].imshow(an_wav_fld[200,:,:].T, extent=model_extent, cmap='seismic')
    ax[1].set_xlabel('x (km)', fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    plt.colorbar(plot, cax=cax, orientation='vertical')
    ax[1].set_title('Analytical')
    plot=ax[2].imshow(an_wav_fld[200,:,:].T-rso.wav_fld_resamp[200,:,:].T, extent=model_extent, cmap='seismic')
    ax[2].set_xlabel('x (km)', fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    plt.colorbar(plot, cax=cax, orientation='vertical')
    ax[2].set_title('Analytical-Devito')
    plt.suptitle('wavefield, t = '+str(tstep)+'s, source is at '+str(src_coordinates))


    """
    try:
    # plot an image of the frequency-domain wavefield at frequency=f0
        print(rso.wav_fld_resamp_FD.shape)
        idx = np.abs(rso.freqs - sim_par['f0']).argmin()
        idx2 = np.searchsorted(rso.freqs, sim_par['f0'])
        print(idx, idx2)
        plt.figure()
        plt.imshow(np.real(rso.wav_fld_resamp_FD[idx,:,:].transpose()), extent=model_extent)
        plt.colorbar()
        plt.title('Frequency = '+ str(sim_par['f0']*1e3)+'Hz')
    except AttributeError:
        print("Frequency-domain wavefield not obtained")
    """

    ### DO anseicca correlation

    file_path=input("Path to anseicca pickle file (or write 'n' to skip): ")

    if file_path=='n':
        print("No pickle file provided. End of code.")
    else:
        sys.path.append(os.path.expanduser('~/Research/code_own/cc_kern_inv/modules_common'))
        sys.path.append(os.path.expanduser('modules_common'))
        import read_pickle_output as rpo

        pfile=os.path.expanduser(file_path)
        print("Reading ", pfile)

        rapo = rpo.read_anseicca_pickle(pfile)

        kcao_X = rapo.cg_dom_geom.X
        kcao_Y = rapo.cg_dom_geom.Y
        kcao_dx = rapo.cg_dom_geom.dx
        spacing = [np.median(np.diff(kcao_X))*1e3, np.median(np.diff(kcao_Y))*1e3]
        ngp_xy = [rapo.cg_dom_geom.X.size, rapo.cg_dom_geom.Y.size]
        origin = np.asarray([rapo.cg_dom_geom.X[0], rapo.cg_dom_geom.Y[0]])

        kcao_t = rapo.kcao_schar.tt
        kcao_dt = rapo.kcao_schar.dt*1e3
        kcao_dom = rapo.kcao_schar.domega
        kcao_fhz = rapo.kcao_schar.fhz
        kcao_nom = rapo.kcao_schar.nsam
        kcao_nom_nneg = rapo.kcao_schar.n_nn_fsam

        # print("Checking new input")
        # print(kcao_X.shape)
        # print(kcao_Y.shape)
        # print(rapo.kcao_sdtrue.shape)
        # print(spacing)
        # print(kcao_dt)
        # print(rapo.kcao_obscross.shape)

        origin = np.asarray([kcao_X[0], kcao_Y[0]])
        vp0 = rapo.cg_scal_mod.wavspeed_scal2D * np.ones(ngp_xy)

        rso = sim_int_sources(origin, spacing, vp0, kcao_dt, rapo.kcao_pss)

        #### Now simulate wavefields in turn at two receiver locations

        ixa=6
        ixb=19

        rec_coordinates = np.empty((rapo.rc_xp.size, 2))
        rec_coordinates[:, 0] = rapo.rc_xp*spacing[0]
        rec_coordinates[:, 1] = rapo.rc_yp*spacing[1]

        xa_coordinates = np.empty((1,2))
        xa_coordinates[:, 0] = rec_coordinates[ixa,0]
        xa_coordinates[:, 1] = rec_coordinates[ixa,1]

        rso.solve(xa_coordinates)
        rso.resample()
        wfld_FD_xa=rso.get_FD()

        xb_coordinates = np.empty((1,2))
        xb_coordinates[:, 0] = rec_coordinates[ixb,0]
        xb_coordinates[:, 1] = rec_coordinates[ixb,1]

        rso.solve(xb_coordinates)
        rso.resample()
        wfld_FD_xb=rso.get_FD()

        # Now perform cross-correlation using the numerically-computed wavefields
        freq_cc_devito = np.zeros(kcao_nom, dtype='complex')

        fhzp = len(kcao_fhz[kcao_fhz>0])
        fhzn = len(kcao_fhz[kcao_fhz<0])
        ssna = abs(fhzn-fhzp)

        f_cc = wfld_FD_xa[1:kcao_nom_nneg,...] * np.conj(wfld_FD_xb[1:kcao_nom_nneg,...])
        fcc = f_cc * rapo.kcao_sdtrue
        freq_cc_devito[1:kcao_nom_nneg] = np.sum(fcc, axis=(-1, -2)) * (kcao_dx)**2
        # Negative frequency coefficients are complex conjugates of flipped positive coefficients.
        freq_cc_devito[kcao_nom_nneg+ssna:] = np.flipud(np.conj(freq_cc_devito[1:kcao_nom_nneg]))
        # constant factor
        ft_fac = kcao_dom/(2*np.pi)*kcao_nom
        freq_cc_devito *= ft_fac
        # convert to time domain
        cc_devito = np.fft.fftshift(np.fft.ifft(freq_cc_devito, axis=0).real)
        # cc_devito = np.fft.ifftshift(np.fft.irfft(freq_cc_devito))

        # Compare
        plt.figure()
        # plt.plot(t_obs, cc_devito, label='Devito')
        # plt.plot(t_obs, cc_obs[:,ixa,ixb],'r', label='Analytical')
        plt.plot(kcao_t, cc_devito, label='this code (devito)')
        plt.plot(kcao_t, rapo.kcao_obscross[:,ixa,ixb],'r', label='pickle (anseicca)')
        plt.legend()
        # plt.title('without scaling')

        # plt.figure()
        # # plt.plot(t_obs, cc_devito/np.max(cc_devito), label='Devito')
        # # plt.plot(t_obs, cc_obs[:,ixa,ixb]/np.max(cc_obs[:,ixa,ixb]),'r', label='Analytical')
        # plt.plot(kcao_t, cc_devito/np.max(cc_devito), label='this code (devito)')
        # plt.plot(kcao_t, rapo.kcao_obscross[:,ixa,ixb]/np.max(rapo.kcao_obscross[:,ixa,ixb]),'r', label='pickle (anseicca)')
        # plt.legend()
        # plt.title('with scaling')

        print("End of code.")

    plt.show()
