#!/usr/bin/python

# Standard modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as clt

# Custom modules
try:
    import config_classical as config
    # classical kernels code
    print("Imported config_classical\n")
except ImportError:
    import config_file as config
    # cross-correlation kernels code
    print("Imported config_file\n")

# get required variables from main (calling) program
# recA = sys.modules['__main__'].loc_A
# recB = sys.modules['__main__'].loc_B
scal_eq = sys.modules['__main__'].scalar
try:
    Z = sys.modules['__main__'].dep_pts_intgrn
    # calling program is the computational code and kernels are computed
except AttributeError:
    try:
        Z = sys.modules['__main__'].Z
        # calling program is the plotting script
    except AttributeError:
        # calling program is the computational code (through "utils_post") but kernels NOT computed
        pass

try:
    vgx, vgzx = np.meshgrid(config.X,Z)
    vgy, vgzy = np.meshgrid(config.Y,Z)
except NameError:
    pass

# global variables
recA = [-config.rloc,0]
recB = [config.rloc,0]

########################################################################################################################

def show_1D_profiles(sections_2D, profile_along, samples_along, lsam_locs, lim_pa=None):

    """ given any 2-D section, i.e. x-z, y-z or x-y, this function extracts and plots 1-D profiles from it at any number of
        specified locations. And it can do this for multiple 2-D sections.
    """

    coord_grid = {'x': config.X, 'y': config.Y, 'z': Z}

    cgp = coord_grid[profile_along]
    cgs = coord_grid[samples_along]

    # 'l' in below variable names stands for "lateral", i.e. one of the two horizontal directions
    lsam_ind = np.searchsorted(cgs,lsam_locs)
    lsam_sec = range(len(sections_2D))
    for es, sec2D in enumerate(sections_2D):
        lsam_sec[es] = sec2D[:,lsam_ind]

    if len(lsam_locs)>1:
    # line collection plot
        colours=['b', 'r', 'g']

        col_actual = range(len(sections_2D))
        # col is for collection-object

        # determine appropriate scaling for collection plot
        step = lsam_locs[1] - lsam_locs[0]
        mv = np.max(lsam_sec[0])
        pow = np.floor(np.log10(step/mv))
        scaling = (10**(pow)) * 2
        # exact scaling appropriate for plotting is subjective and can be chosen by hit-and-trial
        # on a case-by-case basis

        dummies = np.zeros(len(cgp))
        loclist=list()
        act_list=list()
        dum_list=list()
        for j in range(len(sections_2D)):
            for s in range(len(lsam_locs)):
                if j==0:
                    loclist.append( (lsam_locs[s],0) )
                    dum_list.append( zip(dummies,cgp) )
                profile_1D = scaling * lsam_sec[j][:,s]
                act_list.append( zip(profile_1D,cgp) )
            col_actual[j] = clt.LineCollection(act_list, offsets=loclist)#, color=colours[j])

        col_zeros = clt.LineCollection(dum_list, offsets=loclist, alpha=0.5, linestyle='--')
        fig2=plt.figure(figsize=(12,5))
        ax2=fig2.add_subplot(111)
        ax2.add_collection(col_zeros)
        for j in range(len(sections_2D)):
            ax2.add_collection(col_actual[j])
        ax2.set_xlim([lsam_locs[0]-step,lsam_locs[-1]+step])
    else:
    # ordinary plot
        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        for j in range(len(sections_2D)):
            ax2.plot(lsam_sec[j],cgp)
            # ax2.plot(cgp,lsam_sec[j])

    if lim_pa != None:
        ax2.set_ylim(lim_pa,0)
    else:
        ax2.set_ylim(cgp[0],cgp[-1])

##########################################################################################################################

def plot_section_vertical_inline(arrin, yloc):

    """ vertical sections along or parallel to the receiver-pair line """

    yindex = np.searchsorted(config.Y, yloc)
    try:
        yinfo = "Y-position: %.1f" %(config.Y[yindex])
    except IndexError:
        print("Failed to plot vertical section; invalid y-position specified.")
        return None

    section_xz = arrin[:,yindex,:]
    zmax_show = config.dmax
    xstart=-200; xend=200; step=25
    xsam_locs = range(xstart,xend+step,step)
    # show_1D_profiles([section_xz], 'z', 'x', xsam_locs, zmax_show)

    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    cax1=ax1.pcolor(vgx, vgzx, section_xz.real)
    ax1.plot(recA[0], 0, 'wd', markerfacecolor="None")
    ax1.plot(recB[0], 0, 'wd', markerfacecolor="None")
    # ax1.set_ylim(Z[-1],0)
    ax1.set_ylim(zmax_show,0)
    ax1.set_xlabel("Horizontal distance 'x' [km]")
    ax1.set_ylabel("Depth [km]")
    ax1.set_title(yinfo)
    plt.colorbar(cax1,ax=ax1)

##########################################################################################################################

def plot_section_vertical_crossline(arrin, xloc):

    """ vertical sections transverse to the receiver-pair line """

    xindex = np.searchsorted(config.X, xloc)
    try:
        xinfo = "X-position: %.1f" %(config.X[xindex])
    except IndexError:
        print("Failed to plot vertical section; invalid x-position specified.")
        return None
    section_yz = arrin[:,:,xindex]
    zmax_show = config.dmax
    ystart=-200; yend=200; step=25
    ysam_locs = range(ystart,yend+step,step)
    show_1D_profiles([section_yz], 'z', 'y', ysam_locs, zmax_show)

    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    cax1=ax1.pcolor(vgy, vgzy, section_yz.real)
    # ax1.set_ylim(Z[-1],0)
    ax1.set_ylim(zmax_show,0)
    ax1.set_xlabel("Horizontal distance 'y' [km]")
    ax1.set_ylabel("Depth [km]")
    ax1.set_title(xinfo)
    plt.colorbar(cax1,ax=ax1)

##########################################################################################################################

def plot_section_horizontal(inplist, depth, heading, ptype):

    zindex = np.searchsorted(Z,depth)
    zinfo = " (depth %.2f km)" %(Z[zindex])
    mheading = heading + zinfo

    try:
        plotlist = [ il[zindex,...] for il in inplist ]
    except IndexError:
        print("Failed to plot horizontal section; invalid depth specified.")
        return None

    minvt = [np.amin(plotlist[il]) for il in range(len(inplist))]
    maxvt = [np.amax(plotlist[il]) for il in range(len(inplist))]
    # print(minvt)
    # print(maxvt)
    imin = np.argmin(minvt)
    imax = np.argmax(maxvt)
    min_val = np.sort(np.array(plotlist[imin]).flatten())[1]  # picking the second-lowest value
    max_val = np.sort(np.array(plotlist[imax]).flatten())[-2] # picking the second-highest value
    # print(min_val)
    # print(max_val)

    if ptype==1:
        pname = ['First contribution','Second contribution','Complete kernel']
    elif ptype==2:
        if scal_eq:
            pname = [r'$K^{\prime}_{\rho}$',r'$K_c$']
        else:
            pname = [r'$K^{\prime}_{\rho}$',r'$K_{\alpha}$',r'$K_{\beta}$']
    elif ptype==3:
        if scal_eq:
            pname = [r'$K_{\rho}$',r'$K_{\mu}$',r'$K^{\prime}_{\rho}$']
        else:
            pname = [r'$K_{\rho}$',r'$K_{\lambda}$',r'$K_{\mu}$',r'$K^{\prime}_{\rho}$']

    nplots=len(inplist) #3
    if nplots>1:
        fig, (ax) = plt.subplots(1,nplots,sharex=True,sharey=True,figsize=(4*nplots,3))
    else:
        fig=plt.figure()
        ax_use=fig.add_subplot(111)
    for p in range(nplots):
        if nplots>1:
            ax_use = ax[p]
        cax=ax_use.pcolor(config.gx, config.gy, plotlist[p], vmin=min_val, vmax=max_val, cmap=plt.cm.jet)
        # cax=ax_use.pcolor(config.gx, config.gy, plotlist[p], vmin=-2e-5, vmax=4e-5, cmap=plt.cm.jet)
        # ax_use.imshow(plotlist[p], cmap=plt.cm.jet)
        ax_use.plot(recA[0], recA[1], 'wd', markerfacecolor="None")
        ax_use.plot(recB[0], recB[1], 'wd', markerfacecolor="None")
        ax_use.set_title(pname[p]) #, pad=15)
        plt.colorbar(cax, ax=ax_use, format='%.1e')
    # fig.suptitle(mheading)

    # fig2=plt.figure()
    # ax2=fig2.add_subplot(111)
    # cax2=ax2.pcolor(config.gx,config.gy,plotlist[-1])#,vmin=min_val, vmax=max_val, cmap=plt.cm.jet)
    # ax2.plot(recA[0], recA[1], 'wd', markerfacecolor="None")
    # ax2.plot(recB[0], recB[1], 'wd', markerfacecolor="None")
    # ax2.set_title(heading)
    # plt.colorbar(cax2,ax=ax2)

    xstart=-100; xend=100; step=50
    xsam_locs = [110] #range(xstart,xend+step,step)
    # show_1D_profiles(plotlist, 'y', 'x', xsam_locs)

##########################################################################################################################

def plot_source_spectrum():

	figs=plt.figure()
	axs=figs.add_subplot(111)
	axs.plot(np.fft.fftshift(config.fhz),np.fft.fftshift(config.pow_spec_sources),'-o')
	axs.set_xlabel('Frequency [Hz]')

##########################################################################################################################

# def plot_wforms_TD(cc, cc_der_TD=None, cc_der_FD=None):
def plot_wforms_TD(cc, **more_wforms):

    fig_as=plt.figure()
    axas=fig_as.add_subplot(111)
    axas.plot(config.tt,cc,label='$C$')

    if len(more_wforms)==1:
        # this would be the adjoint_source
        adjsrc = more_wforms['asrc']
        axas.plot(config.tt,adjsrc*1e-13,label='ASRC*1e-16')
    elif len(more_wforms)==2:
        # this would be Cdot (time derivative of cc wform), with differentiation
        # performed in the time and frequency domains
        cc_der_TD = more_wforms['cdot_dTD']
        cc_der_FD = more_wforms['cdot_dFD']
        # if isinstance(cc_der_TD,np.ndarray) and isinstance(cc_der_FD,np.ndarray):
        axas.plot(config.tt,cc_der_TD,label='$\dot{C}$ TD calculation')
        axas.plot(config.tt,cc_der_FD,label='$\dot{C}$ FD calculation')
    elif len(more_wforms)==0:
        pass

    axas.set_xlabel('Time [s]')
    axas.legend()

##########################################################################################################################

def plot_horizontal_outmost_domain(sample_GF, inx, iny):

    slocsx=np.array(inx)
    slocsy=np.array(iny)

    mirror_x = -slocsx
    mirror_y = -slocsy

    xlocs=list(zip(slocsx,mirror_x))
    ylocs=list(zip(slocsy,mirror_y))

    # convert from spatial coordinates to grid indices
    getind_x = lambda m: np.searchsorted(config.Xoutmost, m)
    getind_y = lambda m: np.searchsorted(config.Youtmost, m)

    xind = getind_x(xlocs)
    yind = getind_y(ylocs)

    figo=plt.figure()
    axo=figo.add_subplot(111)
    caxo=axo.pcolor(config.omgx,config.omgy,sample_GF.real)
    plt.colorbar(caxo, ax=axo)#, format='%.1e')

    for pair in range(xind.shape[0]):

        mirror_ind = [xind[pair,1],yind[pair,1]]
        print(mirror_ind)

        # boxstart = np.array(map(lambda m: int(m-(ofac*config.ngp_config.hlen)), mirror_ind))
        # boxend = boxstart + config.Xoutmost.size - 1
        boxstart = np.array(list(map(lambda m: m-config.ngp_hlen, mirror_ind)))
        boxend = boxstart + (2*config.ngp_hlen)

        print("Box start in gridpts: ", boxstart)
        print("Box end in gridpts: ", boxend)

        # finally convert to plot coordinates for plotting lines
        plot_coordinates = lambda m: float(m)/(config.Xoutmost.size-1)
        start_pc=list(map(plot_coordinates, boxstart))
        end_pc=list(map(plot_coordinates, boxend))
        #x_pc=map(plot_coordinates, [boxstart[0],boxend[0]])

        boxedge = {0:boxstart, 1:boxend}

        axo.plot(slocsx[pair], slocsy[pair], 'wd', markerfacecolor="None")
        axo.plot(mirror_x[pair], mirror_y[pair], 'wo', markerfacecolor="None")
        axo.plot(0,0,'ro')

        bspc_y = float(config.ofac-1)/(2*config.ofac)
        #bspc_x = float( (config.ofac-1)*config.ngp_config.hlen + ngp_rloc ) / (2*(config.ofac*config.ngp_config.hlen + ngp_rloc))
        bspc_x = float( (config.ofac-1)*config.ngp_hlen ) / (2*config.ofac*config.ngp_hlen)
        #bspc -> box_start_plot_coordinates

        axo.axvline(-config.hlen,ymin=bspc_y,ymax=1-bspc_y)
        axo.axvline(config.hlen,ymin=bspc_y,ymax=1-bspc_y)
        axo.axhline(-config.hlen,xmin=bspc_x,xmax=1-bspc_x)
        axo.axhline(config.hlen,xmin=bspc_x,xmax=1-bspc_x)

        for e in boxedge:
        	#axo.axvline(config.Xoutmost2[boxedge[e][0]],color='w',ls='--')
        	axo.axhline(config.Youtmost[boxedge[e][1]],xmin=start_pc[0],xmax=end_pc[0],ls='--')
        	axo.axvline(config.Xoutmost[boxedge[e][0]],ymin=start_pc[1],ymax=end_pc[1],ls='--')

####################################################################################################
