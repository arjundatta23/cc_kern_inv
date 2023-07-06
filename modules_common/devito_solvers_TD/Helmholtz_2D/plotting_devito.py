import numpy as np
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rc('font', size=16)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None




def plot_velocity(model, source=None, receiver=None, colorbar=True):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size =  np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0]*1e-3,
              model.origin[1] + domain_size[1]*1e-3, model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cm.jet,
                      vmin=np.min(field), vmax=np.max(field),
                      extent=extent)
    plt.xlabel('X position (m)')
    plt.ylabel('Depth (m)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(receiver[:, 0]*1e-3, receiver[:, 1]*1e-3,
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(source[:, 0]*1e-3, source[:, 1]*1e-3,
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0]*1e-3)
    plt.ylim(model.origin[1] + domain_size[1]*1e-3, model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()


def plot_shotrecord(rec, rec_coords, t0, tn, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    scale = np.max(rec) / 5.
    extent = [rec_coords[0,0]*1e-3, rec_coords[-1,0]*1e-3,
              tn*1e-3, t0*1e-3]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent, aspect='auto')
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


