import sys
import numpy as np

# import config_file as config

if not __name__ == '__main__':
# get essential variables from main (calling) program
    try:
        dom_geom = sys.modules['__main__'].dg
    except AttributeError:
        dom_geom = sys.modules['__main__'].cg_dom_geom

# global variables
dxy = dom_geom.dx
ngp_mb = dom_geom.ngp_box
ngp = {2: ngp_mb, 3: 2*ngp_mb-1}
xall = {2: dom_geom.X, 3: dom_geom.X2}
yall = {2: dom_geom.Y, 3: dom_geom.Y2}

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

###############################################################################################################

class somod:

    # ringg -> ring_of_gaussians (used by Datta_et_al, 2019)
    # rgring -> radially_gaussian_ring (used by Hanasoge, 2013)

    @staticmethod
    def mult_gauss(rcen, specs, fac_rel=2):

        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        xp=rcen[0]
        yp=rcen[1]
        ind = specs['r0'].index((xp,yp))
        xw = specs['w'][ind][0]
        yw = specs['w'][ind][1]

        print("From mult_gauss: ", xp)
        print("From mult_gauss: ", yp)

        for j in range(ngp[fac_rel]):
            ans[:,j] = np.exp( -( (xall[fac_rel][j] - xp)**2/(xw*(dxy**2)) + (yall[fac_rel] - yp)**2/(yw*(dxy**2)) ) )

        return ans

    #******************************************************************************************

    @staticmethod
    def ringg(theta, specs, fac_rel=2):
        ans=np.zeros((ngp[fac_rel],ngp[fac_rel]))
        rad=specs['r']
        sigma_fac=specs['w']
        rad_use = {2: rad, 3: 2*rad}

        for j in range(ngp[fac_rel]):
        	x0 = rad_use[fac_rel]*np.cos(theta)
        	y0 = rad_use[fac_rel]*np.sin(theta)
        	ans[:,j] = np.exp( -((xall[fac_rel][j] - x0)**2 + (yall[fac_rel] - y0)**2)/(sigma_fac*(dxy**2)) )

        return ans

    #******************************************************************************************

    @staticmethod
    def gcover(dx,xall,yall):
        ngp = xall.size
        assert ngp == yall.size

        ans=np.zeros((ngp,ngp))

        for j in range(ngp):
        	ans[:,j] = np.exp( -((xall[j] - x0)**2 + (yall - y0)**2)/(sigma_fac*(dx**2)) )

        return ans

    #******************************************************************************************

    @staticmethod
    def rgring(ngp,dx,xall,yall,rad,mag1,mag2=None):
        ans=np.zeros((ngp,ngp))
        for j in range(ngp):
            r_ib = np.sqrt(xall[j]**2 + yall**2)
            if mag2 is None:
            	ampl = mag1
            else:
            	#if abs(xall[j])<10:
            	if xall[j]>-22 and xall[j]<-15:
            		ampl = mag2
            	else:
            		ampl = mag1
            ans[:,j] = ampl * ( np.exp( -(r_ib-rad)**2/(10*(dxy**2))) )

        return ans

##########################################################################################
