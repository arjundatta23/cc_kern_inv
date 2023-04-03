#!/usr/bin/python

# Standard modules
import os

# Custom modules
import config_file as config
import read_velocity_models as u0

##########################################################################################################################

dg = config.dom_geom

##########################################################################################################################

class get_user_input:

    def __init__(self):

        if config.inv_mdlng_type.find('anal_elas_1D') != -1 or config.inv_mdlng_type.find('anal_elas_1D') != -1:
            indir=input("Directory containing 1-D elastic model files: ")
            try:
                assert os.path.isdir(indir)
                self.scalar=False
                self.elastic=True
            except AssertionError:
                raise SystemExit("Problem with code input: must be a directory containing disp/eigen files")
            else:
                self.elas_mod_1D={}
                self.nz = self.input_elastic_1D(indir)
        else:
            self.scalar=True
            self.elastic=False
            self.nz=1

        if config.inv_mdlng_type.find('num_scal_2D') != -1:

            input_vel_mod_syn = input("2-D velocity model to be used in INVERSION (enter 0 to use a homogeneous velocity model): ")
            if input_vel_mod_syn.isdigit():
                print("No velocity model file provided, using homogeneous velocity")
            else:
                self.acou_vel_mod_syn = self.input_acoustic_2D(input_vel_mod_syn)

        if (not config.ext_data and config.tru_mdlng_type.find('num_scal_2D') != -1):
            input_vel_mod_obs = input("2-D velocity model to be used to simulate TEST DATA (enter 0 for homogeneous/same as in inversion): ")
            if input_vel_mod_obs.isdigit():
                try:
                    self.acou_vel_mod_obs = self.acou_vel_mod_syn
                except AttributeError:
                    pass
            else:
                self.acou_vel_mod_obs = self.input_acoustic_2D(input_vel_mod_obs)

    # --------------------------------------------------------------------------

    def input_elastic_1D(self, inpdir):

        fullpath = lambda m: os.path.join(inpdir,m)
        mod_files=[ n for n in os.listdir(inpdir) if n.startswith("mod") ]
        egn_files=[ n for n in os.listdir(inpdir) if n.startswith("eigen.") ]
        disp_files=[ n for n in os.listdir(inpdir) if n.startswith("disp.") ]
        try:
            assert len(mod_files)==1
            assert len(egn_files)==len(disp_files)
            assert len(egn_files)>0 and len(egn_files)<3
        except AssertionError:
            raise SystemExit("Problem with code input: must contain files corresponding to one Earth model only")

        self.elas_mod_1D["mod1dfile"]=fullpath(mod_files[0])
        self.elas_mod_1D["egn_ray"]=fullpath([ f for f in egn_files if f.endswith(".ray") ][0]) # eigenfunctions Rayleigh
        self.elas_mod_1D["disp_ray"]=fullpath([ f for f in disp_files if f.endswith(".ray") ][0]) # dispersion Rayleigh
        if len(egn_files)==2:
            self.elas_mod_1D["egn_lov"]=fullpath([ f for f in egn_files if f.endswith(".lov") ][0]) # eigenfunctions Love
            self.elas_mod_1D["disp_lov"]=fullpath([ f for f in disp_files if f.endswith(".lov") ][0]) # dispersion Love

        # read input depth-dependent model and fix/extract necessary parameters
        upreo = u0.model_1D(self.elas_mod_1D["mod1dfile"])
        upreo.fix_max_depth(dg.zmax)

        self.elas_mod_1D["hif_mod"] = upreo.mod_hif
        self.elas_mod_1D["dep_pts_use"] = upreo.deps_tomax
        self.elas_mod_1D["dep_pts_mod"] = upreo.deps_all
        # hif_mod_use = hif_mod[hif_mod<=dg.zmax]
        # wspeed = ?? # NEED TO CHANGE, SHOULD BE BASED ON VELOCITIES FROM DISP FILE
        print("Layer interfaces in model: ", self.elas_mod_1D["hif_mod"], self.elas_mod_1D["hif_mod"].size)
        print("Depth points to be used in code: ", self.elas_mod_1D["dep_pts_use"], self.elas_mod_1D["dep_pts_use"].size)
        return self.elas_mod_1D["dep_pts_use"].size

    # --------------------------------------------------------------------------

    def input_acoustic_2D(self, input_vel_mod):

        vel_mod_file = os.path.expanduser(input_vel_mod)
        up2d = u0.model_2D(dg.box_len, dg.dx)
        # up2d.grd_file(vel_mod_file)
        up2d.npz_file(vel_mod_file)
        return up2d.vel_acou

##########################################################################################################################
