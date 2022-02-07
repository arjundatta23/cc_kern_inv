#!/usr/bin/python

# Standard modules
import os

##########################################################################################################################

class process_args:

    def __init__(self, cmd_line_args):

        if len(cmd_line_args)>1:
            self.scalar=False
            self.elastic=True
            inpdir=cmd_line_args[1]
            try:
                assert os.path.isdir(inpdir)
            except AssertionError:
                raise SystemExit("Problem with code input: must be a directory containing disp/eigen files")

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

            self.mod1dfile=fullpath(mod_files[0])
            self.egn_ray=fullpath([ f for f in egn_files if f.endswith(".ray") ][0]) # eigenfunctions Rayleigh
            self.disp_ray=fullpath([ f for f in disp_files if f.endswith(".ray") ][0]) # dispersion Rayleigh
            if len(egn_files)==2:
                self.egn_lov=fullpath([ f for f in egn_files if f.endswith(".lov") ][0]) # eigenfunctions Love
                self.disp_lov=fullpath([ f for f in disp_files if f.endswith(".lov") ][0]) # dispersion Love
        else:
            # no input argument provided
            self.scalar=True
            self.elastic=False

##########################################################################################################################
