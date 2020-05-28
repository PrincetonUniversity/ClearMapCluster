#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:41:01 2018

@author: tpisano
"""

import os, sys, shutil, pickle, tifffile, numpy as np
from itertools import product
from xvfbwrapper import Xvfb; vdisplay = Xvfb(); vdisplay.start()
from ClearMap.cluster.preprocessing import updateparams, listdirfull, arrayjob, makedir, removedir, pth_update
from ClearMap.cluster.par_tools import celldetection_operations,join_results_from_cluster_helper
from ClearMap.cluster.utils import load_kwargs
from ClearMap.parameter_file import set_parameters_for_clearmap
import ClearMap.IO as io
from ClearMap.Analysis.Statistics import thresholdPoints

systemdirectory = "/home/wanglab"
###set paths to data
###inputdictionary stucture: key=pathtodata value=list["xx", "##"] where xx=regch, injch, or cellch and ##=two digit channel number
#"regch" = channel to be used for registration, assumption is all other channels are signal
#"cellch" = channel(s) to apply cell detection
#"injch" = channels(s) to quantify injection site
#"##" = when taking a multi channel scan following regexpression, the channel corresponding to the reg/cell/inj channel. I.e. name_of_scan_channel00_Z#### then use "00"
#e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"], ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary={
os.path.join(systemdirectory, "LightSheetTransfer/brody/z268"): [["regch", "00"], ["cellch", "01"]]
}

####Required inputs
params={
"inputdictionary": inputdictionary, #don"t need to touch
"outputdirectory": os.path.join(systemdirectory, "Desktop/z268"),
"xyz_scale": (1.63, 1.63, 10.0), #micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
"tiling_overlap": 0.00, #percent overlap taken during tiling
"blendtype" : "sigmoidal", #False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
"intensitycorrection" : False, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
"rawdata" : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
"slurmjobfactor": 50, #number of array iterations per arrayjob since max job array on SPOCK is 1000
}

def sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r, fEMP_size_r, fEMP_threshold_r,
                                     fIP_method_r, fIP_size_r, dCSP_threshold_r,thresholds_rows,
                                     tick, optimization_chunk=4, pth=False, cleanup=True, save=False,
                                     **kwargs):
    """Function to sweep parameters

    final outputs will be saved in outputdirectory/parameter_sweep
    second copy will be saved in outputdirectory/parameter_sweep_jobid if cleanup=False

    Inputs:
        ----------------
        jobid: chunk of tissue to run (usually int between 20-30)
        #pth (optional): if pth to output folder after running package, function will load the param file automatically
        rescale (optional): str of dtype to rescale to. E.g.: "uint8"
        cleanup = T/F removes subfolders after
        optimization_chunk = this was the old "jobid" in this case it is the chunk of volume to look at
        kwargs (if not pth): "params" from run_clearmap_cluster.py
    """

    #make folder for final output:
    opt = kwargs["outputdirectory"]; makedir(opt)
    out = opt+"/parameter_sweep"; makedir(out)
    out0 = opt+"/parameter_sweep_jobid_{}".format(str(jobid).zfill(4)); makedir(out0)

    rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row=[(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row) for rBP_size, fEMP_hmax,
        fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r,
        fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r,thresholds_rows)][jobid]

    pth = out0+"/rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}tow{}.tif".format(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])

    if not os.path.exists(pth):
        try:
            #set params for sweep
            kwargs["removeBackgroundParameter_size"] = (rBP_size,rBP_size) #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
            kwargs["findExtendedMaximaParameter_hmax"] = fEMP_hmax # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
            kwargs["findExtendedMaximaParameter_size"] = fEMP_size # size in pixels (x,y) for the structure element of the morphological opening
            kwargs["findExtendedMaximaParameter_threshold"] = fEMP_threshold # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
            kwargs["findIntensityParameter_method"] =  fIP_method # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
            kwargs["findIntensityParameter_size"] = (fIP_size,fIP_size,fIP_size) # (tuple)             size of the search box on which to perform the *method*
            kwargs["detectCellShapeParameter_threshold"] = dCSP_threshold # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated

            #tmp
            nkwargs = load_kwargs(kwargs["outputdirectory"])
            kwargs["outputdirectory"] = out0
            nkwargs.update(kwargs)
            pckloc=out0+"/param_dict.p"; pckfl=open(pckloc, "wb"); pickle.dump(nkwargs, pckfl); pckfl.close()

            #run cell detection
            sys.stdout.write("\n\n\n           *****Iteration {} of {}*****\n\n\n".format(jobid, tick))
            sys.stdout.write("    Iteration parameters: {}     {}     {}     {}     {}     {}     {}".format(kwargs["removeBackgroundParameter_size"], 
            kwargs["findExtendedMaximaParameter_hmax"], kwargs["findExtendedMaximaParameter_size"], 
            kwargs["findExtendedMaximaParameter_threshold"], kwargs["findIntensityParameter_method"],
            kwargs["findIntensityParameter_size"], kwargs["detectCellShapeParameter_threshold"]))
            
            celldetection_operations(optimization_chunk, testing=True, **kwargs)

            if save:
                #list, load, and maxip
                raw = [xx for xx in listdirfull(out0+"/optimization/raw") if "~" not in xx and ".db" not in xx]; raw.sort();
                raw_im = np.squeeze(tifffile.imread(raw))
                raw_mx = np.max(raw_im, axis = 0)
                bkg = [xx for xx in listdirfull(out0+"/optimization/background") if "~" not in xx and "Thumbs.db" not in xx]; bkg.sort()
                bkg_im = tifffile.imread(bkg)
                bkg_mx = np.max(bkg_im, axis = 0)
                cell = [xx for xx in listdirfull(out0+"/optimization/cell") if "~" not in xx and ".db" not in xx]; cell.sort()
                cell_im = tifffile.imread(cell)
                cell_mx = np.max(cell_im, axis = 0)
    
                #concatenate and save out:
                bigim = np.concatenate((raw_mx, bkg_mx, cell_mx), axis=1)
                del bkg, bkg_im, bkg_mx, cell, cell_im,cell_mx
                if cleanup: removedir(out0)
                if not cleanup: tifffile.imsave(pth, bigim, compress=1)
    
                #save in main
                npth = out0+"/rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.tif".format(rBP_size,
                            fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])
                tifffile.imsave(npth, bigim.astype("uint16"), compress = 1)

            #make cells detected array
            dct = pth_update(set_parameters_for_clearmap(testing=True, **params))
            out = join_results_from_cluster_helper(**dct["ImageProcessingParameter"])    
            #threshold and export
            points, intensities = io.readPoints(dct["ImageProcessingParameter"]["sink"])
            #Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
            #row = (0,0) : peak intensity from the raw data
            #row = (1,1) : peak intensity from the DoG filtered data
            #row = (2,2) : peak intensity from the background subtracted data
            #row = (3,3) : voxel size from the watershed
            points, intensities = thresholdPoints(points, intensities, threshold = thres_row[0], 
                                row = thres_row[1])
            #change dst to match parameters sweeped
            dst = (os.path.join(out, 
            "cells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.npy".format(rBP_size,
            fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
                   
            os.path.join(out, 
            "intensities_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.npy".format(rBP_size,
            fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])))
            
            io.writePoints(dst, (points, intensities));
            print("\n           finished step thresholding step \n")
        
        except Exception as e:
            print("Error on: {}\n\nerror={}".format(pth,e))
            with open(os.path.join(out, "errored_files.txt"), "a") as fl:
                fl.write("\n\n{}\n{}\n".format(pth, kwargs))
                fl.close

    return
#%%
if __name__ == "__main__":
    #parallelized for cluster
    print(sys.argv)
    stepid = int(sys.argv[1])

    #run step 1 to populate fullsizedata folder
    if stepid == 0:
        #make output folder
        makedir(params["outputdirectory"])
        ###make parameter dictionary and pickle file:
        updateparams(os.getcwd(), **params) # e.g. single job assuming directory_determiner function has been properly set
        #copy folder into output for records
        if not os.path.exists(os.path.join(params["outputdirectory"], "ClearMapCluster")): 
            shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "ClearMapCluster"), 
            ignore=shutil.ignore_patterns("^.git")) #copy run folder into output to save run info
        #make planes
        for stepid in range(0, 30):
            arrayjob(stepid, cores=12, compression=1, **params)

    #run paramter sweep on full resolution data
    if stepid == 1:
        #get array ID
        print(os.environ["SLURM_ARRAY_TASK_ID"])
        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"]) #int(sys.argv[2])
        ######################################################################################################
        #NOTE: To adjust parameter sweep, modify ranges below
        ######################################################################################################
        rBP_size_r = [3,5,7] ###evens seem to not be good  #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
        fEMP_hmax_r = [None]# (float or None) h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
        fEMP_size_r = [0] # size in pixels (x,y) for the structure element of the morphological opening
        fEMP_threshold_r = [None] #range(0,10)
        fIP_method_r = ["Max"] #["Max, "Mean"]
        fIP_size_r = [3]
        dCSP_threshold_r = [100,300,500,700]
        thresholds_rows = [[(500, 10000), (2,2)], [(1500, 10000), (2,2)], [(20,900), (3,3)]]
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        # calculate number of iterations
        tick = 0
        for rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold, thres_row in product(rBP_size_r,fEMP_hmax_r, fEMP_size_r,
            fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r, thresholds_rows):
            tick +=1
        sys.stdout.write("\n\nNumber of iterations is {}:".format(tick))

        #iterate through combination of parameters
        try:
            sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r, 
                                     fEMP_size_r, fEMP_threshold_r, fIP_method_r, 
                                     fIP_size_r, dCSP_threshold_r, thresholds_rows, tick, optimization_chunk=20,
                                     cleanup = False, **params)
        except Exception as e:
            print("Jobid {}, Error given {}".format(jobid, e))

    #end server
    vdisplay.stop()
