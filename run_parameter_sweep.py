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

#set home directtory path
systemdirectory = "/home/wanglab"
###set paths to data
###inputdictionary stucture: key=pathtodata value=list["xx", "##"] where xx=regch, injch, or cellch and ##=two digit channel number
#"regch" = channel to be used for registration, assumption is all other channels are signal
#"cellch" = channel(s) to apply cell detection
#"injch" = channels(s) to quantify injection site
#"##" = when taking a multi channel scan following regexpression, the channel corresponding to the reg/cell/inj channel. I.e. name_of_scan_channel00_Z#### then use "00"
#e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"], ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary={
os.path.join(systemdirectory,"LightSheetData/lightserv/jverpeut/ymazecfos_learning_verpeut/ymazecfos_learning_verpeut-008/imaging_request_1/rawdata/resolution_1.3x/200924_072420_jv_ymazelearn_an8_1_3x_488_008na_1hfds_z10um_50msec_16-50-15"): [["regch", "00"]],
os.path.join(systemdirectory,"LightSheetData/lightserv/jverpeut/ymazecfos_learning_verpeut/ymazecfos_learning_verpeut-008/imaging_request_1/rawdata/resolution_1.3x/200924_072420_jv_ymazelearn_an8_1_3x_647_008na_1hfds_z10um_50msec_16-42-44"): [["cellch", "00"]]
}

####Required inputs
params={
"inputdictionary": inputdictionary, #don"t need to touch
"outputdirectory": os.path.join(systemdirectory,"wang/Jess/lightsheet_output/202010_cfos/parameter_sweep/an8"),
"resample" : False, #False/None, float(e.g: 0.4), amount to resize by: >1 means increase size, <1 means decrease
"xyz_scale": (5.0, 5.0, 10.0), #micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
"tiling_overlap": 0.00, #percent overlap taken during tiling
"blendtype" : "sigmoidal", #False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
"intensitycorrection" : False, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
"rawdata" : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
"slurmjobfactor": 50, #number of array iterations per arrayjob since max job array on SPOCK is 1000
}

def sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r, fEMP_size_r, fEMP_threshold_r,
                                     fIP_method_r, fIP_size_r, dCSP_threshold_r,thresholds_rows,
                                     tick, optimization_chunk=4, pth=False, cleanup=True,
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
    #set value in range of parameters based on jobid
    rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row=[(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row) for rBP_size, fEMP_hmax,
        fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r,
        fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r,thresholds_rows)][jobid]
    #run
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
        #run
        celldetection_operations(optimization_chunk, testing=True, **kwargs)
        #make cells detected array
        dct = pth_update(set_parameters_for_clearmap(testing=True, **nkwargs))
        #threshold and export
        #NOTE: DON'T USE THE CLEARMAP 'READPOINTS' FUNCTION, WEIRD CUT OFF RESULTS
        points = pickle.load(open(os.path.join(nkwargs["outputdirectory"],"cells",os.listdir(os.path.join(nkwargs["outputdirectory"],"cells"))[0]),"rb"))[0][0]
        intensities = pickle.load(open(os.path.join(nkwargs["outputdirectory"],"cells",os.listdir(os.path.join(nkwargs["outputdirectory"],"cells"))[0]),"rb"))[0][1]
        #Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
        #row = (0,0) : peak intensity from the raw data
        #row = (1,1) : peak intensity from the DoG filtered data
        #row = (2,2) : peak intensity from the background subtracted data
        #row = (3,3) : voxel size from the watershed
        points, intensities = thresholdPoints(points, intensities, threshold = thres_row[0], 
                            row = thres_row[1])
        #change dst to match parameters sweeped
        dst = (os.path.join(out0, 
        "cells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.npy".format(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
        os.path.join(out0, 
        "intensities_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.npy".format(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])))
        #export
        io.writePoints(dst, (points, intensities));
        #visualize
        raw = [xx for xx in listdirfull(out0+"/optimization/raw") if "tif" in xx]; raw.sort();
        raw_im = np.squeeze(tifffile.imread(raw))
        #make cell center map
        cellmap=np.zeros(raw_im.shape)
        cells=np.load(dst[0])
        for cell in cells:
            try: #make suure cell is in boundary of volume
                cellmap[cell[2]-1,cell[1]-1:cell[1]+1,cell[0]-1:cell[0]+1]=1 #cell format is xyz, but numpy is zyx
                #cell array number starts with 1, numpy starts with 00
            except Exception as e:
                pass
        #make stack
        rbg = np.stack([raw_im.astype("uint16"), cellmap.astype("uint16"), np.zeros(raw_im.shape)], -1)
        #export
        tifffile.imsave(os.path.join(out0, 
        "thresholdedcells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.tif".format(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
        rbg,compress=6)           
        #export non-thresholded cells
        cell = [xx for xx in listdirfull(out0+"/optimization/cell") if "tif" in xx]; raw.sort();
        cell_im = np.squeeze(tifffile.imread(cell))
        rbg = np.stack([raw_im.astype("uint16"), cell_im.astype("uint16"), np.zeros(raw_im.shape)], -1)
        tifffile.imsave(os.path.join(out0, 
        "nothresholdcells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.tif".format(rBP_size,
        fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
        rbg,compress=6) 
        print("\n           finished step thresholding step and exported overlay \n")
        if cleanup:
            #move overlays to main parameter folder
            shutil.move(os.path.join(out0, 
            "thresholdedcells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.tif".format(rBP_size,
            fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
                        out)
            shutil.move(os.path.join(out0, 
            "nothresholdcells_rBPsize{}_fEMPhmax{}_fEMPsize{}_fEMPthres{}_fIPmethod{}_fIPsize{}_dCSPthreshold{}_thres{}row{}.tif".format(rBP_size,
            fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold,thres_row[0][0],thres_row[1][0])),
                        out)
            shutil.rmtree(os.path.join(out0)) #delete job folder
            
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
        rBP_size_r = [3] ###evens seem to not be good  #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
        fEMP_hmax_r = [None]# (float or None) h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
        fEMP_size_r = [0] # size in pixels (x,y) for the structure element of the morphological opening
        fEMP_threshold_r = [None] #range(0,10)
        fIP_method_r = ["Max"] #["Max, "Mean"]
        fIP_size_r = [3]
        dCSP_threshold_r = [100]
        thresholds_rows = [[(120,10000), (2,2)], [(130,10000), (2,2)]]#intensity range]#cell shape range
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        # calculate number of iterations
        tick = 0
        for rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold, thres_row in product(rBP_size_r,fEMP_hmax_r, fEMP_size_r,
            fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r, thresholds_rows):
            tick +=1
        sys.stdout.write("\n\nNumber of iterations is {}:".format(tick))
        #run
        for jobid in range(0,tick):#temp
            #iterate through combination of parameters
            try:
                sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r, 
                                         fEMP_size_r, fEMP_threshold_r, fIP_method_r, 
                                         fIP_size_r, dCSP_threshold_r, thresholds_rows, tick, optimization_chunk=20,
                                         cleanup=True, **params)
            except Exception as e:
                print("Jobid {}, Error given {}".format(jobid, e))

    #end server
    vdisplay.stop()
