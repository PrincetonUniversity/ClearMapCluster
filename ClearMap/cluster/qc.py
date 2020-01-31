#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:05:53 2018

@author: tpisano
"""
import skvideo.io
import skimage
import numpy as np, os
from skimage.external import tifffile
from skimage.exposure import adjust_gamma
from tools.utils.io import makedir, listdirfull

if __name__ == '__main__':
    src = '/home/wanglab/wang/seagravesk/lightsheet/201710_cfos/f37106_mouse2'
    dst = '/home/wanglab/wang/seagravesk/lightsheet/201710_cfos/registration_qc'
    make_overlay(src, dst)
    
    
    dst = '/home/wanglab/wang/seagravesk/lightsheet/201710_cfos/registration_qc'
    for src in listdirfull('/home/wanglab/wang/seagravesk/lightsheet/201710_cfos/'):    
        try:
            print(src)
            make_overlay(src, dst)
        except Exception as e:
            print(e)
            
    
    dst = '/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/cell_detection_qc'
    for src in listdirfull('/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/bkgd5_cell105_v2'):    
        try:
            print(src)
            cell_detection_overlay(src, dst)
        except Exception as e:
            print(e)
            
    src='/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/bkgd5_cell105_v2/20171129_ymaze_cfos20'
    im = show_plane(src, plane='400')
    sitk.Show(sitk.GetImageFromArray(im))
            
        
    

def make_overlay(src, dst, outdepth='uint16'):
    '''Simple Function to make overlays to test quality of registration
    '''

    
    #auto
    auto = tifffile.imread(os.path.join(src, 'clearmap_cluster_output', 'autofluo_resampled.tif'))
    auto = auto*1.0
    auto = skimage.exposure.adjust_gamma(auto, gamma=.6, gain=1)
    
    #autoreg
    autoreg = tifffile.imread(os.path.join(src, 'clearmap_cluster_output/elastix_auto_to_atlas/result.1.tif'))
    autoreg = autoreg*1.0
    #autoreg = skimage.exposure.adjust_gamma(skimage.exposure.rescale_intensity(autoreg), gamma=.6, gain=2)
    
    #combine
    z,y,x=auto.shape
    im = np.zeros((z,y,x,3))
    im[:,:,:,0] = autoreg
    im[:,:,:,1] = auto 
    
    #save out
    makedir(dst)
    out = os.path.join(dst, os.path.basename(src)+'_auto_registration.avi')
    
    skvideo.io.vwrite(out, im.astype(np.uint8)) #
    
    
    return

def cell_detection_overlay(src, dst, outdepth='uint16', atlas = False):
    
    #find cells
    cells = os.path.join(src, 'clearmap_cluster_output', 'cells.npy')
    cells_allpoints = os.path.join(src, 'clearmap_cluster_output', 'cells-allpoints.npy')
    cells_transformed = np.load(os.path.join(src, 'clearmap_cluster_output', 'cells_transformed_to_Atlas.npy'))
    
    #auto
    if not atlas: atlas = '/jukebox/wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'
    auto = tifffile.imread(atlas)
    auto = auto*1.0
    auto = adjust_gamma(auto, gamma=.6, gain=3)
    
    #combine
    z,y,x=auto.shape
    im = np.zeros((z,y,x,3))
    nonmap=[]
    for x,y,z in cells_transformed:
        try:
            im[z,y,x,0] = 65000
        except:
            nonmap.append((x,y,z))
    nonmap = np.asarray(nonmap)
    im[:,:,:,1] = auto
    
    #save out
    makedir(dst)
    out = os.path.join(dst, os.path.basename(src)+'_cells')
    tifffile.imsave(out+'.tif', im.astype(np.int8), compress=1)    
    #skvideo.io.vwrite(out+'.mp4', im.astype(np.uint8), backend='ffmpeg') #
    return

def show_plane(src, plane='400'):
    '''
    '''
    from ClearMap.cluster.utils import load_kwargs    
    kwargs = load_kwargs(src)
    vol = [xx for xx in kwargs['volumes'] if xx.ch_type =='cellch'][0]
    
    #find cells
    cells = np.load(os.path.join(src, 'clearmap_cluster_output', 'cells.npy'))
    arr = cells[cells[:,2]==int(plane)]
    im = tifffile.imread([xx for xx in listdirfull(vol.full_sizedatafld_vol) if 'Z{}'.format(str(plane).zfill(4)) in xx][0])
    
    #combine
    imm = np.zeros_like(im)
    for x,y,z in arr:
        imm[y,x]=55000
    
    out = np.zeros((3, im.shape[0], im.shape[1]))
    out[0]=imm
    out[1]=im
    return out
    