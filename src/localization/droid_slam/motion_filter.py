import cv2
import torch
import lietorch
import numpy as np

from collections import OrderedDict
from droid_net import DroidNet
from segmentor import SemSegNet

import geom.projective_ops as pops
from modules.corr import CorrBlock

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('src')

from DytanVO.Network.VONet import VONet
from DytanVO.Network.rigidmask.VCNplus import SegNet, WarpModule, flow_reg
from DytanVO.evaluator.transformation import se2SE
from DytanVO.Datasets.utils import make_intrinsics_layer, ToTensor, Compose, ResizeData, DownscaleFlow
from DytanVO.Datasets.cowmask import cow_masks

class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0",
                 filter_semantics=False, filter_motion=False):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        self.motion_seg = filter_motion
        self.semantic_seg = filter_semantics

        self.k = 0
        
        #### Motion and Semantic Segmentation ####
        if self.semantic_seg:
            self.segmentor = SemSegNet("DeepLabV3")
        if self.motion_seg:
            self.testres = 1.2
            maxw, maxh = [int(self.testres * 1024), int(self.testres * 448)]
            max_h = int(maxh // 64 * 64)
            max_w = int(maxw // 64 * 64)
            if max_h < maxh: max_h += 64
            if max_w < maxw: max_w += 64
            maxh = max_h
            maxw = max_w
            self.segnet = SegNet([1, maxw, maxh], md=[4, 4, 4, 4, 4], fac=1, exp_unc=True)
            segmodelname = 'models/' + "segnet-sf.pth"
            self.segnet = self.load_seg_model(self.segnet, segmodelname)
            
            self.segnet.to(device)
            self.segnet_initialize = False
            self.iter_num = 3
            self.sigmoid = lambda x: 1/(1 + np.exp(-x))

            self.vonet = VONet() 

            # load VO model separately (flow + pose) or at once
            modelname = 'models/flownet.pkl'
            self.load_vo_model(self.vonet.flowNet, modelname)
            modelname = 'models/posenet.pkl'
            self.load_vo_model(self.vonet.flowPoseNet, modelname)

            self.vonet.cuda()

            self.test_count = 0
            self.pose_norm = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
            self.flow_norm = 20 # scale factor for flow

            # To transform coordinates from NED to Blender
            Ry90 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
            Rx90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            self.camT = Rx90.dot(Ry90)

            self.resizedata = ResizeData((448, 640))

            self.seg_thresh = 0.95
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    def _segment(self, image):
        """ masks of persons """
        return self.segmentor(image[0], device=self.device)
    
    def _mask_motion(self, image, intrinsics):
        motion_mask = self.__get_motion_mask(image.cpu(), intrinsics)
        self.k += 1
        return motion_mask

    def __get_motion_mask(self, image, intrinsics):
        """ get motion mask for dynamics """
        H, W = image.shape[-2:]

        # set motion_mask to 0 for first step
        if self.k == 0:
            motion_mask = torch.zeros(1, H, W, device=self.device).bool()

        with torch.cuda.amp.autocast(enabled=False):
            # create motion mask
            if self.k > 0:
                torch.set_default_dtype(torch.float32)
                intrinsics_motion = [intrinsics[0].item(), intrinsics[2].item(), intrinsics[3].item(), 0.05]
                transform = Compose([ResizeData((448, 640)), DownscaleFlow(), ToTensor()])
                seg_thresh = self.seg_thresh

                img0_raw   = self.prev_image.squeeze(0).permute(1, 2, 0).numpy()
                img1_raw   = image.squeeze(0).permute(1, 2, 0).numpy()
                intrinsic = make_intrinsics_layer(img0_raw.shape[-2], img0_raw.shape[-3], intrinsics_motion[0], intrinsics_motion[1], intrinsics_motion[2], intrinsics_motion[3])

                res = {'img0': img0_raw, 'img1': img1_raw, 'intrinsic': intrinsic}

                res = transform(res)
                img0 = res['img0'].cuda().unsqueeze(0)
                img1 = res['img1'].cuda().unsqueeze(0)
                intrinsic = res['intrinsic'].cuda().unsqueeze(0)

                del res

                if not self.segnet_initialize:
                    self.vonet.eval()
                    self.segnet.eval()
                    self.initialize_segnet_input(img0_raw, intrinsics_motion)
                    self.segnet_initialize = True
                
                with torch.no_grad():
                    imgL_noaug, imgLR = self.transform_segnet_input(img0_raw, img1_raw)
                    flowdc = self.segnet.module.forward_VCN(imgLR)

                    flow_output, _ = self.vonet([img0, img1], only_flow=True)
                    
                    seg_thresholds = np.linspace(seg_thresh, 0.98, self.iter_num - 1)
                    for iter in range(self.iter_num):
                        flow = flow_output.clone()
                        if iter == 0:
                            if self.k < 2 or not torch.any(self.video.all_motion_masks[self.k-1]):
                                cow_sigma_range = (20, 60)
                                log_sigma_range = (np.log(cow_sigma_range[0]), np.log(cow_sigma_range[1]))
                                cow_prop_range = (0.3, 0.6)
                                segmask = cow_masks(flow.shape[-2:], log_sigma_range, cow_sigma_range[1], cow_prop_range).astype(np.float32)
                                segmask = segmask[None, None, ...]
                                segmask = torch.from_numpy(np.concatenate((segmask,) * img0.shape[0], axis=0)).cuda()
                            else: # if already motion masks computed, take these as initialization
                                segmask = self.video.all_motion_masks[self.k-1].float().unsqueeze(0)
                                segmask = F.interpolate(segmask, (flow.shape[-2:]), mode='bilinear')

                        _, pose_output = self.vonet([img0, img1, intrinsic, flow, segmask], only_pose=True)

                        # Do not pass segnet in the last iteration
                        if iter == self.iter_num - 1:
                            break

                        seg_thresh = seg_thresholds[iter] if iter < self.iter_num-1 else seg_thresh
                        pose_input = pose_output.data.cpu().detach().numpy().squeeze()
                        pose_input = pose_input * self.pose_norm
                        pose_input = self.camT.T.dot(se2SE(pose_input)).dot(self.camT)
                        
                        disc_aux = [self.intr_list, imgL_noaug, pose_input[:3,:]]               
                        fgmask, _ = self.segnet(imgLR, disc_aux, flowdc)
                        
                        fgmask = cv2.resize(fgmask.cpu().numpy(), (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                        fg_probs = self.sigmoid(fgmask)
                        segmask = np.zeros(fgmask.shape[:2])
                        segmask[fg_probs > seg_thresh] = 1.0

                        # Store segmentation mask for DROID-SLAM
                        motion_mask = torch.from_numpy(segmask).unsqueeze(0).bool().to(self.device)

                        # Resize/Crop segmask (Resize + Crop + Downscale 1/4)
                        dummysample = {'segmask': segmask}
                        dummysample = self.resizedata(dummysample)
                        segmask = dummysample['segmask']
                        segmask = cv2.resize(segmask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                        segmask = segmask[None, None, ...].astype(np.float32)
                        segmask = torch.from_numpy(np.concatenate((segmask,) * img0.shape[0], axis=0)).cuda()

        # Calculate the percentage of True values
        true_percentage = torch.sum(motion_mask) / (H * W)

        # If more than 60% are True, set everything to False
        if true_percentage > 0.6:
            motion_mask[:] = False

        # Store motion mask between frame 0 and 1 also for frame 0
        if self.k == 1:
            self.video.motion_masks[0] = motion_mask
            self.video.all_motion_masks[0] = self.video.motion_masks[0]

        # Store previous image [otherwise only keyframes are stored]:
        self.prev_image = image

        # return motion_mask
        return motion_mask

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, num_images=0):
        """ main update operation - run on every frame in video """

        self.num_images = num_images

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        H, W = image.shape[-2:]

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        if self.semantic_seg:
            # create segmentation mask
            segmentation_mask = self._segment(image.to(self.device))
        else:
            segmentation_mask = torch.zeros(1, H, W, device=self.device).bool()
        # Store all masks (also of non-keyframes)
        self.video.all_segmentation_masks[self.k] = segmentation_mask

        if self.motion_seg:
            # create motion mask
            motion_mask = self.__get_motion_mask(image, intrinsics)                
        else:
            motion_mask = torch.zeros(1, H, W, device=self.device).bool()
        # Store all masks and midas disps (also of non-keyframes)
        self.video.all_motion_masks[self.k] = motion_mask

        self.k += 1

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0], segmentation_mask, motion_mask)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], segmentation_mask, motion_mask)

            else:
                self.count += 1

    def load_vo_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('VO Model %s loaded...' % modelname)
        return model
    
    def load_seg_model(self, model, modelname):
        model = nn.DataParallel(model, device_ids=[0])
        preTrainDict = torch.load(modelname, map_location='cpu')
        self.mean_L = preTrainDict['mean_L']
        self.mean_R = preTrainDict['mean_R']
        preTrainDict['state_dict'] = {k:v for k,v in preTrainDict['state_dict'].items()}
        model.load_state_dict(preTrainDict['state_dict'], strict=False)
        print('Segmentation Model %s loaded...' % modelname)
        return model
    
    def initialize_segnet_input(self, imgL_o, intrinsics):
        maxh = imgL_o.shape[0] * self.testres
        maxw = imgL_o.shape[1] * self.testres
        self.max_h = int(maxh // 64 * 64)
        self.max_w = int(maxw // 64 * 64)
        if self.max_h < maxh: self.max_h += 64
        if self.max_w < maxw: self.max_w += 64
        self.input_size = imgL_o.shape

        # modify module according to inputs
        for i in range(len(self.segnet.module.reg_modules)):
            self.segnet.module.reg_modules[i] = flow_reg([1, self.max_w//(2**(6-i)), self.max_h//(2**(6-i))], 
                            ent=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(self.segnet.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(self.segnet.module.warp_modules)):
            self.segnet.module.warp_modules[i] = WarpModule([1, self.max_w//(2**(6-i)), self.max_h//(2**(6-i))]).cuda()

        # foramt intrinsics input
        fl, cx, cy, bl = intrinsics
        fl_next = fl  # assuming focal length remains the same across frames
        self.intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        self.intr_list.append(torch.Tensor([self.input_size[1] / self.max_w]).cuda()) # delta fx
        self.intr_list.append(torch.Tensor([self.input_size[0] / self.max_h]).cuda()) # delta fy
        self.intr_list.append(torch.Tensor([fl_next]).cuda())

    def transform_segnet_input(self, imgL_o, imgR_o):
        imgL = cv2.resize(imgL_o, (self.max_w, self.max_h))
        imgR = cv2.resize(imgR_o, (self.max_w, self.max_h))
        imgL_noaug = torch.Tensor(imgL / 255.)[np.newaxis].float().cuda()
        
        # flip channel, subtract mean
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(self.mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(self.mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        imgLR = torch.cat([imgL,imgR],0)

        return imgL_noaug, imgLR