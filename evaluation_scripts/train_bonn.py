import sys
sys.path.append('src/localization/droid_slam')
sys.path.append('src')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob 
import time
import argparse
import gc
import random
import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from droid import Droid

from NeRF.config.config import Config
from NeRF.nerf.dataloader import get_test_dataset, get_train_dataset
from NeRF.nerf.model import init_model
from NeRF.nerf.render.render import evaluation, evaluation_path
from NeRF.nerf.render.trainer import Trainer

from metric import camera_to_rel_deg

def image_stream(stride, datapath, camera_params):
    """ image generator """

    fx, fy, cx, cy = camera_params

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.000000])

    # read all png images in folder
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
    
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)
        image = cv2.undistort(image, K_l, d_l)
        image = cv2.resize(image, (320+32, 240+16))
        image = torch.from_numpy(image).permute(2,0,1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        yield t, image[None], intrinsics, len(images_list)

def reconstruction(cfg, images, poses, timestamps, device,
                   intrinsics, motion_masks, segmentation_masks):
        if cfg.data.datasampler_type == "rays":
            train_dataset = get_train_dataset(cfg,
                                              is_stack=False,
                                              images=images,
                                              poses=poses,
                                              timestamps=timestamps,
                                              intrinsics=intrinsics,
                                              motion_masks=motion_masks,
                                              segmentation_masks=segmentation_masks)
        else:
            train_dataset = get_train_dataset(cfg,
                                              is_stack=True,
                                              images=images,
                                              poses=poses,
                                              timestamps=timestamps,
                                              intrinsics=intrinsics,
                                              motion_masks=motion_masks,
                                              segmentation_masks=segmentation_masks)
            
        cfg.data.scene_bbox_min = train_dataset.scene_bbox[0].tolist()
        cfg.data.scene_bbox_max = train_dataset.scene_bbox[1].tolist()

        # Real test dataset for final evaluation is created in HexPlane dataloader
        test_dataset = get_test_dataset(cfg,
                                        is_stack=True,
                                        images=images,
                                        poses=poses,
                                        timestamps=timestamps,
                                        intrinsics=intrinsics,
                                        motion_masks=motion_masks,
                                        segmentation_masks=segmentation_masks)
        
        cfg.model.time_grid_final = train_dataset.poses.shape[0] - 1
            
        ndc_ray = test_dataset.ndc_ray
        white_bg = test_dataset.white_bg
        near_far = test_dataset.near_far

        if cfg.systems.add_timestamp:
            logfolder = f'{cfg.systems.basedir}/{os.path.basename(args.datapath)}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        else:
            logfolder = f"{cfg.systems.basedir}/{os.path.basename(args.datapath)}"

        # init log file
        os.makedirs(logfolder, exist_ok=True)
        os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
        os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
        os.makedirs(f"{logfolder}/rgba", exist_ok=True)
        summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
        cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
        with open(cfg_file, "w") as f:
            OmegaConf.save(config=cfg, f=f)

        # init model.
        aabb = train_dataset.scene_bbox.to(device)
        HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

        # init trainer.
        trainer = Trainer(
            HexPlane,
            cfg,
            reso_cur,
            train_dataset,
            test_dataset,
            summary_writer,
            logfolder,
            device,
        )
        print("Training the model...")
        pose_net = trainer.train()

        torch.save(HexPlane, f"{logfolder}/{cfg.expname}.th")
        torch.save(pose_net, f"{logfolder}/{cfg.expname}_poses.th")
        # Render training viewpoints.
        if cfg.render_train:
            os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
            train_dataset = get_train_dataset(cfg, is_stack=True)
            evaluation(
                train_dataset,
                HexPlane,
                cfg,
                f"{logfolder}/imgs_train_all/",
                prefix="train",
                N_vis=-1,
                N_samples=-1,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )

        # Render test viewpoints.
        if cfg.render_test:
            os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
            evaluation(
                test_dataset,
                HexPlane,
                cfg,
                f"{logfolder}/{cfg.expname}/imgs_test_all/",
                prefix="test",
                N_vis=-1,
                N_samples=-1,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )

        # Render validation viewpoints.
        if cfg.render_path:
            os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
            evaluation_path(
                test_dataset,
                HexPlane,
                cfg,
                f"{logfolder}/{cfg.expname}/imgs_path_all/",
                prefix="validation",
                N_vis=-1,
                N_samples=-1,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--filter_motion", action="store_true", help="Enable motion filtering")
    parser.add_argument("--filter_semantics", action="store_true", help="Enable semantic (person) filtering")

    parser.add_argument("--config", default="src/NeRF/config/rgbd_slam.yaml")
    args = parser.parse_args()

    base_cfg = OmegaConf.structured(Config())
    base_yaml_path = args.config
    if base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg)  # merge configs

    # Fix random seed
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    if args.filter_motion:
        print("Motion filtering is enabled.")
    else:
        print("Motion filtering is disabled.")

    if args.filter_semantics:
        print("Semantic filtering is enabled.")
    else:
        print("Semantic filtering is disabled.")
        
    camera_params = [542.822841, 542.576870, 315.593520, 237.756098]

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)
    time.sleep(5)

    tstamps = []
    for (t, image, intrinsics, num_images) in tqdm(image_stream(args.stride, args.datapath, camera_params=camera_params)):
        droid.track(t, image, intrinsics=intrinsics, num_images=num_images)

    traj_est = droid.terminate(image_stream(args.stride, args.datapath, camera_params=camera_params))

    segmentation_masks = droid.video.all_segmentation_masks.cpu()
    motion_masks = droid.video.all_motion_masks.cpu()

    poses_for_nerf = traj_est.copy()

    ### run evaluation ###

    print("#"*20 + " Results...")

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::args.stride]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=np.roll(traj_est[:, 3:], 1),
        timestamps=np.array(tstamps))
    
    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(torch.from_numpy(np.stack(traj_est.poses_se3)).cuda(),
                                                       torch.from_numpy(np.stack(traj_ref.poses_se3)).cuda(),
                                                       'cuda', 16)

    # metrics to report
    Racc_5 = (rel_rangle_deg < 5).float().mean()
    Racc_15 = (rel_rangle_deg < 15).float().mean()

    print("Racc_5: {}".format(Racc_5.item()))
    print("Racc_15: {}".format(Racc_15.item()))

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    print("End of localization.")
    
    # Free GPU memory from SLAM
    droid = None
    gc.collect()
    torch.cuda.empty_cache()

    print("Start fitting dynamic NeRF.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    reconstruction(cfg, images_list, poses_for_nerf, tstamps, device,
                   camera_params, motion_masks, segmentation_masks)