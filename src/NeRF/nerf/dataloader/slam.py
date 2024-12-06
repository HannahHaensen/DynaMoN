import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R

from .ray_utils import *

def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()

def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()

def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w


class RGBDDataset(Dataset):
    def __init__(self, images, poses, timestamps, intrinsics, motion_masks,
                 segmentation_masks, scene_bbox_min, scene_bbox_max,
                 split='train', downsample=1.0, is_stack=False, N_vis=-1,
                 cal_fine_bbox=False):

        self.N_vis = N_vis
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.ndc_ray = False
        self.depth_data = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.images = images
        self.poses = poses
        self.timestamps = timestamps
        self.intrinsics = intrinsics
        self.masks = motion_masks | segmentation_masks

        self.white_bg = False
        self.near_far = [0.0, 10.0]
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.world_bound_scale = 1.1

        self.scene_bbox = torch.Tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        if cal_fine_bbox and split == "train":
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        self.name_dataset = "rgbd_slam"
   
    def read_meta(self):

        traj_droid = self.poses

        rot = R.from_quat(traj_droid[:, 3:])
        rot = rot.as_matrix()
        poses = np.eye(4, 4)[None, :].repeat(rot.shape[0], axis=0)
        poses[:, :3, :3] = rot
        poses[:, :3, 3] = traj_droid[:, :3]

        t = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        poses = [np.dot(t, p) for p in poses]

        t = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        poses = [np.dot(t, p) for p in poses]
        
        w, h = int(640 / self.downsample), int(480 / self.downsample)
        self.img_wh = [w, h]

        self.focal_x, self.focal_y, self.cx, self.cy = self.intrinsics

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx],[0, self.focal_y, self.cy],[0, 0, 1]]).float()

        self.poses = []
        timestamps = []
        self.all_times = []

        # Use every 8-th frame for testing
        if self.split == "train":
            idxs = list(range(0, len(self.images)))
            idxs = [num for num in idxs if (num+1) % 8 != 0]
            N_images_train = len(idxs)
            if self.is_stack:
                self.all_rays = []
                self.all_rgbs = []
                self.all_masks = []
            else:
                self.all_rays = torch.zeros(h * w * N_images_train, 6)
                self.all_rgbs = torch.zeros(h * w * N_images_train, 3)
                self.all_masks = torch.zeros(h * w * N_images_train)
        else:
            idxs = list(range(0, len(self.images)))
            idxs = [num for num in idxs if (num+1) % 8 == 0]
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []

        for t, i in enumerate(tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})')):#img_list:#
            c2w = torch.from_numpy(poses[i]).float()
            self.poses += [c2w]

            image_path = self.images[i]
            img = Image.open(image_path)
            
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(-1, w * h).permute(1, 0) # (h*w, 3) RGBA
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            mask = F.interpolate(self.masks[i].unsqueeze(1).float(),
                                 scale_factor=2).view(w * h).bool() # (h'w)

            if self.is_stack:
                self.all_rgbs += [img]
                self.all_masks += [mask]
            else:
                self.all_rgbs[t*h*w: (t+1)*h*w] = img
                self.all_masks[t*h*w: (t+1)*h*w] = mask

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            if self.is_stack:
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            else:
                self.all_rays[t*h*w: (t+1)*h*w] = torch.cat([rays_o, rays_d], 1)

            timestamps.append(self.timestamps[i])

        for i in range(len(timestamps)):
            timestamp = torch.tensor(timestamps[i], dtype=torch.float64).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses)

        if not self.is_stack:
            self.all_times = torch.cat(self.all_times, 0)
        else:
            print("Stacking ...")
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)
            self.all_times = torch.stack(self.all_times, 0)
            self.all_masks = torch.stack(self.all_masks, 0)
            print("Stack performed !!")

        # Normalization over all timestamps
        self.timestamps = np.array(self.timestamps)
        self.all_times = ((self.all_times - self.timestamps.min()) / (self.timestamps.max() - self.timestamps.min()) * 2.0 - 1.0).float()

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        poses = torch.eye(4).unsqueeze(0).repeat(self.poses.shape[0], 1, 1)
        poses[:, :, :] = self.poses
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(poses)[:,:3]
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays':   self.all_rays[idx],
                      'rgbs':   self.all_rgbs[idx],
                      'time':   self.all_times[idx],
                      'masks':  self.all_masks[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            mask = self.all_masks[idx]

            sample = {'rays':   rays,
                      'rgbs':   img,
                      'time':   time,
                      'masks':  mask}

        return sample
    
    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, 0.0, 0.0)
                for angle in np.linspace(60, 100, 500 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, 1.0 * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get validation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = val_poses[i]
            rays_o, rays_d = get_rays(self.directions, c2w.float())  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        if self.is_stack:
            rays_o = self.all_rays[:, :, 0:3]
            viewdirs = self.all_rays[:, :, 3:6]
        else:
            rays_o = self.all_rays[:, 0:3]
            viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        if self.is_stack:
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
        else:
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max
    
    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose