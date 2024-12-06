from .slam import RGBDDataset

def get_train_dataset(cfg, is_stack=False, images=[], poses=[],
                      timestamps=[], intrinsics=[],
                      motion_masks=[], segmentation_masks=[]):
    if cfg.data.dataset_name == "slam":
        train_dataset = RGBDDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            motion_masks,
            segmentation_masks,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True, images=[], poses=[],
                     timestamps=[], intrinsics=[],
                     motion_masks=[], segmentation_masks=[]):
    if cfg.data.dataset_name == "slam":
        test_dataset = RGBDDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            motion_masks,
            segmentation_masks,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset
