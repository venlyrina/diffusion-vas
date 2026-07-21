import os
import cv2
import numpy as np
import argparse
import imageio
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from PIL import Image


import torch.utils.checkpoint
from torchvision import transforms

from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from vggt_depth_adapter import vggt_depth_to_diffvas

import warnings
warnings.filterwarnings("ignore")




def _uniform_sample_indices(total_frames, target_frames):
    """Match uniform whole-sequence sampling: linspace over [0, total_frames-1]."""
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")
    if target_frames > total_frames:
        raise ValueError(
            f"VGGT has {target_frames} depth frames but input sequence has only "
            f"{total_frames} frames. Cannot align by downsampling."
        )
    if target_frames == total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, target_frames).round().astype(int).tolist()


def _sample_temporal_tensor(x, indices):
    """Select temporal frames from tensors shaped [B,F,...]."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return torch.index_select(x, 1, idx)


def init_amodal_segmentation_model(model_path_mask):
    pipeline_mask = DiffusionVASPipeline.from_pretrained(
        model_path_mask, torch_dtype=torch.float16
    ).to("cuda")
    pipeline_mask.enable_model_cpu_offload()
    pipeline_mask.set_progress_bar_config(disable=True)

    return pipeline_mask


def init_rgb_model(model_path_rgb):
    pipeline_rgb = DiffusionVASPipeline.from_pretrained(
        model_path_rgb, torch_dtype=torch.float16
    ).to("cuda")
    pipeline_rgb.enable_model_cpu_offload()
    pipeline_rgb.set_progress_bar_config(disable=True)

    return pipeline_rgb


def init_depth_model(model_path_depth, depth_encoder):

    from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

    depth_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_model = DepthAnythingV2(**depth_model_configs[depth_encoder]).to('cuda')
    depth_model.load_state_dict(
        torch.load(model_path_depth))
    depth_model.eval()

    return depth_model


def load_and_transform_masks(image_folder, resolution=(512, 1024)):
    # Define mask transformation: resize, to tensor, repeat grayscale channel, normalize
    mask_transform = transforms.Compose([
        transforms.Resize(resolution),  # Resize to resolution
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel to 3 channels
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize
    ])

    # List and sort image file paths in the folder
    image_paths = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ])

    processed_frames = []  # List to store transformed frames
    original_size = None  # To capture original image size

    for image_path in image_paths:
        image = Image.open(image_path).convert('L')  # Open image and convert to grayscale
        binary_image = image.point(lambda p: 255 if p > 128 else 0)  # Binarize image
        if original_size is None:
            original_size = binary_image.size[::-1]  # Save original size as (height, width)
        transformed_frame = mask_transform(binary_image)  # Apply transformation
        processed_frames.append(transformed_frame)  # Append to list

    mask_tensor = torch.stack(processed_frames).unsqueeze(0)  # Stack frames and add batch dimension
    return mask_tensor, original_size  # Return tensor and original size


def load_and_transform_rgbs(image_folder, resolution=(512, 1024)):
    """Load RGB images from a folder, transform them, and return as tensor, original size, and raw images."""
    # Define RGB transformation: resize, to tensor, and normalize
    rgb_transform = transforms.Compose([
        transforms.Resize(resolution),  # Resize to resolution
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize
    ])

    # List and sort image file paths in the folder
    image_paths = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ])

    transformed_frames = []  # List for transformed frames
    raw_images = []  # List for raw images
    original_size = None  # To capture original image size

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        raw_images.append(np.array(image))
        if original_size is None:
            original_size = image.size[::-1]
        transformed_frame = rgb_transform(image)
        transformed_frames.append(transformed_frame)

    rgb_tensor = torch.stack(transformed_frames).unsqueeze(0)

    return rgb_tensor, original_size, np.array(raw_images)


def rgb_to_depth(rgb_tensor, depth_model):

    # Remove the batch dimension (shape becomes [num_frames, 3, height, width])
    rgb_images = rgb_tensor.squeeze(0)
    rgb_images = (((rgb_images + 1.0) / 2.0) * 255)

    depth_maps = []

    # Loop through each frame in the tensor
    for i in range(rgb_images.shape[0]):
        rgb_image_np = rgb_images[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        depth_map = depth_model.infer_image(rgb_image_np)
        depth_maps.append(depth_map)

    depth_maps_np = np.array(depth_maps)
    depth_maps_np = (depth_maps_np - depth_maps_np.min()) / (depth_maps_np.max() - depth_maps_np.min())

    depth_maps_np = depth_maps_np * 2 - 1
    depth_tensor = torch.tensor(depth_maps_np, dtype=torch.float32)

    depth_tensor_3channel = depth_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: [num_frames, 3, height, width]
    depth_tensor_3channel = depth_tensor_3channel.unsqueeze(0)

    return depth_tensor_3channel


def overlay_mask_on_image(rgb_img, mask, cmap_idx=None, random_color=False, boundary_thickness=3, darken_factor=2):
    # Ensure the input image is RGB and in the range [0, 1]
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"
    # assert rgb_img.min() >= 0 and rgb_img.max() <= 1, "Expected rgb_img values in the range [0, 1]"

    # Select a color for the mask overlay
    cmap = plt.get_cmap("tab10")

    cmap_idx = 4
    if cmap_idx is None and random_color:
        cmap_idx = np.random.randint(0, cmap.N)  # Randomly choose a colormap index if not provided

    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    boundary_color = color[:3] * darken_factor  # Darken the color by the darken_factor
    boundary_color = np.concatenate([boundary_color, [1.0]])  # Make boundary fully opaque

    # Create a boundary mask
    dilated_mask = binary_dilation(mask, iterations=boundary_thickness)
    boundary_mask = dilated_mask & ~mask

    # Create a colored mask in the range [0, 1]
    mask_image = np.zeros_like(rgb_img, dtype=np.float32)
    boundary_image = np.zeros_like(rgb_img, dtype=np.float32)

    for i in range(3):  # Apply the mask and boundary to each channel
        mask_image[..., i] = mask * color[i]
        boundary_image[..., i] = boundary_mask * boundary_color[i]

    # Combine the RGB image with the colored mask and boundary
    overlayed_image = np.clip(rgb_img * 0.5 + mask_image + boundary_image, 0, 1)

    return overlayed_image




def main(args):

    generator = torch.manual_seed(23)

    model_path_mask = args.model_path_mask
    pipeline_mask = init_amodal_segmentation_model(model_path_mask)

    model_path_rgb = args.model_path_rgb
    pipeline_rgb = init_rgb_model(model_path_rgb)
    
    # VGGT is the preferred depth source when --vggt is provided.
    # Depth Anything V2 is kept as a fallback for backward compatibility.
    depth_model = None
    if args.vggt is None:
        depth_encoder = args.depth_encoder
        model_path_depth = args.model_path_depth + f"/depth_anything_v2_{depth_encoder}.pth"
        print(f"[Depth] Using Depth Anything V2: {model_path_depth}")
        depth_model = init_depth_model(model_path_depth, depth_encoder)
    else:
        print(f"[Depth] Using cached VGGT depth: {args.vggt}")

    # --video points to a prepared Diffusion-VAS sequence directory containing:
    #   masks/  and  rgbs/
    # This avoids hard-coding data_path/seq_name while preserving the old CLI.
    if args.video is not None:
        seq_path = os.path.normpath(args.video)
        seq_name = os.path.basename(seq_path.rstrip("/\\"))
        if not seq_name:
            seq_name = "video"
    else:
        data_path = args.data_path
        seq_name = args.seq_name
        seq_path = os.path.join(data_path, seq_name)

    # Auto-detect RGB folder:
    # - original Diffusion-VAS demo layout: rgbs/
    # - SAM3 tracking exporter layout:     rgb/
    rgb_candidates = [
        os.path.join(seq_path, "rgbs"),
        os.path.join(seq_path, "rgb"),
    ]
    rgbs_path = next((p for p in rgb_candidates if os.path.isdir(p)), None)
    if rgbs_path is None:
        raise FileNotFoundError(
            f"Missing RGB directory under: {seq_path}. "
            "Expected either 'rgbs/' or 'rgb/'."
        )

    # Auto-detect masks:
    # - original Diffusion-VAS layout: masks/*.png
    # - SAM3 exporter layout:          masks/actor_N/*.png
    masks_root = os.path.join(seq_path, "masks")
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f"Missing masks directory: {masks_root}")

    direct_mask_files = [
        name for name in os.listdir(masks_root)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    if direct_mask_files:
        masks_path = masks_root
    else:
        masks_path = os.path.join(masks_root, args.actor)
        if not os.path.isdir(masks_path):
            actor_dirs = sorted(
                name for name in os.listdir(masks_root)
                if os.path.isdir(os.path.join(masks_root, name))
            )
            raise FileNotFoundError(
                f"Mask actor directory not found: {masks_path}. "
                f"Available: {actor_dirs}. Use --actor actor_N."
            )

    print(f"[Input] RGB frames: {rgbs_path}")
    print(f"[Input] Masks:      {masks_path}")

    data_output_path = args.data_output_path
    output_seq_path = os.path.join(data_output_path, seq_name)
    os.makedirs(f"{data_output_path}/{seq_name}", exist_ok=True)
    
    # output gif paths
    modal_masks_overlay_path = f"{output_seq_path}/modal_masks_overlay.gif"
    pred_amodal_masks_path = f"{output_seq_path}/pred_amodal_masks.gif"
    pred_amodal_masks_overlay_path = f"{output_seq_path}/pred_amodal_masks_overlay.gif"

    modal_rgb_path = f"{output_seq_path}/modal_rgb.gif"
    modal_rgb_overlay_path = f"{output_seq_path}/modal_rgb_overlay.gif"
    
    pred_amodal_rgb_path = f"{output_seq_path}/pred_amodal_rgb.gif"
    pred_amodal_rgb_overlay_path = f"{output_seq_path}/pred_amodal_rgb_overlay.gif"
    
    # load input modal masks and rgb images
    pred_res = (256, 512) # sometimes a higher resolution (e.g.,512x1024) might produce better results
    modal_pixels, ori_shape = load_and_transform_masks(masks_path, resolution=pred_res)
    rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(rgbs_path, resolution=pred_res)

    if modal_pixels.shape[1] != rgb_pixels.shape[1]:
        raise ValueError(
            f"Frame-count mismatch between masks ({modal_pixels.shape[1]}) "
            f"and RGBs ({rgb_pixels.shape[1]})."
        )

    if args.vggt is not None:
        # Read only VGGT depth metadata here to determine temporal alignment.
        with np.load(args.vggt, allow_pickle=False) as _vggt_data:
            if "depth" not in _vggt_data.files:
                raise KeyError(
                    f"Key 'depth' not found in {args.vggt}. "
                    f"Available keys: {_vggt_data.files}"
                )
            vggt_frames = int(_vggt_data["depth"].shape[0])

        source_frames = int(rgb_pixels.shape[1])

        if vggt_frames != source_frames:
            sample_indices = _uniform_sample_indices(source_frames, vggt_frames)
            print(
                f"[Temporal] Aligning full sequence ({source_frames} frames) "
                f"to VGGT ({vggt_frames} frames) with uniform linspace sampling."
            )
            print(f"[Temporal] Sample indices: {sample_indices}")

            modal_pixels = _sample_temporal_tensor(modal_pixels, sample_indices)
            rgb_pixels = _sample_temporal_tensor(rgb_pixels, sample_indices)

            # raw_rgb_pixels is returned by load_and_transform_rgbs() as a
            # NumPy array shaped [F,H,W,3], so sample axis 0 as well.
            if isinstance(raw_rgb_pixels, np.ndarray):
                if raw_rgb_pixels.shape[0] != source_frames:
                    raise ValueError(
                        f"raw_rgb_pixels frame mismatch: "
                        f"{raw_rgb_pixels.shape[0]} != {source_frames}"
                    )
                raw_rgb_pixels = raw_rgb_pixels[sample_indices]
            elif torch.is_tensor(raw_rgb_pixels):
                if raw_rgb_pixels.ndim >= 2 and raw_rgb_pixels.shape[1] == source_frames:
                    raw_rgb_pixels = _sample_temporal_tensor(
                        raw_rgb_pixels, sample_indices
                    )
                elif raw_rgb_pixels.shape[0] == source_frames:
                    idx = torch.as_tensor(
                        sample_indices, dtype=torch.long, device=raw_rgb_pixels.device
                    )
                    raw_rgb_pixels = torch.index_select(raw_rgb_pixels, 0, idx)

        depth_pixels = vggt_depth_to_diffvas(
            args.vggt,
            expected_frames=rgb_pixels.shape[1],
            invert=not args.vggt_no_invert,
            normalization=args.vggt_normalization,
        )
    else:
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    # Use the actual aligned temporal length everywhere downstream.
    num_frames = int(modal_pixels.shape[1])
    if int(rgb_pixels.shape[1]) != num_frames or int(depth_pixels.shape[1]) != num_frames:
        raise ValueError(
            f"Aligned frame mismatch before Diff-VAS: "
            f"masks={modal_pixels.shape[1]}, rgb={rgb_pixels.shape[1]}, "
            f"depth={depth_pixels.shape[1]}"
        )
    print(f"[Temporal] Diff-VAS num_frames={num_frames}")

    print("amodal segmentation by diffusion-vas ...")
    # predict amodal masks (amodal segmentation)
    pred_amodal_masks = pipeline_mask(
        modal_pixels,
        depth_pixels,
        height=pred_res[0],
        width=pred_res[1],
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]

    pred_amodal_masks = [np.array(img) for img in pred_amodal_masks]

    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    
    # save pred_amodal_masks
    modal_mask_union = (modal_pixels[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
    pred_amodal_masks = np.logical_or(pred_amodal_masks, modal_mask_union).astype('uint8')

    pred_amodal_masks_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)
                                       for frame in pred_amodal_masks])
    imageio.mimsave(pred_amodal_masks_path, (pred_amodal_masks_save * 255).astype(np.uint8), fps=8)
    
    
    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
        2).repeat(1, 1, 3, 1, 1)

    modal_obj_mask = (modal_pixels > 0).float()
    modal_background = 1 - modal_obj_mask
    rgb_pixels = (rgb_pixels + 1) / 2

    tmp_cmap_idx = np.random.randint(0, plt.get_cmap("tab10").N)
    rgb_pixels_save = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in
         rgb_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])

    amodal_masks_overlay = []
    for i in range(num_frames):
        tmp_rgb_amodal = overlay_mask_on_image(rgb_pixels_save[i], pred_amodal_masks_save[i].astype(np.uint8),
                                               cmap_idx=tmp_cmap_idx)
        amodal_masks_overlay.append(tmp_rgb_amodal)
    modal_mask_union = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for frame in
         modal_obj_mask[0, :, 0, :, :].cpu().numpy().astype(np.uint8)])

    # save amodal_masks_overlay
    amodal_masks_overlay_np = np.stack(amodal_masks_overlay, axis=0)
    imageio.mimsave(pred_amodal_masks_overlay_path, (amodal_masks_overlay_np * 255).astype(np.uint8), fps=8)

    modal_masks_overlay = []
    for i in range(num_frames):
        tmp_rgb_modal = overlay_mask_on_image(rgb_pixels_save[i], modal_mask_union[i].astype(np.uint8),
                                              cmap_idx=tmp_cmap_idx)
        modal_masks_overlay.append(tmp_rgb_modal)

    # save modal_masks_overlay
    modal_masks_overlay_np = np.stack(modal_masks_overlay, axis=0)
    imageio.mimsave(modal_masks_overlay_path, (modal_masks_overlay_np * 255).astype(np.uint8), fps=8)

    # save modal_rgb
    modal_rgb_pixels = rgb_pixels * modal_obj_mask + modal_background
    modal_rgb_pixels_save = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in
         modal_rgb_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    imageio.mimsave(modal_rgb_path, (modal_rgb_pixels_save * 255).astype(np.uint8), fps=8)

    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

    print("content completion by diffusion-vas ...")
    # predict amodal rgb (content completion)
    pred_amodal_rgb = pipeline_rgb(
        modal_rgb_pixels,
        pred_amodal_masks_tensor,
        height=pred_res[0],  # my_res[0]
        width=pred_res[1],  # my_res[1]
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]

    pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

    # save pred_amodal_rgb
    pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
    pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                     for frame in pred_amodal_rgb])
    imageio.mimsave(pred_amodal_rgb_path, pred_amodal_rgb_save, fps=8)

    # save pred_amodal_rgb_overlay
    transparency_factor = 0.5
    white_background = np.ones_like(raw_rgb_pixels) * 255
    raw_rgb_semi_transparent = np.clip(
        raw_rgb_pixels * transparency_factor + white_background * (1 - transparency_factor), 0, 255
    ).astype(np.uint8)
    pred_amodal_rgb_overlay = np.where(pred_amodal_masks_save[..., None] == 1, pred_amodal_rgb_save, raw_rgb_semi_transparent)
    imageio.mimsave(pred_amodal_rgb_overlay_path, pred_amodal_rgb_overlay, fps=8)

    # save modal_rgb_overlay
    modal_pixels = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for frame in
         modal_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    modal_rgb_overlay = np.where(np.array((modal_pixels > 0)[:, :, :, :]) == 1, raw_rgb_pixels, raw_rgb_semi_transparent)
    imageio.mimsave(modal_rgb_overlay_path, modal_rgb_overlay, format='GIF', fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video amodal segmentation and content completion using Diffusion-VAS."
    )

    parser.add_argument(
        "--model_path_mask",
        type=str,
        default="checkpoints/diffusion-vas-amodal-segmentation",
        help="Path to diffusion-vas amodal segmentation checkpoint.",
    )

    parser.add_argument(
        "--model_path_rgb",
        type=str,
        default="checkpoints/diffusion-vas-content-completion",
        help="Path to diffusion-vas content completion checkpoint.",
    )

    parser.add_argument(
        "--depth_encoder",
        type=str,
        default="vitl",  # or 'vits', vitl, 'vitg'
        help="Depth encoder type.",
    )

    parser.add_argument(
        "--model_path_depth",
        type=str,
        default="checkpoints/",
        help="Path to depth anything v2's checkpoint's parent folder.",
    )


    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Path to a prepared sequence directory. RGB frames may be in "
            "'rgbs/' or 'rgb/'. Masks may be directly in 'masks/' or under "
            "'masks/actor_N/'. Overrides --data_path/--seq_name."
        ),
    )

    parser.add_argument(
        "--actor",
        type=str,
        default="actor_0",
        help=(
            "Actor mask subfolder under masks/ for SAM3 exporter layout, "
            "for example actor_0 or actor_1. Ignored when masks are directly in masks/."
        ),
    )

    parser.add_argument(
        "--vggt",
        type=str,
        default=None,
        help=(
            "Path to a VGGT .npz output containing the 'depth' array. "
            "When provided, VGGT depth replaces Depth Anything V2 and the "
            "DA2 checkpoint is not loaded."
        ),
    )

    parser.add_argument(
        "--vggt_no_invert",
        action="store_true",
        help="Disable near/far inversion for VGGT depth (useful for A/B testing).",
    )

    parser.add_argument(
        "--vggt_normalization",
        type=str,
        choices=["minmax", "percentile"],
        default="minmax",
        help=(
            "VGGT depth normalization. 'minmax' matches the original DA2 "
            "sequence-global preprocessing; 'percentile' is experimental."
        ),
    )


    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="Path to input data.",
    )

    parser.add_argument(
        "--seq_name",
        type=str,
        default="birdcage",
        help="Input sequence name.",
    )

    parser.add_argument(
        "--data_output_path",
        type=str,
        default="outputs",
        help="Output path.",
    )

    args = parser.parse_args()

    main(args)

