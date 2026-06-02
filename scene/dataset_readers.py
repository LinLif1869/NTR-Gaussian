#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
import torch
from games.mesh_time_splatting.utils.graphics_utils import MeshPointCloud, MeshBasePointCloud
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text,\
    read_points3D_obj
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.dynamic_rgbt_metadata import build_frame_times, find_nerfies_root, load_optional_thermal_metadata
from utils.ironbow_utils import ironbow_to_gray_rgb_pil

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_image: np.array
    alpha_mask: np.array
    time : float
    thermal_image: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: float = None
    spacefeatures: torch.tensor = None
    thermal_metadata: dict = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              normal_image=None, alpha_mask=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"] 

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            alpha_mask = norm_data[:, :, 3]
            alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")
            # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_image_path = os.path.join(path, normal_cam_name)
            if os.path.exists(normal_image_path):
                normal_image = Image.open(normal_image_path)
                
                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data==128).sum(-1)==3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:,:,:3] * normal_norm_data[:, :, 3:4] + bg * (1 - normal_norm_data[:, :, 3:4])
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr*255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx 
                FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],time= None, normal_image=normal_image, alpha_mask=alpha_mask))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=None)
    return scene_info

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        if max_time_float ==0.0:
            timestamp_mapper[time] = time
        else:
            timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float

def readCamerasTimeFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"][2:] + extension)
            img_name = cam_name.split('/')[-1]
            if "time" in frame and frame["time"] != None:
                time = mapper[frame["time"]]
                time = torch.tensor(time).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
            else:
                time = None
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            alpha_mask = norm_data[:, :, 3]
            alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")
            # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_image_path = os.path.join(path, normal_cam_name)
            if os.path.exists(normal_image_path):
                normal_image = Image.open(normal_image_path)
                
                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data==128).sum(-1)==3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:,:,:3] * normal_norm_data[:, :, 3:4] + bg * (1 - normal_norm_data[:, :, 3:4])
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr*255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None
            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            if img_name.startswith('8'):
                fovx = frame['FovX']

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],time = time,normal_image=None, alpha_mask=None))
            
    return cam_infos

def readNerfSyntheticMeshTimeInfo(path, white_background, eval, num_splats=3, extension=".png"):  
    print("path",path)
    if path.endswith('Jadebay'):
        timestamp_mapper =None
        max_time =None
    else:
        timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasTimeFromTransforms(path, "transforms_train.json", white_background, extension,timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasTimeFromTransforms(path, "transforms_test.json", white_background, extension,timestamp_mapper)
    print("Reading Mesh object")
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "mesh.ply")

    obj_path = os.path.join(path, "mesh.obj")
    if not os.path.exists(ply_path):
        print("Converting mesh.obj to .ply, will happen only the first time you open the scene.")
        xyz, rgb = read_points3D_obj(obj_path)
        # xyz = transform_vertices_function(
        # torch.tensor(xyz)).numpy()
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    feature_path = os.path.join(path, "space_features.npy")
    if os.path.exists(feature_path):
        # space_features = torch.tensor(np.load(feature_path)[:, -33:-1]).unsqueeze(0).cuda()
        space_features = torch.tensor(np.load(feature_path)).cuda()
        # print(space_features.shape)
    else:
        space_features = torch.ones(pcd.points.shape[0], 32).cuda()
    # mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    # vertices = mesh_scene.vertices
    # vertices = transform_vertices_function(
    #     torch.tensor(vertices),
    # )
    # faces = mesh_scene.faces
    # triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()



    
    # shs = np.random.random((vertices.shape[vertices.shape[0]], 3)) / 255.0
    # pcd = MeshBasePointCloud(points=vertices, colors=SH2RGB(shs))

    # ply_path = os.path.join(path, "points3d.ply")

    # if not os.path.exists(ply_path):
    # if True:
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts_each_triangle = num_splats
    #     num_pts = num_pts_each_triangle * triangles.shape[0]
    #     print(
    #         f"Generating random point cloud ({num_pts})..."
    #     )

    #     # We create random points inside the bounds traingles
    #     alpha = torch.rand(
    #         triangles.shape[0],
    #         num_pts_each_triangle,
    #         3
    #     )

    #     xyz = torch.matmul(
    #         alpha,
    #         triangles
    #     )
    #     xyz = xyz.reshape(num_pts, 3)

    #     shs = np.random.random((num_pts, 3)) / 255.0

    #     pcd = MeshPointCloud(
    #         alpha=alpha,
    #         points=xyz,
    #         colors=SH2RGB(shs),
    #         normals=np.zeros((num_pts, 3)),
    #         vertices=vertices,
    #         faces=faces,
    #         transform_vertices_function=transform_vertices_function,
    #         triangles=triangles.cuda()
    #     )

    # storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime = max_time,
                           spacefeatures = space_features)
    return scene_info
def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, extension=".png"
):
    import trimesh

    timestamp_mapper =None
    max_time =None
    print("Reading Training Transforms")
    train_cam_infos = readCamerasTimeFromTransforms(path, "transforms_train.json", white_background, extension,timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasTimeFromTransforms(path, "transforms_val.json", white_background, extension,timestamp_mapper)
    print("Reading Mesh object")
    mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = num_splats
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda()
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime = max_time)
    return scene_info

def _load_json(path):
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

def _find_nerfies_image_dir(root):
    for dirname in ("rgb", "images"):
        image_dir = os.path.join(root, dirname)
        if not os.path.isdir(image_dir):
            continue
        for scale in ("2x", "1x", "4x"):
            scale_dir = os.path.join(image_dir, scale)
            if os.path.isdir(scale_dir):
                return scale_dir
        return image_dir
    raise FileNotFoundError(f"Nerfies image folder not found under {root}")

def _find_image_path(image_dir, image_id):
    for extension in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = os.path.join(image_dir, image_id + extension)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image '{image_id}' not found under {image_dir}")

def _camera_from_nerfies(camera_json, width, height):
    orientation = np.asarray(camera_json["orientation"], dtype=np.float32)
    position = np.asarray(camera_json["position"], dtype=np.float32)
    R = orientation.T
    T = -position @ R

    focal = camera_json["focal_length"]
    if isinstance(focal, (list, tuple)):
        focal_x = float(focal[0])
        focal_y = float(focal[1]) if len(focal) > 1 else focal_x
    else:
        focal_x = focal_y = float(focal)

    base_size = camera_json.get("image_size")
    if base_size is not None:
        focal_x *= width / float(base_size[0])
        focal_y *= height / float(base_size[1])
    focal_y *= float(camera_json.get("pixel_aspect_ratio", 1.0))
    return R, T, focal2fov(focal_y, height), focal2fov(focal_x, width)

def _thermal_image_to_gray(image):
    rgb = np.asarray(image.convert("RGB"))
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return image.convert("RGB")
    return ironbow_to_gray_rgb_pil(image)

def _load_nerfies_point_cloud(path, modality_root):
    roots = [path, modality_root, os.path.join(path, "rgb"), os.path.join(path, "thermal")]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for filename in ("mesh.ply", "points3D.ply", "points3d.ply"):
            ply_path = os.path.join(root, filename)
            if os.path.exists(ply_path):
                return fetchPly(ply_path), ply_path
        points_path = os.path.join(root, "points.npy")
        if os.path.exists(points_path):
            points = np.load(points_path)
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError(f"Expected points.npy with shape (N, >=3), got {points.shape}")
            xyz = np.asarray(points[:, :3], dtype=np.float32)
            if points.shape[1] >= 6:
                rgb = np.asarray(points[:, 3:6], dtype=np.float32)
                if rgb.max() <= 1.0:
                    rgb *= 255.0
            else:
                rgb = np.full((xyz.shape[0], 3), 127, dtype=np.float32)
            ply_path = os.path.join(root, "points3D.ply")
            storePly(ply_path, xyz, rgb.clip(0, 255))
            return fetchPly(ply_path), ply_path
    raise FileNotFoundError("Nerfies scene requires mesh.ply, points3D.ply, points3d.ply, or points.npy")

def _load_space_features(path, modality_root, num_points):
    for root in (path, modality_root, os.path.join(path, "rgb"), os.path.join(path, "thermal")):
        feature_path = os.path.join(root, "space_features.npy")
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            if features.ndim != 2 or features.shape[0] != num_points or features.shape[1] < 32:
                raise ValueError(
                    f"Expected space_features.npy with shape ({num_points}, >=32), got {features.shape}"
                )
            return torch.tensor(features, dtype=torch.float32).cuda()
    print("space_features.npy not found; using constant 32D features.")
    return torch.ones(num_points, 32, dtype=torch.float32).cuda()

def _get_nerfies_splits(dataset_json, eval, llffhold=8):
    ids = list(dataset_json.get("ids", []))
    if not ids:
        raise ValueError("Nerfies dataset.json does not contain any ids")
    if not eval:
        return ids, []

    test_ids = set(dataset_json.get("val_ids", []))
    if not test_ids:
        test_ids = {image_id for index, image_id in enumerate(ids) if index % llffhold == 0}
    configured_train_ids = set(dataset_json.get("train_ids", []))
    train_ids = configured_train_ids or (set(ids) - test_ids)
    return [image_id for image_id in ids if image_id in train_ids], [
        image_id for image_id in ids if image_id in test_ids
    ]

def readNerfiesThermalSceneInfo(path, eval, llffhold=8):
    modality_root = find_nerfies_root(path)
    if modality_root is None:
        raise FileNotFoundError("Nerfies scene requires dataset.json and camera/")

    dataset_json = _load_json(os.path.join(modality_root, "dataset.json"))
    metadata_path = os.path.join(modality_root, "metadata.json")
    frame_metadata = _load_json(metadata_path) if os.path.exists(metadata_path) else {}
    train_ids, test_ids = _get_nerfies_splits(dataset_json, eval, llffhold)
    all_ids = train_ids + [image_id for image_id in test_ids if image_id not in set(train_ids)]
    image_dir = _find_nerfies_image_dir(modality_root)

    raw_times = []
    for index, image_id in enumerate(all_ids):
        metadata = frame_metadata.get(image_id, {})
        raw_times.append(float(metadata.get("time_id", metadata.get("warp_id", index))))
    min_time = min(raw_times)
    max_time = max(raw_times)
    duration = max(max_time - min_time, 1.0)

    cameras = {}
    for index, (image_id, raw_time) in enumerate(zip(all_ids, raw_times)):
        image_path = _find_image_path(image_dir, image_id)
        thermal_image = Image.open(image_path).convert("RGB")
        image = _thermal_image_to_gray(thermal_image)
        width, height = image.size
        camera_json = _load_json(os.path.join(modality_root, "camera", image_id + ".json"))
        R, T, FovY, FovX = _camera_from_nerfies(camera_json, width, height)
        cameras[image_id] = CameraInfo(
            uid=index, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_id, width=width, height=height,
            normal_image=None, alpha_mask=None,
            time=torch.tensor((raw_time - min_time) / duration).view(1, 1, 1).cuda(),
            thermal_image=thermal_image,
        )

    train_cameras = [cameras[image_id] for image_id in train_ids]
    test_cameras = [cameras[image_id] for image_id in test_ids]
    pcd, ply_path = _load_nerfies_point_cloud(path, modality_root)
    space_features = _load_space_features(path, modality_root, len(pcd.points))
    thermal_metadata = load_optional_thermal_metadata(path)
    fps = float(thermal_metadata.get("fps", 30.0))
    if fps <= 0:
        fps = 30.0
        thermal_metadata["fps"] = fps
    frame_count = len(dataset_json["ids"])
    frame_times = build_frame_times(frame_count, fps)
    thermal_metadata["frame_count"] = frame_count
    thermal_metadata["time_interval"] = frame_times[-1] if len(frame_times) > 1 else 1.0 / fps
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=getNerfppNorm(train_cameras or test_cameras),
        ply_path=ply_path,
        maxtime=thermal_metadata["time_interval"],
        spacefeatures=space_features,
        thermal_metadata=thermal_metadata,
    )

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Blender_Mesh_time": readNerfSyntheticMeshTimeInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Nerfies": readNerfiesThermalSceneInfo,
}
