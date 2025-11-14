"""
@File: utils.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The dataset utils.
"""

import numpy as np
import sys
import math
import cv2 as cv

def render_feature_as_tex(uv_face_id_map, uv_bary_weight_map, faces, features):
    resolution = uv_face_id_map.shape[0]

    ret_feat_map = np.zeros(shape=[resolution, resolution, 3])
    idx, idy = np.where(uv_face_id_map >= 0)

    uv_face_idx = uv_face_id_map[idx,idy]
    vert_idx = faces[uv_face_idx,:]

    p = features[vert_idx[:,:].reshape(-1),:].reshape([-1,3,3])
    w = uv_bary_weight_map[idx,idy,:]

    fin_feats = p[:,0,:] * w[:,0,None] + p[:,1,:] * w[:,1,None] + p[:,2,:] * w[:,2,None]
    ret_feat_map[idx,idy,:] = fin_feats

    return ret_feat_map, fin_feats

def safe_cross_prod(st, ed):
    stXed = np.cross(st, ed, axisa=-1, axisb=-1)
    empty_inds = np.where(np.sum(np.abs(stXed),axis=-1) < 1e-9)[0]
    stXed[empty_inds,:] = np.array([0.,1.,0.])
    return stXed

def gen_uv_barycentric(face_idx, face_texture_coords, resolution=512):
    uv_face_id = np.ones(shape=(resolution,resolution)) * (-1.0)
    uv_bary_weights = np.zeros(shape=(resolution,resolution,3))

    for i in range(face_idx.shape[0]):
        cur_face_uv_coords = face_texture_coords[i]
        uu_min = np.clip(np.min(cur_face_uv_coords[:,0]) * resolution - 2, 0, resolution - 1)
        uu_max = np.clip(np.max(cur_face_uv_coords[:,0]) * resolution + 2, 0, resolution - 1)
        vv_min = np.clip(np.min(cur_face_uv_coords[:,1]) * resolution - 2, 0, resolution - 1)
        vv_max = np.clip(np.max(cur_face_uv_coords[:,1]) * resolution + 2, 0, resolution - 1)
        uu_min, uu_max, vv_min, vv_max = int(uu_min), int(uu_max), int(vv_min), int(vv_max)

        # uu height, vv weigth
        for xx in range(uu_min, uu_max + 1):
            for yy in range(vv_min, vv_max + 1):
                fin_x, fin_y = xx, yy
                #resolution - yy - 1
                if uv_face_id[fin_x,fin_y] == -1:
                    px, py = (xx)/resolution, (yy)/resolution
                    
                    p0x, p0y = cur_face_uv_coords[0, 0], cur_face_uv_coords[0, 1]
                    p1x, p1y = cur_face_uv_coords[1, 0], cur_face_uv_coords[1, 1]
                    p2x, p2y = cur_face_uv_coords[2, 0], cur_face_uv_coords[2, 1]
                    
                    signed_area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)

                    w_1 = 1 / (2 * signed_area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
                    w_2 = 1 / (2 * signed_area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)
                    w_0 = 1 - w_1 - w_2

                    if (w_0 >= 0) and (w_1 >= 0) and (w_2 >= 0):
                        uv_face_id[fin_x, fin_y] = i
                        uv_bary_weights[fin_x, fin_y, 0] = w_0
                        uv_bary_weights[fin_x, fin_y, 1] = w_1
                        uv_bary_weights[fin_x, fin_y, 2] = w_2
                else:
                    pass

    uv_face_id = uv_face_id.astype(np.int32)

    return uv_face_id, uv_bary_weights

def dilate_barycentric_maps(uv_face_id, uv_bary_weights):
    k_x = [-1, -1, -1, 0, 0, 1, 1, 1]
    k_y = [-1, 0, 1, -1, 1, -1, 0, 1]

    hh, ww = uv_face_id.shape[0], uv_face_id.shape[1]
    dist_mat = np.ones((hh, ww)) * (-1.0)
    new_uv_face_id_mat = np.ones((hh, ww),dtype=np.int32) * (-1)
    new_uv_bary_weights = np.zeros((hh, ww, 3)) * (-1.0)

    for i in range(hh):
        for j in range(ww):
            if uv_face_id[i,j] == -1:
                continue
            new_uv_face_id_mat[i,j] = uv_face_id[i,j]
            new_uv_bary_weights[i,j,:] = uv_bary_weights[i,j,:]
            dist_mat[i,j] = 0

            for k in range(len(k_x)):
                d_x, d_y = k_x[k], k_y[k]
                t_x, t_y = i + d_x, j + d_y
                if (t_x < 0) or (t_y < 0) or (t_x >= hh) or (t_y >= ww):
                    continue
                dist_new = dist_mat[t_x, t_y]
                if (dist_new == -1) or ((np.abs(d_x)+np.abs(d_y))<dist_new):
                    dist_mat[t_x, t_y] = np.abs(d_x)+np.abs(d_y)
                    new_uv_bary_weights[t_x,t_y,:] = uv_bary_weights[i,j,:]
                    new_uv_face_id_mat[t_x,t_y] = uv_face_id[i, j]

    return new_uv_face_id_mat, new_uv_bary_weights

def split_number(number, split_num):
    ret = []
    if number < split_num:
        print('wrong splitting')
        sys.exit(0)
    else:
        current_sum = 0
        for i in range(split_num):
            if i == (split_num - 1):
                ret.append(number - current_sum)
            else:
                ret.append(int(math.floor(number / split_num)))
                current_sum += int(math.floor(number / split_num))

    return ret

def createMeshSequenceTensorByList(inputMeshesFile, scale=1.0, frame_list = []):

    meshesFileVertices = open(inputMeshesFile, 'r')
    allFrames= {}
    real_frame_list = []
    counter=0
    
    frame_list.reverse()

    for frameVerticesLine in meshesFileVertices:
        if len(frame_list) == 0:
            break 
        
        if (counter == 0):
            counter = counter + 1
            continue
        
        verticesLineSplit = frameVerticesLine.split()
        frame = int(verticesLineSplit[0])

        if frame == frame_list[-1]:
            vertices = verticesLineSplit[1:]
            allFrames[frame]= np.array([ float(x) for x in vertices], dtype=np.float32).reshape([-1, 3]) * scale
            frame_list.pop()
            real_frame_list.append(frame)

    meshesFileVertices.close()
    return allFrames, real_frame_list



def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

# https://github.com/ratcave/wavefront_reader
def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(' ')

                    # assume texture maps are in the same level
                    # WARNING: do not include space in your filename!!                    
                    if 'map' in prefix:
                        material[prefix] = split_data[-1].split('\\')[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials

def load_obj_mesh_mtl(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    # face per material
    face_data_mat = {}
    face_norm_data_mat = {}
    face_uv_data_mat = {}

    # current material name
    mtl_data = None
    cur_mat = None

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'mtllib':
            mtl_data = read_mtlfile(mesh_file.replace(mesh_file.split('/')[-1],values[1]))
        elif values[0] == 'usemtl':
            cur_mat = values[1]
        elif values[0] == 'f':
            # local triangle data
            l_face_data = []
            l_face_uv_data = []
            l_face_norm_data = []

            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, [values[3], values[4], values[1]]))
                l_face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]) if int(x.split('/')[0]) < 0 else int(x.split('/')[0])-1, values[1:4]))
                l_face_data.append(f)
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, [values[3], values[4], values[1]]))
                    l_face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]) if int(x.split('/')[1]) < 0 else int(x.split('/')[1])-1, values[1:4]))
                    l_face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, [values[3], values[4], values[1]]))
                    l_face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]) if int(x.split('/')[2]) < 0 else int(x.split('/')[2])-1, values[1:4]))
                    l_face_norm_data.append(f)
            
            face_data += l_face_data
            face_uv_data += l_face_uv_data
            face_norm_data += l_face_norm_data

            if cur_mat is not None:
                if cur_mat not in face_data_mat.keys():
                    face_data_mat[cur_mat] = []
                if cur_mat not in face_uv_data_mat.keys():
                    face_uv_data_mat[cur_mat] = []
                if cur_mat not in face_norm_data_mat.keys():
                    face_norm_data_mat[cur_mat] = []
                face_data_mat[cur_mat] += l_face_data
                face_uv_data_mat[cur_mat] += l_face_uv_data
                face_norm_data_mat[cur_mat] += l_face_norm_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    norms = np.array(norm_data)
    norms = normalize_v3(norms)
    face_normals = np.array(face_norm_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, norms, face_normals, uvs, face_uvs)

    if cur_mat is not None and mtl_data is not None:
        for key in face_data_mat:
            face_data_mat[key] = np.array(face_data_mat[key])
            face_uv_data_mat[key] = np.array(face_uv_data_mat[key])
            face_norm_data_mat[key] = np.array(face_norm_data_mat[key])
        
        out_tuple += (face_data_mat, face_norm_data_mat, face_uv_data_mat, mtl_data)

    return out_tuple

def load_calibrations_v10(calibration_file_name):
    cali_arr = []
    f = open(calibration_file_name,'r').readlines()
    """
    Skeletool Camera Calibration File V1.0
    name        0
    sensor      14.1864 10.3776
    size        1285 940
    animated    0
    intrinsic   766.3206 0 635.1871 0 0 765.7619 448.8466 0 0 0 1 0 0 0 0 1 
    extrinsic   -0.453446 -0.3859819 -0.8033708 703.8229 0.6526945 -0.7576083 -0.004404724 150.761 -0.6069403 -0.5263531 0.5954627 4868.044 0 0 0 1 
    radial      0
    """
    cali_arr = []
    f = open(calibration_file_name,'r').readlines()
    st_line = 1
    block_size = 7
    while st_line < len(f):
        id = int(f[st_line].split()[1])
        sensor = [float(x) for x in f[st_line+1].rstrip().split()[1:3]]     
        intrinsics = [
            [float(x) for x in f[st_line+4].rstrip().split()[1:4]],
            [float(x) for x in f[st_line+4].rstrip().split()[5:8]],
            [float(x) for x in f[st_line+4].rstrip().split()[9:12]]
        ]
        extrinsics = [
            [float(x) for x in f[st_line+5].rstrip().split()[1:5]],
            [float(x) for x in f[st_line+5].rstrip().split()[5:9]],
            [float(x) for x in f[st_line+5].rstrip().split()[9:13]]
        ]
        cur_cali_dict = {
            'id': id,
            'sensor': np.array(sensor),
            'intrinsics':np.array(intrinsics),
            'extrinsics':np.array(extrinsics)
        }
        cali_arr.append(cur_cali_dict)
        st_line += block_size
    return cali_arr

def load_K_Rt_from_P(P=None):

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def compute_tangent(vertices, faces, normals, uvs, faceuvs):    
    c1 = np.cross(normals, np.array([0,1,0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    return tan, btan
