import math
import numpy as np
import cv2
import json

class_id_to_str = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    31: 'train',
    32: 'motorcyclist',
    33: 'bicyclist'
}

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def LabelLoader2D3D_Synscapes(meta_dir, file_id):
    with open(meta_dir + file_id + ".json") as meta_file:
        meta = json.load(meta_file)

    cam_info = meta["camera"]
    instance_info = meta["instance"]

    resx = cam_info['intrinsic']['resx']
    resy = cam_info['intrinsic']['resy']

    bboxes_2d = instance_info['bbox2d']
    bboxes_3d = instance_info['bbox3d']
    classes = instance_info['class']
    occlusions = instance_info['occluded']
    truncations = instance_info['truncated']

    labels = list()
    for object_id in bboxes_2d:
        label = dict()

        ########################################################################
        # 2D label:
        ########################################################################
        label_2D = dict()

        label_2D['class'] = class_id_to_str[classes[object_id]]
        label_2D['truncated'] = truncations[object_id]
        label_2D['occluded'] = occlusions[object_id]
        label_2D['poly'] = np.array([[resx*bboxes_2d[object_id]['xmin'],
                                      resy*bboxes_2d[object_id]['ymin']],
                                      [resx*bboxes_2d[object_id]['xmax'],
                                      resy*bboxes_2d[object_id]['ymin']],
                                      [resx*bboxes_2d[object_id]['xmax'],
                                      resy*bboxes_2d[object_id]['ymax']],
                                      [resx*bboxes_2d[object_id]['xmin'],
                                      resy*bboxes_2d[object_id]['ymax']]],
                                      dtype='int32')

        label["label_2D"] = label_2D

        ########################################################################
        # 3D label:
        ########################################################################
        label_3D = dict()

        label_3D['class'] = class_id_to_str[classes[object_id]]

        extr = cam_info['extrinsic']
        intr = cam_info['intrinsic']
        rpy = (math.radians(extr['roll']), math.radians(extr['pitch']), math.radians(extr['yaw']))
        t = (extr['x'], extr['y'], extr['z'])
        R = eulerAnglesToRotationMatrix(rpy)
        T = translationMatrix(t)
        label_3D['R'] = R
        label_3D['T'] = T
        P0 = np.array([[intr['fx'], 0, intr['u0'], 0],
                       [0, intr['fy'], intr['v0'], 0],
                       [0, 0, 1, 0]])
        label_3D['P0_mat'] = P0

        x = np.array(bboxes_3d[object_id]['x'])
        y = np.array(bboxes_3d[object_id]['y'])
        z = np.array(bboxes_3d[object_id]['z'])

        l = np.linalg.norm(x)
        label_3D['l'] = l
        w = np.linalg.norm(y)
        # label_3D['w'] = 0.85*w # NOTE! 7dLabs 3dbboxes also contain the side mirrors
        label_3D['w'] = w # NOTE! 7dLabs 3dbboxes also contain the side mirrors
        h = np.linalg.norm(z)
        label_3D['h'] = h

        center_vehicle_coords = np.array(bboxes_3d[object_id]['origin']) + x/2.0 + y/2.0 # (NOTE! center of bbox, at the BOTTOM of the bbox)
        center_vehicle_coords_hom = np.ones((4,))
        center_vehicle_coords_hom[0:3] = center_vehicle_coords
        # transform from vehicle coords to camera SENSOR coords (translation and rotation):
        center_camera_sensor_coords_hom = np.dot(np.linalg.inv(np.dot(R, T)), center_vehicle_coords_hom)
        center_camera_sensor_coords = np.zeros((3,))
        center_camera_sensor_coords[0] = center_camera_sensor_coords_hom[0]/center_camera_sensor_coords_hom[3]
        center_camera_sensor_coords[1] = center_camera_sensor_coords_hom[1]/center_camera_sensor_coords_hom[3]
        center_camera_sensor_coords[2] = center_camera_sensor_coords_hom[2]/center_camera_sensor_coords_hom[3]
        # transform from camera SENSOR coords to camera coords (new_x = -y, new_y = -z, new_z = x):
        center_camera_coords = np.roll(center_camera_sensor_coords, (0, 2), axis=0)
        center_camera_coords[0] = -center_camera_coords[0]
        center_camera_coords[1] = -center_camera_coords[1]
        label_3D['center'] = center_camera_coords.astype(np.float32)

        r_z_vehicle_coords = np.arctan2(x[1], x[0]) # (arctan2(x_y, x_x))
        r_y_camera_coords = wrapToPi(-(r_z_vehicle_coords + np.pi/2))
        label_3D['r_y'] = r_y_camera_coords

        location = center_camera_coords # (NOTE! center of bbox, at the BOTTOM of the bbox)
        r_y = r_y_camera_coords
        Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                           [-math.sin(r_y), 0, math.cos(r_y)]],
                          dtype='float32')
        p0 = np.dot(Rmat, np.asarray([l/2.0, 0, 0.85*w/2.0], dtype='float32'))
        p1 = np.dot(Rmat, np.asarray([-l/2.0, 0, 0.85*w/2.0], dtype='float32'))
        p2 = np.dot(Rmat, np.asarray([-l/2.0, 0, -0.85*w/2.0], dtype='float32'))
        p3 = np.dot(Rmat, np.asarray([l/2.0, 0, -0.85*w/2.0], dtype='float32'))
        p4 = np.dot(Rmat, np.asarray([l/2.0, -h, 0.85*w/2.0], dtype='float32'))
        p5 = np.dot(Rmat, np.asarray([-l/2.0, -h, 0.85*w/2.0], dtype='float32'))
        p6 = np.dot(Rmat, np.asarray([-l/2.0, -h, -0.85*w/2.0], dtype='float32'))
        p7 = np.dot(Rmat, np.asarray([l/2.0, -h, -0.85*w/2.0], dtype='float32'))
        label_3D['points'] = np.array(location + [p0, p1, p2, p3, p4, p5, p6, p7])
        label_3D['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1], [0, 1], [2, 3], [6, 7], [4, 5]]

        label["label_3D"] = label_3D

        labels.append(label)

    return labels

def eulerAnglesToRotationMatrix(theta):
    Rx = np.array([[1,         0,                  0],
                   [0,         math.cos(theta[0]), -math.sin(theta[0])],
                   [0,         math.sin(theta[0]), math.cos(theta[0])]
                   ])

    Ry = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                   [0,                     1,      0],
                   [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                   ])

    Rz = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                   [math.sin(theta[2]),    math.cos(theta[2]),     0],
                   [0,                     0,                      1]
                   ])

    R = np.dot(Rz, np.dot(Ry, Rx))
    R4 = np.identity(4)
    R4[:3, :3] = R

    return np.asarray(R4)

def translationMatrix(t):
    M = np.identity(4)
    M[:3, 3] = t[:3]

    return np.asarray(M)
