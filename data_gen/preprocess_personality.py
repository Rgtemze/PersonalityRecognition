import sys

import numpy as np

sys.path.extend(['../'])

from tqdm import tqdm

from data_gen.rotation import *
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5.2))
pose_ax = fig.add_subplot(1, 1, 1, projection='3d')

def visualize(poses3d):
    edges = [[0, 3], [1, 0], [2, 0], [3, 6], [4, 1], [5, 2], [6, 9], [7, 4], [8, 5], [9, 9], [10, 7], [11, 8], [12, 9],
             [13, 9], [14, 9], [15, 12], [16, 13], [17, 14], [18, 16], [19, 17], [20, 18], [21, 19], [22, 20], [23, 21]]

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d in poses3d:
        # pose3d -= pose3d[0]
        for i_start, i_end in edges:
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=0)
        pose_ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], s=2)
        # for index, point in enumerate(pose3d):
        #     pose_ax.text(point[0], point[1], point[2], index)
    # fig.tight_layout()
    plt.draw()
    plt.pause(1000)

    pose_ax.clear()

def pre_normalization(data, zaxis=[0, 6], xaxis=[13, 14]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    # print('pad the null frames with the previous frames')
    # for i_s, skeleton in enumerate(tqdm(s)):  # pad
    #     if skeleton.sum() == 0:
    #         print(i_s, ' has no skeleton')
    #     for i_p, person in enumerate(skeleton):
    #         if person.sum() == 0:
    #             continue
    #         if person[0].sum() == 0:
    #             index = (person.sum(-1).sum(-1) != 0)
    #             tmp = person[index].copy()
    #             person *= 0
    #             person[:len(tmp)] = tmp
    #         for i_f, frame in enumerate(person):
    #             if frame.sum() == 0:
    #                 if person[i_f:].sum() == 0:
    #                     rest = len(person) - i_f
    #                     num = int(np.ceil(rest / i_f))
    #                     pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
    #                     s[i_s, i_p, i_f:] = pad
    #                     break

    print('sub the center joint #6 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 6:7, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 6) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

def preprocess(folder, type, op):
    input_name = f'D:/ziya/st-gcn/{folder}/X_{type if op == "skeleton" else type + "_processed"}.npy'
    output_name = f'D:/ziya/st-gcn/{folder}/X_{type}_{"processed" if op == "skeleton" else "laban"}.npy'
    print(f"Preprocessing {type} ({input_name}) in folder {folder}, doing {op}, will output {output_name}")
    data = np.load(input_name)
    if(len(data) == 0):
        print("Skipped")
        return

    if op == "skeleton":
        print(data.shape)
        data = np.expand_dims(data, 4)
        data = np.transpose(data, [0, 3, 1, 2, 4])
        data = pre_normalization(data)
        data = np.transpose(data, [0, 2, 3, 1, 4])
        data = np.squeeze(data, axis=4)
        print(data.shape)
    elif op == "laban":
        # OUR CONTRIBUTION
        info_data = np.load(f"D:/ziya/st-gcn/{folder}/info_{type}.npy")
        extended_info = np.zeros((0, 2))
        print(info_data[0])
        for info in info_data:
            info = np.tile(info[:2], (int(info[2]), 1))
            extended_info = np.vstack((extended_info, info))

        start = 0
        output = []
        for info in info_data:
            end = start + int(info[2])
            person_session = data[start:end]
            init_shape = person_session.shape
            assert person_session.shape[0] == int(info[2])
            person_session = person_session.reshape(init_shape[0] * init_shape[1], init_shape[2], init_shape[3])

            diff = np.diff(person_session, axis=1, prepend=0).reshape(init_shape)
            output.append(diff)
            start = end
        output = np.vstack(output)


        bone_pairs = [[0, 3], [1, 0], [2, 0], [3, 6], [4, 1], [5, 2], [6, 9], [7, 4], [8, 5], [9, 9], [10, 7], [11, 8],
                      [12, 9], [13, 9], [14, 9], [15, 12], [16, 13], [17, 14], [18, 16], [19, 17], [20, 18], [21, 19], [22, 20],
                      [23, 21]]
        flow = np.copy(data)

        X_VECTOR = np.array([1, 0, 0])
        Y_VECTOR = np.array([0, 1, 0])
        Z_VECTOR = np.array([0, 0, 1])

        space = np.copy(data)
        weight = np.copy(data)

        for v1, v2 in tqdm(bone_pairs):
            bone_vector = data[:, :, v1, :] - data[:, :, v2, :]
            if v1 != v2:
                bone_vector = bone_vector / np.sqrt(np.sum(np.square(bone_vector), keepdims=True, axis=-1))

            # bone_vector.shape == (1408, 1000, 3)
            angle_x = np.arccos(np.dot(bone_vector, X_VECTOR))[..., None]
            angle_y = np.arccos(np.dot(bone_vector, Y_VECTOR))[..., None]
            angle_z = np.arccos(np.dot(bone_vector, Z_VECTOR))[..., None]
            angle = np.dstack((angle_x, angle_y, angle_z))
            flow[:, :, v1, :] = angle
            space[:, :, v1, :] = angle
            weight[:, :, v1, :] = angle

        flow_d1 = np.diff(flow, axis=1, prepend=0)
        flow_d2 = np.diff(flow_d1, axis=1, prepend=0)
        print(flow_d2.shape)
        output = np.concatenate((output, flow_d2, space, weight), axis=-1)


        print(data.shape) # (1408, 1000, 24, 3)
        print(info_data.shape)
        print(extended_info.shape)
        print(output.shape)

        data = output
    else:
        raise NotImplementedError(f"The operation {op} is not implemented")

    np.save(output_name, data)

if __name__ == '__main__':
    folder = "body_keypoints_full_val"
    op = "laban"
    # op = "skeleton"
    for type in ["test", "train", "val"]:
        preprocess(folder, type, op)
