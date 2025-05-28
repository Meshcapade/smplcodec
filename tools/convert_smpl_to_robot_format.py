import os
import numpy as np
import torch

from smplcodec import SMPLCodec
from smplx import SMPLX


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def smpl_to_robot_dict(smplcdc: SMPLCodec,
                       model_path: str):
    '''
    Convert standard SMPLCodec data to "robot-friendly" data (dict containing numpy arrays)
    '''
        
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    num_betas = smplcdc.shape_parameters.size
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, smplcdc.gender.name.lower(), 'SMPLX_{}.npz'.format(smplcdc.gender.name.lower()))

    model_params = dict(model_path=model_path,
                        model_type=smplcdc.smpl_version.name.lower(),
                        joint_mapper=None,
                        create_global_orient=False,
                        create_body_pose=True,
                        create_betas=False,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        num_betas=num_betas,
                        flat_hand_mean=True,
                        gender=smplcdc.gender.name.lower(),
                        dtype=dtype,
                        use_pca=False,
                        ext=model_path.split('.')[-1],
                        num_expression_coeffs=100,
                        encoding='latin1'
                        )
    
    model = SMPLX(**model_params).to(device=device)

    betas = torch.tensor(smplcdc.shape_parameters.reshape(1, -1), device=device, dtype=dtype)
    
    num_frames = smplcdc.body_pose.shape[0]
    num_joints = model.NUM_JOINTS + 1 # get_num_joints returns 1 joint less than there actually is?!? Not counting pelvis??
    
    body_position_array = np.zeros((num_frames, num_joints, 3))
    body_orient_array = np.zeros((num_frames, num_joints-1, 4)) # global orient separately...
    root_position_array = np.zeros((num_frames, 3))
    root_orient_array = np.zeros((num_frames, 4))
    # root_velocity_array = np.zeros((num_frames, 6))
    # foot_contact_array = np.zeros((num_frames, 4))
    
    default_head_pose = torch.zeros((1, 9), device=device, dtype=dtype)
    default_hand_pose = torch.zeros((1, 45), device=device, dtype=dtype)
    
    for i in range(num_frames):
        pose = torch.tensor(smplcdc.body_pose[i, 1:].reshape(1, -1), device=device, dtype=dtype)
        global_rot = torch.tensor(smplcdc.body_pose[i, 0].reshape(1, -1), device=device, dtype=dtype)
        transl = torch.tensor(smplcdc.body_translation[i].reshape(1, -1), device=device, dtype=dtype)
        head_pose = torch.tensor(smplcdc.head_pose[i].reshape(1, -1), device=device, dtype=dtype) if smplcdc.head_pose is not None else default_head_pose
        left_hand_pose = torch.tensor(smplcdc.left_hand_pose[i].reshape(1, -1), device=device, dtype=dtype) if smplcdc.left_hand_pose is not None else default_hand_pose
        right_hand_pose = torch.tensor(smplcdc.right_hand_pose[i].reshape(1, -1), device=device, dtype=dtype) if smplcdc.right_hand_pose is not None else default_hand_pose

        # run forward function on parameters to get joints
        model_output = model(body_pose=pose,
                             global_orient=global_rot,
                             betas=betas,
                             transl=transl,
                             jaw_pose=head_pose[:, :3] if head_pose is not None else None,
                             leye_pose=head_pose[:, 3:6] if head_pose is not None else None,
                             reye_pose=head_pose[:, 6:9] if head_pose is not None else None,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose)

        
        joints = model_output.joints[:, :num_joints] # smplx forward function adds vertices as joints, and we only use the "real" ones here
                
        # _Body Positions_: Shape: [number of frames, number of body parts, 3] (x, y, z coordinates for each keypoint such as arm, knee, ankle, shoulder, hip, etc.)
        body_position_array[i] = joints.detach().cpu().numpy().squeeze()
        
        # _Body Orientations_: Shape: [number of frames, number of body parts, 3 or 4](Either roll, pitch, yaw or quaternion values for each keypoint.)
        # convert to quaternions
        body_orient_array[i] = np.concatenate((axis_angle_to_quaternion(pose.reshape(-1,3)).detach().cpu().numpy().squeeze(),
                                               axis_angle_to_quaternion(head_pose.reshape(-1,3)).detach().cpu().numpy().squeeze(),
                                               axis_angle_to_quaternion(left_hand_pose.reshape(-1,3)).detach().cpu().numpy().squeeze(),
                                               axis_angle_to_quaternion(right_hand_pose.reshape(-1,3)).detach().cpu().numpy().squeeze()))
        
        # _Root Position_: Shape: [number of frames, 3](x, y, z position of the root.)
        root_position_array[i] = joints.detach().cpu().numpy().squeeze()[0]
        
        # _Root Orientation_: Shape: [number of frames, 3 or 4](Root orientation as roll, pitch, yaw or quaternion.)
        root_orient_array[i] = axis_angle_to_quaternion(global_rot.reshape(-1,3)).detach().cpu().numpy().squeeze()
                                              
        # _Feet Contact_: Shape: [number of frames, 2 or 4](2 or 1 information contact for each foot.)
        
        # _Root Velocity:_ Shape: [number of frames, 6](Linear and angular velocity of the root) (edited)
        # if i > 0:
        #     root_velocity_array[i] = np.concatenate(((root_position_array[i] - root_position_array[i-1]) * smplcdc.frame_rate,
        #                                             (root_orient_array[i] - root_orient_array[i-1]) * smplcdc.frame_rate))
    
    robot_pose_dict = {'body_positions': body_position_array,
                       'body_orientations': body_orient_array,
                       'root_position': root_position_array,
                       'root_orientation': root_orient_array,
                       'feet_contact': None,
                       'root_velocity': None
                       }
    
    return robot_pose_dict
    
    
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--anim_fn", type=str, required=True, help="Input .smpl animation file")
    parser.add_argument("--model_folder", type=str, required=True, help="Where to find the SMPLX model (folder containing male/SMPLX_male.npz, female/SMPLX_female.npz, neutral/SMPLX_neutral.npz)")
    parser.add_argument("--output_fn", type=str, help="Where to save output")

    args = vars(parser.parse_args())

    anim_fn = args.get('anim_fn')
    model_folder = args.get('model_folder')
    output_fn = args.get('output_fn')
    
    if os.path.exists(anim_fn):
        smplcdc = SMPLCodec.from_file(anim_fn)
        
        robot_dict = smpl_to_robot_dict(smplcdc=smplcdc,
                                        model_path=model_folder)
        
        if output_fn is None:
            output_fn = '{}.npz'.format(os.path.splitext(anim_fn)[0])
        if os.path.exists(output_fn):
            print('Output file {} already exists! Not saving!'.format(output_fn))
        else:
            np.savez(output_fn, **robot_dict)
            
            print('Saved converted data to {}.'.format(output_fn))