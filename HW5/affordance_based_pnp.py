import numpy as np
import argparse
import os
import pathlib
import torch
from itertools import chain
from matplotlib import pyplot as plt
from skimage.io import imsave
from collections import deque, defaultdict 
import json
from utils.common import load_chkpt, get_splits, draw_grasp
from env import UR5PickEnviornment
import affordance_model




############################################## 
# This script perform inference for affordance based pick and place 
# It assume that the affordance model is trained and saved

def main():
    # parse input 
    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('-t', '--task', default='train',
        help='which task to do: "train", "test", "empty_bin"')
    parser.add_argument('--headless', action='store_true',
        help='launch pybullet GUI or not')
    parser.add_argument('--seed', type=int, default=3,
        help='random seed for empty_bin task')
    parser.add_argument('--save_data', default=False, action='store_true',
        help='save data for additional self-improvement on the affordance model')
    parser.add_argument('--n_past_actions', default=0, type=int)
    args = parser.parse_args()

    # load model
    model_class = affordance_model.AffordanceModel
    model_dir = os.path.join('data/affordance')
    chkpt_path = os.path.join(model_dir, 'best.ckpt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(n_past_actions=args.n_past_actions)
    model.to(device)
    load_chkpt(model, chkpt_path, device)
    model.eval()

        # load env
    env = UR5PickEnviornment(gui=not args.headless)
    
    if args.task == 'train' or args.task == 'test':
        names = get_splits()[args.task]
        n_attempts = 3
        vis_dir = os.path.join(model_dir, 'eval_pick_'+ args.task +'ing_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        results = list()
        for name_idx, name in enumerate(names):
            print('Picking: {}'.format(name))
            env.remove_objects()
            for i in range(n_attempts):
                print('Attempt: {}'.format(i))
                seed = name_idx * 100 + i + 10000
                if i == 0:
                    env.load_ycb_objects([name], seed=seed)
                else:
                    env.reset_objects(seed)
                
                rgb_obs, depth_obs, _ = env.observe()
                coord, angle, vis_img = model.predict_grasp(rgb_obs)
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                print('Success!' if result else 'Failed:(')
                fname = os.path.join(vis_dir, '{}_{}.png'.format(name, i))
                imsave(fname, vis_img)
                results.append(result)
        success_rate = np.array(results, dtype=np.float32).mean()
        print("Testing on {} objects. Success rate: {}".format(args.task,success_rate))
    else:
        names = list(chain(*get_splits().values()))
        n_attempts = 25
        vis_dir = os.path.join(model_dir, 'eval_empty_bin_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        print("Loading objects.")
        env.remove_objects()
        env.load_ycb_objects(names, seed=args.seed)
        n_objects = len(names)
        num_in = env.num_object_in_tote1()
        print("{}/{} objects moved".format(n_objects - num_in, n_objects))

        for attempt_id in range(n_attempts):
            print("Attempt {}".format(attempt_id))
            rgb_obs, depth_obs, _ = env.observe()
            coord, angle, vis_img = model.predict_grasp(rgb_obs)
            pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
            result = env.execute_grasp(*pick_pose)
            if result:
                # place
                env.execute_place()
            
            num_in = env.num_object_in_tote1()
            print("{}/{} objects moved".format(n_objects - num_in, n_objects))

            fname = os.path.join(vis_dir, '{}.png'.format(attempt_id))
            imsave(fname, vis_img)
        print("{} objects left in the bin.".format(num_in))

if __name__ == '__main__':
    main()