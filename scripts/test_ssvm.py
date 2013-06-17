import numpy as np
import gurobipy
import cPickle
import trajoptpy
from trajoptpy import math_utils as mu
import openravepy
from joblib import Memory, Parallel, delayed
import h5py
from rapprentice import registration, conversions, transformations
import func_matrix_service as fms


mem = Memory(cachedir='/tmp/joblib')

def dot(x, y):
    assert len(x) == len(y)
    return sum(i*j for (i, j) in zip(x, y))

def sum_squares(l):
    return sum(i*i for i in l)

def find(tuple_list, k, cmp_fn=None):
    out = []
    for a, b in tuple_list:
        if (cmp_fn is None and a == k) or (cmp_fn is not None and cmp_fn(a)):
            out.append((a, b))
    return out

def interp_hmat(hmat1, hmat2, frac):
    '''
    interpolate between two rigid transformations
    frac = 0 gives hmat1, frac = 1 gives hmat 2
    positions interpolated linearly, rotations by slerp
    '''
    transs, rots = conversions.hmats_to_transs_rots([hmat1, hmat2])
    out_trans = (1.-frac)*transs[0] + frac*transs[1]
    out_rot = transformations.quaternion_slerp(rots[0], rots[1], frac)
    return conversions.trans_rot_to_hmat(out_trans, out_rot)

def resample_hmats(hmats, target_len):
    '''
    given a list of hmats (or Nx4x4 array) that represent poses evenly spaced out in time,
    produces a new list of target_len hmats that interpolate those poses over the same time interval
    '''
    if len(hmats) == target_len:
        return hmats
    
    out = np.empty((target_len, 4, 4))
    frac_inds = (len(hmats) - 1) * np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(hmats)), np.linspace(0, 1, len(hmats)))
    for idx, i in enumerate(frac_inds[:-1]):
        low_int_ind = int(np.floor(i))
        hi_int_ind = low_int_ind + 1
        frac = i - low_int_ind
        out[idx,:,:] = interp_hmat(hmats[low_int_ind], hmats[hi_int_ind], frac)
    out[-1,:,:] = hmats[-1]
    return out

def resample_traj(traj, target_len):
    return mu.interp2d(np.linspace(0, 1, target_len), np.linspace(0, 1, len(traj)), traj)

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, eval_costs_only=False):
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    joint_vel_coeff = 1
    collision_coeff = 10
    pose_coeff = 20

    ee_linkname = ee_link.GetName()
    
    init_traj = old_traj.copy()
    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : False,
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [joint_vel_coeff]}
        },
        {
            "type" : "collision",
            "params" : {"coeffs" : [collision_coeff],"dist_pen" : [0.005]}
        }                
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }
    if eval_costs_only:
        request["basic_info"]["max_iter"] = 0
        
    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
    for (i_step,pose) in enumerate(poses):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":ee_linkname,
                "timestep":i_step,
                "pos_coeffs":[pose_coeff,pose_coeff,pose_coeff],
                "rot_coeffs":[pose_coeff,pose_coeff,pose_coeff],
             }
            })

    import json
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, robot.GetEnv())
    result = trajoptpy.OptimizeProblem(prob)
    
    raw_costs = result.GetCosts()
    # turn returned costs into features
    # collapse/average collision costs and pose costs

    joint_vel_feature = find(raw_costs, 'joint_vel')[0][1] / joint_vel_coeff / n_steps
    collision_feature = sum(cost[1] for cost in find(raw_costs, None, lambda k: k.startswith('collision'))) / collision_coeff / n_steps
    pose_feature = sum(cost[1] for cost in find(raw_costs, None, lambda k: k.startswith('pose'))) / pose_coeff / n_steps
    feature_vec = np.array([joint_vel_feature, collision_feature, pose_feature])
    
    if eval_costs_only:
        return feature_vec

    return result.GetTraj(), feature_vec


class FeatureExtractor(object):
    def __init__(self, robot, h5file='/media/3tb/demos/master.h5'):
        self.robot = robot
        self.demofile = h5py.File(h5file, 'r')
        self.demo_keys = self.demofile.keys()
        self.func_mat_client = fms.Client()
        #import IPython; IPython.embed()
        self.func_mat_shape = self.func_mat_client.fetch_mat_shape()

        DIST_MAT_FILES = ['hist_dist_mat.pkl', 'tps_dist_mat.pkl', 'tps_t_dist_mat.pkl', 'sc_dist_mat.pkl', 'sc_t_dist_mat.pkl', 'ffinv_dist_mat.pkl']
        self.dist_mats = []
        for name in DIST_MAT_FILES:
            with open('/media/3tb/demos/' + name, 'r') as f:
                self.dist_mats.append(cPickle.load(f))
        self.dim_discrete = len(self.dist_mats)
        self.dist_mats = np.asarray(self.dist_mats)

    def extract_continuous(self, traj, demo_j, demo_i):
        '''features of continuous part (i.e. unweighted trajopt costs) for traj when registering demo_j to demo_i'''
        # for this set of demos, the right arm is always used, so compare trajectories of only the right arm

        frame = 'r_gripper_tool_frame'
        manip_name = 'rightarm'
        
        f = self.func_mat_client.lookup(demo_j, demo_i)
#         
        seg_j = self.demofile[self.demo_keys[demo_j]]
#         seg_i = demofile[demo_idx2name[demo_i]]
#         cloud_j = np.squeeze(seg_j['cloud_xyz'])
#         cloud_i = np.squeeze(seg_i['cloud_xyz'])
#         f = registration.tps_rpm(cloud_j, cloud_i)
        #joint_traj_j = np.asarray(seg_j[manip_name])
        ee_traj_j = np.asarray(seg_j[frame]["hmat"])
        ee_traj_i = f.transform_hmats(ee_traj_j)
        
        cont_features = plan_follow_traj(self.robot, manip_name, self.robot.GetLink(frame), resample_hmats(ee_traj_i, len(traj)), traj, eval_costs_only=True)
        return cont_features
    
    def extract_discrete(self, i, j):
        return self.dist_mats[:,i,j]

    def get_demo_traj(self, i, arm='rightarm'):
        return np.asarray(self.demofile[self.demo_keys[i]][arm])

    def get_num_demos(self):
        return len(self.demo_keys)

    def get_num_cont_features(self):
        return 3
    
    def get_num_disc_features(self):
        return len(self.dist_mats)

class SSVM(object):
    def __init__(self, extractor, loss_func, c=1):
        '''
        SVM for structured prediction
        arguments:
            dim_continuous, dim_discrete: Int (num of cont and discrete features)
            cont_feature_func : Traj x Demo x Demo -> R^{dim_continuous} : features for continuous part \phi^c(\tau, j | i)
            
            loss_func : Traj x Traj -> R
        '''
        self.model = gurobipy.Model('ssvm')
        self.c = float(c)
        self.loss_func = loss_func
        self.ex = extractor
        
        # slack variables
        self.num_demos = self.ex.get_num_demos()
        self.var_slack = []
        for i in range(self.num_demos):
            self.var_slack.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='slack_%d' % i))
            
        # weights for features on continuous part
        self.dim_c = self.ex.get_num_cont_features()
        self.var_w_c = []
        for i in range(self.dim_c):
            self.var_w_c.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_c_%d' % i))
        
        # weights for features on discrete part
        self.dim_d = self.ex.get_num_disc_features()
        self.var_w_d = []
        for i in range(self.dim_d):
            self.var_w_d.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_d_%d' % i))
        
        self.model.update()
        
        objective = sum_squares(self.var_w_c) + sum_squares(self.var_w_d) + self.c*sum(self.var_slack)
        self.model.setObjective(objective)

    def add_traj_sample(self, traj):        
        for i in range(self.num_demos):
            loss = self.loss_func(traj, self.ex.get_demo_traj(i))
            disc_feature_sum = sum(self.ex.disc_feature_func(k, i) for k in range(self.num_demos) if k != i)

            for j in range(self.num_demos):
                if j == i: continue
                
                val = loss
                
                val += dot(self.var_w_c, self.ex.cont_feature_func(self.ex.get_demo_traj(i), j, i)) # TODO: cache these
                val -= dot(self.var_w_c, self.ex.cont_feature_func(traj, j, i))
                
                val += 1./(self.num_demos-1.) * dot(self.var_w_d, disc_feature_sum - self.ex.disc_feature_func(j, i))
                val -= dot(self.var_w_d, self.ex.disc_feature_func(j, i)) # TODO: cache these
                
                self.model.addConstr(self.var_slack[i] >= val)

    def solve(self):
        self.model.optimize()
        return np.asarray([v.x for v in self.var_w_c]), np.asarray([v.x for v in self.var_w_d])


@mem.cache
def loss_func(traj1, traj2):
    if len(traj1) != len(traj2):
        print 'dude are you sure'
        assert False
#     if len(traj1) > len(traj2):
#         traj1, traj2 = traj2, traj1
#     if len(traj1) < len(traj2):
#         traj1 = mu.interp2d(np.linspace(0, 1, len(traj2)), np.linspace(0, 1, len(traj1)), traj1)
    return ((traj1 - traj2)**2).sum().sum()


def main():
    # set up robot (to calculate features of continuous part and for trajectory optimization)    
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]
    
    extractor = FeatureExtractor(robot)

    ssvm = SSVM(extractor, loss_func, c=1)
    ssvm.add_traj_sample(traj)
    print ssvm.solve()

if __name__ == '__main__':
    main()
