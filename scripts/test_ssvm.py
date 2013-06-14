import numpy as np
import gurobipy
import cPickle
import trajoptpy
import openravepy
from trajoptpy import math_utils as mu
from joblib import Memory, Parallel, delayed
import collections
import h5py
from rapprentice import registration, conversions, transformations

mem = Memory(cachedir='/tmp/joblib')

def dot(x, y):
    assert len(x) == len(y)
    return sum(i*j for (i, j) in zip(x, y))

def sum_squares(l):
    return sum(i*i for i in l)

def interp_hmat(hmat1, hmat2, frac):
    transs, rots = conversions.hmats_to_transs_rots([hmat1, hmat2])
    out_trans = (1.-frac)*transs[0] + frac*transs[1]
    out_rot = transformations.quaternion_slerp(rots[0], rots[1], frac)
    return conversions.trans_rot_to_hmat(out_trans, out_rot)

def sample_hmats(hmats, target_len):
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

    print 'pos', out[:,:3,3]

    return out

def find(tuple_list, k, cmp_fn=None):
    inds = []
    for i, (a, b) in enumerate(tuple_list):
        if (cmp_fn is None and a == k) or (cmp_fn is not None and cmp_fn(a, k)): inds.append(i)
    return i

class SSVM(object):
    def __init__(self, dim_continuous, dim_discrete, demo_trajs, loss_func, cont_feature_func, disc_feature_func, c=1):
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
        self.cont_feature_func = cont_feature_func
        self.disc_feature_func = disc_feature_func
        self.demo_trajs = demo_trajs
        #self.traj_samples = []
        
        # slack variables
        self.num_demos = len(demo_trajs)
        self.var_slack = []
        for i in range(self.num_demos):
            self.var_slack.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='slack_%d' % i))
            
        # weights for features on continuous part
        self.dim_c = dim_continuous
        self.var_w_c = []
        for i in range(self.dim_c):
            self.var_w_c.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_c_%d' % i))
        
        # weights for features on discrete part
        self.dim_d = dim_discrete
        self.var_w_d = []
        for i in range(self.dim_d):
            self.var_w_d.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_d_%d' % i))
        
        self.model.update()
        
        objective = sum_squares(self.var_w_c) + sum_squares(self.var_w_d) + self.c*sum(self.var_slack)
        self.model.setObjective(objective)

    def add_traj_sample(self, traj):
        #self.traj_samples.append(traj)
        
        for i in range(self.num_demos):
            loss = self.loss_func(traj, self.demo_trajs[i])
            disc_feature_sum = sum(self.disc_feature_func(k, i) for k in range(self.num_demos) if k != i)

            for j in range(self.num_demos):
                if j == i: continue
                
                val = loss
                
                val += dot(self.var_w_c, self.cont_feature_func(self.demo_trajs[i], j, i)) # TODO: cache these
                val -= dot(self.var_w_c, self.cont_feature_func(traj, j, i))
                
                val += 1./(self.num_demos-1.) * dot(self.var_w_d, disc_feature_sum - self.disc_feature_func(j, i))
                val -= dot(self.var_w_d, self.disc_feature_func(j, i)) # TODO: cache these
                
                self.model.addConstr(self.var_slack[i] >= val)

    def solve(self):
        self.model.optimize()
        return np.asarray([v.x for v in self.var_w_c]), np.asarray([v.x for v in self.var_w_d])


@mem.cache
def loss_func(traj1, traj2):
    if len(traj1) > len(traj2):
        traj1, traj2 = traj2, traj1
    if len(traj1) < len(traj2):
        traj1 = mu.interp2d(np.linspace(0, 1, len(traj2)), np.linspace(0, 1, len(traj1)), traj1)
    return ((traj1 - traj2)**2).sum().sum()


    
def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, eval_costs_only=False):
        
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    joint_vel_coeff = 1
    collision_coeff = 10
    pose_coeff = 20
    
    #arm_inds  = robot.GetManipulator(manip_name).GetArmIndices()

    ee_linkname = ee_link.GetName()
    
    init_traj = old_traj.copy()
    #init_traj[0] = robot.GetDOFValues(arm_inds)
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

    joint_vel_cost = find(raw_costs, 'joint_vel')
    
    
    if eval_costs_only:
        return result.GetCosts()
    traj = result.GetTraj()
    costs = result.GetCosts()
# 
#     saver = openravepy.RobotStateSaver(robot)
#     pos_errs = []
#     for i_step in xrange(1,n_steps):
#         row = traj[i_step]
#         robot.SetDOFValues(row, arm_inds)
#         tf = ee_link.GetTransform()
#         pos = tf[:3,3]
#         pos_err = np.linalg.norm(poses[i_step][4:7] - pos)
#         pos_errs.append(pos_err)
#     pos_errs = np.array(pos_errs)
#         
#     print "planned trajectory for %s. max position error: %.3f. all position errors: %s"%(manip_name, pos_errs.max(), pos_errs)
#     import IPython; IPython.embed()    
    return traj, costs    

def make_initial_ref_trajs():
    pass

def main():
    # set up robot (to calculate features of continuous part and for trajectory optimization)    
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]

    H5FILE = '/media/3tb/demos/master.h5'
    demofile = h5py.File(H5FILE, 'r')
    demo_idx2name = demofile.keys()
    # for this set of demos, the right arm is always used, so compare trajectories of only the right arm
    def extract_continuous_features(traj, demo_j, demo_i):
        '''features of continuous part (i.e. unweighted trajopt costs) for traj when registering demo_j to demo_i'''
        frame = 'r_gripper_tool_frame'
        manip_name = 'rightarm'
        
        seg_j = demofile[demo_idx2name[demo_j]]
        seg_i = demofile[demo_idx2name[demo_i]]
        cloud_j = np.squeeze(seg_j['cloud_xyz'])
        cloud_i = np.squeeze(seg_i['cloud_xyz'])
        
        f = registration.tps_rpm(cloud_j, cloud_i)
        #joint_traj_j = np.asarray(seg_j[manip_name])
        ee_traj_j = np.asarray(seg_j[frame]["hmat"])
        ee_traj_i = f.transform_hmats(ee_traj_j)
        
        costs = plan_follow_traj(robot, manip_name, robot.GetLink(frame), sample_hmats(ee_traj_i, len(traj)), traj, eval_costs_only=True)

    DIST_MAT_FILES = ['hist_dist_mat.pkl', 'tps_dist_mat.pkl', 'tps_t_dist_mat.pkl', 'sc_dist_mat.pkl', 'sc_t_dist_mat.pkl', 'ffinv_dist_mat.pkl']
    dist_mats = []
    for name in DIST_MAT_FILES:
        with open('/media/3tb/demos/' + name, 'r') as f:
            dist_mats.append(cPickle.load(f))
    dim_discrete = len(dist_mats)
    dist_mats = np.asarray(dist_mats)
    def extract_discrete_features(i, j):
        #return np.asarray([mat[i,j] for mat in dist_mats.itervalues()])
        return dist_mats[:,i,j]

    ssvm = SSVM(dim_continuous, dim_discrete, num_demos, loss_func, cont_feature_func, extract_discrete_features, c=1)
    ssvm.add_traj_sample(traj)
    print ssvm.solve()

if __name__ == '__main__':
    main()
