import numpy as np
import gurobipy
import cPickle
import trajoptpy
from trajoptpy import math_utils as mu
import openravepy
from joblib import Memory, Parallel, delayed
import h5py
from rapprentice import registration, conversions, transformations, ros2rave


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

@mem.cache
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

CANONICAL_LEN = 50

def load_data():
    print 'reading demo file...'
    h5file = '/media/3tb/demos/master.h5'
    demofile = h5py.File(h5file, 'r')
    print 'done'
    
    print 'reading func matrix...'
    with open('/media/3tb/demos/func_matrix.pkl', 'r') as f:
        func_mat = cPickle.loads(f.read())
    print 'done'

    print 'reading distance matrices...'
    DIST_MAT_FILES = ['hist_dist_mat.pkl', 'tps_dist_mat.pkl', 'tps_t_dist_mat.pkl', 'sc_dist_mat.pkl', 'sc_t_dist_mat.pkl', 'ffinv_dist_mat.pkl']
    dist_mats = []
    for name in DIST_MAT_FILES:
        with open('/media/3tb/demos/' + name, 'r') as f:
            dist_mats.append(cPickle.loads(f.read()))
    dist_mats = np.asarray(dist_mats)
    print 'done'
    
    return demofile, func_mat, dist_mats

@mem.cache(ignore=['robot'])
def plan_follow_traj(robot, manip_name, ee_linkname, new_hmats, old_traj, eval_costs_only=False):
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    joint_vel_coeff = 1
    collision_coeff = 10
    pose_coeff = 20
    
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
    def __init__(self, robot):
        self.robot = robot
        self.demofile, self.func_mat, self.dist_mats = load_data()
        self.demo_keys = self.demofile.keys()
        self.dim_discrete = len(self.dist_mats)

    def extract_continuous(self, traj, demo_j, demo_i, force_identity_reg=False):
        '''features of continuous part (i.e. unweighted trajopt costs) for traj when registering demo_j to demo_i'''
        # for this set of demos, the right arm is always used, so compare trajectories of only the right arm

        frame = 'r_gripper_tool_frame'
        manip_name = 'rightarm'

        assert len(traj) == CANONICAL_LEN

        with self.robot:
            f = self.func_mat[demo_j][demo_i]

            seg_j = self.demofile[self.demo_keys[demo_j]]
            #joint_traj_j = np.asarray(seg_j[manip_name])
            ee_traj_j = np.asarray(seg_j[frame]["hmat"])
            if force_identity_reg:
                ee_traj_i = resample_hmats(ee_traj_j, CANONICAL_LEN)
            else:
                ee_traj_i = f.transform_hmats(resample_hmats(ee_traj_j, CANONICAL_LEN))

            # set initial joint positions, because something like the torso will mess things up
            init_joint_names = seg_j["joint_states"]["name"]
            init_joint_vals = seg_j["joint_states"]["position"]
            r2r = ros2rave.RosToRave(self.robot, init_joint_names)
            r2r.set_values(self.robot, init_joint_vals[0])
            traj_costs = plan_follow_traj(self.robot, manip_name, frame, ee_traj_i, traj, eval_costs_only=True)

        cont_features = -traj_costs

        return cont_features
    
    def extract_discrete(self, i, j):
        return self.dist_mats[:,i,j]

    def extract(self, traj, i, j):
        return np.r_[self.extract_continuous(traj, i, j), self.extract_discrete(i, j)]

    def get_demo_traj(self, i, arm='rightarm'):
        return resample_traj(np.asarray(self.demofile[self.demo_keys[i]][arm]), CANONICAL_LEN)

    def get_num_demos(self):
        return 20#len(self.demo_keys)

    def get_num_cont_features(self):
        return 3
    
    def get_num_disc_features(self):
        return len(self.dist_mats)

    def get_num_features(self):
        return self.get_num_cont_features() + self.get_num_disc_features()

class SSVM(object):
    def __init__(self, extractor, loss_func, c=1):
        '''
        SVM for structured prediction
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

        # weights on features
        self.var_w = []
        for i in range(self.ex.get_num_features()):
            self.var_w.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_%d' % i))

#         # weights for features on continuous part
#         self.dim_c = self.ex.get_num_cont_features()
#         self.var_w_c = []
#         for i in range(self.dim_c):
#             self.var_w_c.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_c_%d' % i))
#         
#         # weights for features on discrete part
#         self.dim_d = self.ex.get_num_disc_features()
#         self.var_w_d = []
#         for i in range(self.dim_d):
#             self.var_w_d.append(self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='w_d_%d' % i))
        
        self.model.update()

    def add_traj_sample(self, traj):
        traj = resample_traj(traj, CANONICAL_LEN)

        import progress.bar
        bar = progress.bar.Bar('Adding constraints for single trajectory', max=self.num_demos*self.num_demos)

        for i in range(self.num_demos):
            loss = self.loss_func(traj, self.ex.get_demo_traj(i))
#             disc_feature_sum = sum(self.ex.extract_discrete(k, i) for k in range(self.num_demos) if k != i)

            for j in range(self.num_demos):
                #val = loss + dot(self.var_w_c, self.ex.extract_continuous(traj, j, i)) + dot(self.var_w_d, self.ex.extract_discrete(j, i))
                val = loss + dot(self.var_w, self.ex.extract(traj, j, i))                
                self.model.addConstr(self.var_slack[i] >= val)
                
                bar.next()

#             for j in range(self.num_demos):
#                 if j == i: continue
#                 
#                 val = loss
#                 
#                 val += dot(self.var_w_c, self.ex.extract_continuous(self.ex.get_demo_traj(i), j, i)) # TODO: cache these
#                 val -= dot(self.var_w_c, self.ex.extract_continuous(traj, j, i))
#                 
#                 val += 1./(self.num_demos-1.) * dot(self.var_w_d, disc_feature_sum - self.ex.extract_discrete(j, i))
#                 val -= dot(self.var_w_d, self.ex.extract_discrete(j, i)) # TODO: cache these
#                 
#                 self.model.addConstr(self.var_slack[i] >= val)
        bar.finish()

    def _convex_step(self, concave_ub_features):
        objective = sum_squares(self.var_w) + self.c*sum(self.var_slack) - self.c*dot(self.var_w, concave_ub_features.sum(axis=0).tolist())
        self.model.setObjective(objective)
        self.model.optimize()
        return np.asarray([w_i.x for w_i in self.var_w])

    def _concave_step(self, w):
        out_best_features = np.empty((self.ex.get_num_demos(), self.ex.get_num_features()))
        out_best_j = np.empty((self.ex.get_num_demos()))
        for i in range(self.ex.get_num_demos()):
            best_score, best_j, best_features = -float('inf'), 0, None
            for j in range(self.ex.get_num_demos()):
                features = self.ex.extract(self.ex.get_demo_traj(i), j, i)
                score = w.dot(features)
                if score > best_score:
                    best_score, best_j, best_features = score, j, features
            out_best_features[i] = best_features
            out_best_j[i] = best_j
        return out_best_j, out_best_features

    def cccp(self, w_init, term_tol=1e-5):
        num_iter = 10
        w = w_init
        prev_obj_val = None
        for t in range(num_iter):
            _, ub_features = self._concave_step(w)
            w = self._convex_step(ub_features)
            print t, self.model.objVal, w
            if prev_obj_val is not None and abs(prev_obj_val - self.model.objVal) < term_tol:
                print 'converged'
                break
            prev_obj_val = self.model.objVal
        return w

def loss_func(traj1, traj2):
    if len(traj1) != len(traj2):
        print 'dude are you sure'
        assert False
#     if len(traj1) > len(traj2):
#         traj1, traj2 = traj2, traj1
#     if len(traj1) < len(traj2):
#         traj1 = mu.interp2d(np.linspace(0, 1, len(traj2)), np.linspace(0, 1, len(traj1)), traj1)
    return ((traj1 - traj2)**2).sum().sum()


def run_tests(extractor):
    import unittest

    class Tests(unittest.TestCase):
        def setUp(self):
            #self.demofile, self.func_mat, self.dist_mats = load_data()
            self.saver = openravepy.RobotStateSaver(extractor.robot)

        def tearDown(self):
            self.saver.Restore()

        def test_demo_joints_and_hmats(self):
            '''make sure that setting 'rightarm' gives the right hmats, so that the pose costs are correct'''
            frame = 'r_gripper_tool_frame'
            manip_name = 'rightarm'

            manip = extractor.robot.GetManipulator(manip_name)
            link = extractor.robot.GetLink(frame)
            for i in range(extractor.get_num_demos()):
                seg = extractor.demofile[extractor.demo_keys[i]]
                ee_traj = np.asarray(seg[frame]["hmat"])
    
                # set initial joint positions, because something like the torso will mess things up
                init_joint_names = seg["joint_states"]["name"]
                init_joint_vals = seg["joint_states"]["position"]
                r2r = ros2rave.RosToRave(extractor.robot, init_joint_names)
                r2r.set_values(extractor.robot, init_joint_vals[0])

                for z in range(len(init_joint_vals)):
                    extractor.robot.SetDOFValues(seg[manip_name][z], manip.GetArmIndices())
                    self.assertTrue(np.allclose(link.GetTransform(), ee_traj[z], atol=1e-5))

        def test_trivial_costs(self):
            '''demo trajectories' pose costs when registering demos to themselves should be zero'''
            for i in range(extractor.get_num_demos()):
                features = extractor.extract_continuous(extractor.get_demo_traj(i), i, i, force_identity_reg=True)
                self.assertTrue(features[1] < .0001) # collision cost
                self.assertTrue(features[2] < .005)  # pose cost completely zero because of upsampling


    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)


def main():
    np.random.seed(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # set up robot (to calculate features of continuous part and for trajectory optimization)    
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]
    
    extractor = FeatureExtractor(robot)
    
    if args.test:
        run_tests(extractor)
        return

    #import IPython; IPython.embed()
    #for i in range(extractor.get_num_demos()):
    #    print extractor.extract_continuous(extractor.get_demo_traj(i, 'rightarm'), i, i)

    ssvm = SSVM(extractor, loss_func, c=1)
    
    # initial seeding trajectories
    curr_w = np.zeros(extractor.get_num_features())
    for i in range(extractor.get_num_demos()):
        ssvm.add_traj_sample(extractor.get_demo_traj(i))
        new_w = ssvm.cccp(curr_w)
        print 'w: %s -> %s' % (str(curr_w), str(new_w))
        curr_w = new_w
        raw_input('continue?')

if __name__ == '__main__':
    main()
