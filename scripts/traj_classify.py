import numpy as np
from sklearn import svm
import openravepy
import trajoptpy
import h5py
from rapprentice import registration

def arm_moved(joint_traj):    
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .05).any()

class Trajectory(object):
    def __init__(self, arms_used, joints):
        assert arms_used in ['r', 'l', 'lr']
        assert (arms_used in ['r', 'l'] and joints.shape[1] == 7) or (arms_used == 'lr' and joints.shape[1] == 14)
        self.arms_used = arms_used
        self.joints = joints

    def arm(self, arm):
        if self.arms_used == 'l' or self.arms_used == 'r':
            assert arm == self.arms_used
            return self.joints
        if self.arms_used == 'lr':
            if arm == 'l': return self.joints[:,:7]
            elif arm == 'r': return self.joints[:,7:]
        assert False

class Storage(object):
    def __init__(self, filename):
        self.f = h5py.File(filename, 'a')
 
    def close(self):
        self.f.close()

    def _group(self, root, name):
        try:
            return root[name]
        except KeyError:
            return root.create_group(name)

    def get_num_segs_stored(self):
        return len(self._group['data'])

    LABELS = [1, 0]

    def add_traj(self, seg, cloud_xyz, traj, features, label):
        #assert arms_used in ['r', 'l', 'lr']
        #assert (arms_used in ['r', 'l'] and traj.shape[1] == 7) or (arms_used == 'lr' and traj.shape[1] == 14)
        assert label in self.LABELS
        assert isinstance(traj, Trajectory)

        seg_group = self._group(self.f, seg)
        data_group = self._group(seg_group, 'data')

        name = '%05d' % len(data_group)
        assert name not in data_group

        traj_group = data_group.create_group(name)
        traj_group['arms_used'] = traj.arms_used
        traj_group['joints'] = traj.joints
        traj_group['cloud_xyz'] = cloud_xyz
        traj_group['features'] = features
        traj_group['label'] = label
        
        self.f.flush()


class Optimizer(object):
    def __init__(self, robot, manip_name, ee_linkname):
        self.robot = robot
        self.manip_name = manip_name
        self.ee_linkname = ee_linkname
        self.set_params(joint_vel_coeff=1, collision_coeff=10, pose_coeff=20, dist_pen=.005)

    def _run_trajopt(self, desired_hmats, traj, eval_costs_only=False):
        n_steps = len(desired_hmats)
        assert traj.shape[0] == n_steps
        assert traj.shape[1] == 7
        saver = openravepy.RobotStateSaver(self.robot)

        request = {
            "basic_info" : {
                "n_steps" : n_steps,
                "manip" : self.manip_name,
                "start_fixed" : False,
            },
            "costs" : [
                {
                    "type" : "joint_vel",
                    "params": {"coeffs" : [self.joint_vel_coeff]}
                },
                {
                    "type" : "collision",
                    "params" : {"coeffs" : [self.collision_coeff],"dist_pen" : [self.dist_pen]}
                }
            ],
            "constraints" : [],
            "init_info" : {
                "type":"given_traj",
                "data":[x.tolist() for x in traj]
            }
        }
        if eval_costs_only:
            request["basic_info"]["max_iter"] = 0
        poses = [openravepy.poseFromMatrix(hmat) for hmat in desired_hmats]
        for (i_step,pose) in enumerate(poses):
            request["costs"].append(
                {"type":"pose",
                 "params":{
                    "xyz":pose[4:7].tolist(),
                    "wxyz":pose[0:4].tolist(),
                    "link":self.ee_linkname,
                    "timestep":i_step,
                    "pos_coeffs":[self.pose_coeff] * 3,
                    "rot_coeffs":[self.pose_coeff] * 3,
                 }
                })
        import json
        prob = trajoptpy.ConstructProblem(json.dumps(request), self.robot.GetEnv())
        return trajoptpy.OptimizeProblem(prob)

    def set_params(self, joint_vel_coeff, collision_coeff, pose_coeff, dist_pen):
        self.joint_vel_coeff = joint_vel_coeff
        self.collision_coeff = collision_coeff
        self.pose_coeff = pose_coeff
        self.dist_pen = dist_pen

    def compute_costs(self, desired_hmats, traj):
        #import IPython; IPython.embed()
        n_steps = len(desired_hmats)
        raw_costs = self._run_trajopt(desired_hmats, traj, eval_costs_only=True).GetCosts()
        joint_vel_cost = [c[1] for c in raw_costs if c[0] == 'joint_vel'][0] / self.joint_vel_coeff / n_steps
        collision_cost = sum(c[1] for c in raw_costs if c[0].startswith('collision')) / self.collision_coeff / n_steps
        pose_cost = sum(c[1] for c in raw_costs if c[0].startswith('pose')) / self.pose_coeff / n_steps
        return np.array([joint_vel_cost, collision_cost, pose_cost])

    def optimize_traj(self, desired_hmats, init_traj):
        return self._run_trajopt(desired_hmats, init_traj).GetTraj()


# def train_on_trajs(traj_features, y):
#     X = traj_features.view((float, len(traj_features.dtype.names)))
#     clf = svm.SVC()
#     clf.fit(X, y)

def joints_to_hmats(manip, joints, link):
    hmats = np.empty((len(joints), 4, 4))
    robot = manip.GetRobot()
    with robot:
        for i in range(len(joints)):
            robot.SetDOFValues(joints[i], manip.GetArmIndices())
            hmats[i,:,:] = link.GetTransform()
    return hmats

class DemoReader(object):
    def __init__(self, demofile, robot):
        self.f = h5py.File(demofile, 'r')
        self.robot = robot
    
    def get_traj(self, seg, lr):
        if lr == 'l': return np.asarray(self.f[seg]['leftarm'])
        if lr == 'r': return np.asarray(self.f[seg]['rightarm'])
        assert False
    
    def get_hmats(self, seg, lr):
        link = '%s_gripper_tool_frame' % lr
#         hmats = np.asarray(self.f[seg][link]['hmat']).copy()
        # need to adjust hmats so that the trajectory in the demo actually matches them
        # TODO: better yet, just calculate them here?
#         with self.robot:
#             manip_name = {'l': 'leftarm', 'r': 'rightarm'}[lr]
#             self.robot.SetDOFValues(self.get_traj(seg, lr)[0], self.robot.GetManipulator(manip_name).GetArmIndices())
#             offset_xyz = self.robot.GetLink(link).GetTransform()[:3,3] - hmats[0,:3,3]
#         hmats[:,:3,3] += offset_xyz
        manip_name = {'l': 'leftarm', 'r': 'rightarm'}[lr]
        hmats = joints_to_hmats(self.robot.GetManipulator(manip_name), self.get_traj(seg, lr), self.robot.GetLink(link))
        return hmats

    def get_cloud(self, seg):
        return np.squeeze(self.f[seg]['cloud_xyz'])

    def get_seg_names(self):
        return self.f.keys()

def extract_features(robot, demos, seg, cloud_xyz, traj):
    assert isinstance(traj, Trajectory)
    
    # how well does traj execute seg warped onto the cloud?
    # cloud_xyz must be in the robot frame (same as traj) for pose costs to be meaningful
    old_xyz = demos.get_cloud(seg)
    f = registration.tps_rpm(old_xyz, cloud_xyz)
    def trajopt_costs(lr):
        ee_linkname = '%s_gripper_tool_frame' % lr
        manip_name = {'l': 'leftarm', 'r': 'rightarm'}[lr]
        old_ee_traj = demos.get_hmats(seg, lr)
        new_ee_traj = f.transform_hmats(old_ee_traj)
        return Optimizer(robot, manip_name, ee_linkname).compute_costs(new_ee_traj, traj.arm(lr))
    joint_vel_feature, collision_feature, pose_feature = sum(trajopt_costs(arm) for arm in traj.arms_used) / float(len(traj.arms_used))

    return np.array([joint_vel_feature, collision_feature, pose_feature])

def add_traj_from_demo(robot, storage, demos, name, label):
    left_traj, right_traj = demos.get_traj(name, 'l'), demos.get_traj(name, 'r')
    uses_left, uses_right = arm_moved(left_traj), arm_moved(right_traj)
    if uses_left and uses_right:
        arms_used = 'lr'
        joints = np.c_[left_traj, right_traj]
    elif uses_left:
        arms_used = 'l'
        joints = left_traj
    elif uses_right:
        arms_used = 'r'
        joints = right_traj
    else:
        assert False
    traj = Trajectory(arms_used, joints)

    demo_xyz = demos.get_cloud(name)
    features = extract_features(robot, demos, name, demo_xyz, traj)
    storage.add_traj(name, demo_xyz, traj, features, label)

def load_env():
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]
    return env, robot

def run_tests():
    import unittest

    env, robot = load_env()

    class Tests(unittest.TestCase):
        def setUp(self):
            self.saver = openravepy.RobotStateSaver(robot)

        def tearDown(self):
            self.saver.Restore()

        def test_optimizer_compute_costs(self):
            o = Optimizer(robot, 'rightarm', 'r_gripper_tool_frame')

            init_dof_values = [-97.425923483496987, 464.68215439740209, 364.2127375116857, -122.55840983540598, 374.59509707936593, 219.65377697890753, -106.85014449124387, 308.53660745910383, 173.93009539836365, -62.868508132713423, 600.17696877273204, 523.0537199714023, 0.17424607120185159, 0.00027344572577769205, 0.90205597060074949, 0.5407465122740126, -1.4985295621409396, -0.82190631049865959, -0.18803799472951907, -39.16134936538657, -1.7636793534458424, -1.1272691487861763, 22.486141306502695, 0.080543659528626105, 0.46657147587005277, 1.029090016690251, 1.1624587490173848, -0.28572512142484785, 14.254065096905455, -0.76378198372612238, -1.8433090146275135, 128.11829553312387, 0.080721515931221996, 0.46769791858209897]
            init_dof_inds = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14, 26, 29, 27, 28, 31, 30, 32, 33, 37, 34, 17, 15, 16, 19, 18, 20, 21, 25, 22]
            robot.SetDOFValues(init_dof_values[:20], init_dof_inds[:20])
            robot.SetDOFValues(init_dof_values[20:], init_dof_inds[20:])

            traj = np.array(
              [[ -0.82190631,  -0.18803799,  -1.49852956,  -1.76367935, -39.16134937,  -1.12726915,  22.48614131],
               [ -0.82124306,  -0.18719205,  -1.49852956,  -1.76367935, -39.16134937,  -1.12726915,  22.48614131],
               [ -0.77755127,  -0.18017067,  -1.50783014,  -1.76773293, -39.16134937,  -1.12726915,  22.48614131],
               [ -0.7239936 ,  -0.15682247,  -1.5694064 ,  -1.77569532, -39.16551436,  -1.1260509 ,  22.48492306],
               [ -0.66745128,  -0.13347427,  -1.67171277,  -1.77062835, -39.25511949,  -1.09916239,  22.45803454]])
            desired_hmats = np.array(
              [[[ 0.3324868 , -0.48247268,  0.8103534 ,  0.53156346],
                [ 0.32705275,  0.864903  ,  0.38076147, -0.22886698],
                [-0.8845841 ,  0.13843015,  0.44536285,  0.89131299],
                [ 0.        ,  0.        ,  0.        ,  1.        ]],
               [[ 0.33175998, -0.48296621,  0.8103573 ,  0.53154793],
                [ 0.32782096,  0.86449684,  0.38102314, -0.22843558],
                [-0.88457263,  0.13924388,  0.4451319 ,  0.89103734],
                [ 0.        ,  0.        ,  0.        ,  1.        ]],
               [[ 0.30593489, -0.52052333,  0.79715702,  0.53060893],
                [ 0.34066035,  0.841718  ,  0.41888105, -0.20317839],
                [-0.88901877,  0.14340946,  0.43483256,  0.88480923],
                [ 0.        ,  0.        ,  0.        ,  1.        ]],
               [[ 0.23921815, -0.55597431,  0.79603219,  0.52376051],
                [ 0.32687318,  0.81809491,  0.47315393, -0.17887405],
                [-0.91429131,  0.14701457,  0.37743624,  0.85305163],
                [ 0.        ,  0.        ,  0.        ,  1.        ]],
               [[ 0.23410816, -0.6143965 ,  0.75346553,  0.529508  ],
                [ 0.26201516,  0.78619892,  0.55967787, -0.16580832],
                [-0.93623791,  0.06639423,  0.34503678,  0.80973606],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]])

            joint_vel_cost, collision_cost, pose_cost = o.compute_costs(desired_hmats, traj)
            true_joint_vel_cost = ((traj[1:] - traj[:-1])**2).sum().sum() / len(traj)
            self.assertAlmostEqual(joint_vel_cost, true_joint_vel_cost)
            self.assertAlmostEqual(collision_cost, 0.)
            self.assertAlmostEqual(pose_cost, 0.)
            o.set_params(999, 999, 999, 999) # cost should be invariant to these weights
            self.assertAlmostEqual(joint_vel_cost, true_joint_vel_cost)
            self.assertAlmostEqual(collision_cost, 0.)
            self.assertAlmostEqual(pose_cost, 0.)

    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demos', type=str)
    parser.add_argument('--storage', type=str)
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    env, robot = load_env()
    storage = Storage(args.storage)
    demos = DemoReader(args.demos, robot)
    for i, seg in enumerate(demos.get_seg_names()):
        print '[%d/%d]' % (i+1, len(demos.get_seg_names()))
        label = 1
        add_traj_from_demo(robot, storage, demos, seg, label)

if __name__ == '__main__':
    main()
