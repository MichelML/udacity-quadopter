import numpy as np
import copy
from scipy.spatial import distance
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=[0.,0., 0., 0., 0., 0.], init_velocities=[0.,0., 0.],
                 init_angle_velocities=[0.,0., 0.], runtime=5., target_pos=[0., 0., 200., 0., 0., 0.]):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Episodes
        self.num_episodes = 0

        # Goal
        self.target_pos = target_pos

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * (len(init_pose) + len(init_velocities) + len(init_angle_velocities))
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.action_range = self.action_high - self.action_low
        
        # Score
        self.score = -10000.
        self.best_score = -10000.
        
        # Position
        self.prev_pos_diff = 100000
        self.prev_angle_pos_diff = 100000

    def get_dist_between_3d_points(self, pos, target):
        return distance.cdist([pos], [target])[0][0]

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward.""" 
        dist_between_pos_and_target = self.get_dist_between_3d_points(self.sim.pose[:3], self.target_pos[:3])
        reward_pos = 0
        if dist_between_pos_and_target < 1:
            reward_pos = 100
        elif dist_between_pos_and_target < self.prev_pos_diff:
            reward_pos = 3
        else:
            reward_pos = -3
            
        angles_dist_of_target = self.get_dist_between_3d_points(self.sim.pose[3:], self.target_pos[3:])
        reward_angles = 0
        if angles_dist_of_target < 0.03:
            reward_angles = 1
        else:
            reward_angles = -1
            
        self.prev_pos_diff = copy.copy(dist_between_pos_and_target)
        self.prev_angle_pos_diff = copy.copy(angles_dist_of_target)
        return reward_pos + reward_angles

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward(rotor_speeds)
            self.score += reward
            pose_all.append(self.get_sim_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        if self.score > self.best_score:
            self.best_score = copy.copy(self.score)
        self.score = 0
        self.num_episodes += 1
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.get_sim_state()] * self.action_repeat)
        return state
    
    def get_sim_state(self):
        return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v])
