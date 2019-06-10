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
        self.num_episode = 0

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
        self.best_score_episode = 0
        
        # Position
        self.previous_z = self.sim.pose[2]
        
    def get_dist_between_points(self, pos, target):
        return distance.cdist([pos], [target])[0][0]

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward.""" 
        dist_from_target_xy = self.get_dist_between_points(self.sim.pose[:2], self.target_pos[:2])
        current_z = self.sim.pose[2]
        
        reward = 0
        if dist_from_target_xy > 10:
            reward = 0
        elif 95. < current_z < 100.:
            reward = 10000
        elif self.previous_z < 95 and current_z < 95 and self.previous_z < current_z:
            reward = 1
        elif self.previous_z > 100 and current_z > 100 and self.previous_z > current_z:
            reward = 1

        self.previous_z = self.sim.pose[2]
        return reward * self.z_velocity_discount(current_z)
      
    def z_velocity_discount(self, z_position, c=60.,a=5.,b=1.05,d=-30., h=132.96):
        # https://www.desmos.com/calculator/o1etf6qg3l
        expected_z_velocity = c/(1+(a*b**(z_position - h))) + d
        diff_from_expected_z_velocity = abs(expected_z_velocity - self.sim.v[2])
        discount = 1. - self.normalize_z_velocity_diff(diff_from_expected_z_velocity)
        
        return discount
        
    def normalize_z_velocity_diff(self, diff):
        min_diff = 0.
        max_diff = 30.
        normalized_diff = (diff-min_diff)/(max_diff-min_diff)
        
        return normalized_diff if normalized_diff <= 1. else 1.

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
            self.best_score_episode = self.num_episode
            self.best_score = copy.copy(self.score)
        self.score = 0
        self.num_episode += 1
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.get_sim_state()] * self.action_repeat)
        return state
    
    def get_sim_state(self):
        return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v])
