import numpy as np
import copy
from scipy.spatial import distance
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, single_state_size, init_pose=[0.,0., 0., 0., 0., 0.], init_velocities=[0.,0., 0.],
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

        self.state_size = self.action_repeat * single_state_size
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
        if abs(self.sim.pose[0] - self.target_pos[0]) > 10 or abs(self.sim.pose[1] - self.target_pos[1]) > 10:
            reward = 0
        elif 28. <= current_z <= 32.:
            reward = 10000
        elif self.previous_z < 28. and current_z < 28. and self.previous_z < current_z:
            reward = 1
        elif self.previous_z > 32. and current_z > 32. and self.previous_z > current_z:
            reward = 1

        self.previous_z = self.sim.pose[2]
        return reward * np.prod(self.get_all_discounts())
    
    def x_velocity_discount(self):
        # linear function f(abs(target_x - x)) = expected_x_velocity
        expected_vx = abs(self.target_pos[0] - self.sim.pose[0])
        expected_vx = expected_vx if self.sim.pose[0] < self.target_pos[0] else -expected_vx
        vx = self.sim.v[0]
        diff = abs(expected_vx - vx)
        discount = 1. - self.normalize(diff, 0., 30.)
        
        return discount
    
    def y_velocity_discount(self):
        # linear function f(abs(target_y - y)) = expected_y_velocity
        expected_vy = abs(self.target_pos[1] - self.sim.pose[1])
        expected_vy = expected_vy if self.sim.pose[1] < self.target_pos[1] else -expected_vy
        vy = self.sim.v[1]
        diff = abs(expected_vy - vy)
        discount = 1. - self.normalize(diff, 0., 30.)
        
        return discount
    
    def z_velocity_discount(self):
        # linear function f(abs(target_z - z)) = expected_z_velocity
        expected_vz = abs(self.target_pos[2] - self.sim.pose[2])
        expected_vz = expected_vz if self.sim.pose[2] < self.target_pos[2] else -expected_vz
        vz = self.sim.v[2]
        diff = abs(expected_vz - vz)
        discount = 1. - self.normalize(diff, 0., 30.)
        
        return discount
    
    def constant_discount(self, current_val, constant=0., min_val=0., max_val=30.):
        expected_val = constant
        diff = abs(expected_val - current_val)
        discount = 1. - self.normalize(diff, min_val, max_val)
 
        return discount
    
    def get_all_discounts(self):
        return [
            self.z_velocity_discount(),
            self.x_velocity_discount(),
            self.y_velocity_discount(),
            self.constant_discount(self.sim.pose[3], min_val=0., max_val=6.28319),
            self.constant_discount(self.sim.pose[4], min_val=0., max_val=6.28319),
            self.constant_discount(self.sim.pose[5], min_val=0., max_val=6.28319),
            self.constant_discount(self.sim.angular_v[0]),
            self.constant_discount(self.sim.angular_v[1]),
            self.constant_discount(self.sim.angular_v[2])
        ]
        
    def normalize(self, x, min_x, max_x):
        normalized_diff = (x-min_x)/(max_x-min_x)
        
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
        return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v, self.get_all_discounts()])
