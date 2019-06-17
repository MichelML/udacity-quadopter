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
        self.init_pose = init_pose
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
        
    def get_dist_between_points(self, pos, target):
        return distance.cdist([pos], [target])[0][0]

    def get_net_reward(self, rotor_speeds):
        splitted_rewards, total_reward = self.get_rewards()
        splitted_discounts, total_discount = self.get_discounts()
        net_reward = max(total_reward - total_discount, splitted_rewards[1])

        return splitted_rewards, splitted_discounts, total_reward, total_discount, net_reward

    def get_rewards(self):
        max_points = self.target_pos[2]
        z_reward = max(max_points - self.get_dist_between_points(self.sim.pose[:3], self.target_pos[:3]), 0.)
        time_reward = .2
        total_reward = sum([z_reward, time_reward])
        
        return [z_reward, time_reward], total_reward

    def get_discounts(self):
        xy_displacement_discount = 0.005*self.get_dist_between_points(self.sim.pose[:2], self.target_pos[:2])
        euler_angles_discount = 0.01*self.get_dist_between_points(self.sim.pose[3:], self.target_pos[3:])
        total_discount = sum([xy_displacement_discount, euler_angles_discount])
        
        return [xy_displacement_discount, euler_angles_discount], total_discount

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        splitted_r = np.zeros(7)
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            splitted_rewards, splitted_discounts, total_reward, total_discount, net_reward = self.get_net_reward(rotor_speeds)
            splitted_r = splitted_r + np.concatenate([splitted_rewards, splitted_discounts, [total_reward, total_discount, net_reward]])
            reward += net_reward
            self.score += reward
            pose_all.append(self.get_sim_state(splitted_r))
        next_state = np.concatenate(pose_all)
        return next_state, splitted_r, done

    def reset(self):
        if self.score > self.best_score:
            self.best_score_episode = self.num_episode
            self.best_score = copy.copy(self.score)
        self.score = 0
        self.num_episode += 1
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.get_sim_state(np.zeros(7))] * self.action_repeat)
        return state
    
    def get_sim_state(self, splitted_rewards):
        return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v, self.sim.linear_accel, self.sim.angular_accels, splitted_rewards])
