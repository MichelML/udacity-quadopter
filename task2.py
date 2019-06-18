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
        self.points_per_component = 10.
        
    def distance_between_points(self, pos, target):
        return distance.cdist([pos], [target])[0][0]

    def get_rewards(self):
        position = max(self.points_per_component - self.distance_between_points(self.sim.pose[:3], self.target_pos[:3]), 0.)
        
        target_velocity = [0.,0.,0.]
        velocity = 0.2*max(self.points_per_component - self.distance_between_points(self.sim.v, target_velocity), 0.)
        
        target_angular_velocity = [0., 0., 0.]
        angular_velocity = 0.2*max(self.points_per_component - self.distance_between_points(self.sim.angular_v, target_angular_velocity), 0.)
        
        target_euler = [0.,0.,0.]
        euler_angles = 0.2*max(self.points_per_component - self.distance_between_points(self.sim.pose[3:], target_euler), 0.)
        
        target_linear_accel = [0.,0.,0.]
        linear_accel = 0.2*max(self.points_per_component - self.distance_between_points(self.sim.linear_accel, target_linear_accel), 0.)
        
        target_angular_accel = [0.,0.,0.]
        angular_accel = 0.2*max(self.points_per_component - self.distance_between_points(self.sim.angular_accels, target_angular_accel), 0.)
        
        time = .5
        
        individual_rewards = [position, euler_angles, velocity, angular_velocity, linear_accel, angular_accel, time]
        total_reward = sum(individual_rewards)
        
        return individual_rewards, total_reward
    
    def get_distances_from_final_target(self):
        v = self.distance_between_points(self.sim.v, [0.]*3)
        angular_v = self.distance_between_points(self.sim.v, [0.]*3)
        euler_angles = self.distance_between_points(self.sim.pose[3:], [0.]*3)
        position = self.distance_between_points(self.sim.pose[:3], self.target_pos[:3])
        linear_accel = self.distance_between_points(self.sim.linear_accel, [0.]*3)
        angular_accels = self.distance_between_points(self.sim.angular_accels, [0.]*3)
        
        return [v, angular_v, euler_angles, position, linear_accel, angular_accels]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        all_rewards = np.zeros(8)
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            individual_rewards, total_reward = self.get_rewards()
            all_rewards = all_rewards + np.concatenate([individual_rewards, [total_reward]])
            reward += total_reward
            self.score += reward
            pose_all.append(self.get_sim_state())
        next_state = np.concatenate(pose_all)
        return next_state, all_rewards, done

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
        return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v, self.sim.linear_accel, self.sim.angular_accels, self.get_distances_from_final_target(), [self.sim.time], self.sim.prop_wind_speed])
