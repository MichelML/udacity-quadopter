import pandas as pd
import matplotlib.pyplot as plt

def plot_position_velocity(results):
    plt.subplot(211)
    plt.title("Position & Velocity")
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    plt.subplot(212)
    plt.plot(results['time'], results['x_velocity'], label='x_v')
    plt.plot(results['time'], results['y_velocity'], label='y_v')
    plt.plot(results['time'], results['z_velocity'], label='z_v')
    plt.legend()
    plt.show()
    
def plot_euler_angular_v(results):
    plt.subplot(211)
    plt.title("Euler Angles & Angular Velocity")
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    plt.subplot(212)
    plt.plot(results['time'], results['phi_velocity'], label='phi_v')
    plt.plot(results['time'], results['theta_velocity'], label='theta_v')
    plt.plot(results['time'], results['psi_velocity'], label='psi_v')
    plt.legend()
    plt.show()

def plot_reward_over_time(results):
    plt.title("Net Reward & Deconstructed rewards")
    plt.plot(results['time'], results['net_reward'], label='Net Reward')
    plt.plot(results['time'], results['total_reward'], label='Raw Reward')
    plt.plot(results['time'], results['total_discount'], label='Discount')
    plt.plot(results['time'], results['velocity_discount'], label='XY displacement')
    plt.plot(results['time'], results['angular_velocity_discount'], label='XY displacement')
    plt.plot(results['time'], results['euler_angles_discount'], label='Euler angles')
    plt.legend()
    plt.show()
    
def plot_reward_over_episodes(results):
    plt.title("Total score per episode")
    plt.plot(results['episode'], results['rewards'])
    plt.legend()
    plt.show()

def plot_all(results):
    plot_reward_over_time(results)
    plot_position_velocity(results)
    plot_euler_angular_v(results)
