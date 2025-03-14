import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gymnasium as gym
from collections import deque
import time

class DQNVisualizer:
    """
    Visualizer for DQN agents that can render gameplay and performance metrics
    to video files or display them in real-time.
    """
    def __init__(self, 
                 env_name, 
                 agent, 
                 output_dir='./runs',
                 video_size=(1280, 720), 
                 fps=30,
                 show_metrics=True,
                 show_q_values=True,
                 show_preprocessed=True):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        env_name : str
            Name of the Gym environment
        agent : object
            DQN agent with select_action(state) method
        output_dir : str
            Directory to save videos
        video_size : tuple
            Size of the output video (width, height)
        fps : int
            Frames per second for the output video
        show_metrics : bool
            Whether to display metrics on the video
        show_q_values : bool
            Whether to display Q-values on the video
        show_preprocessed : bool
            Whether to display preprocessed frames
        """
        self.env_name = env_name
        self.agent = agent
        self.output_dir = output_dir
        self.video_size = video_size
        self.fps = fps
        self.show_metrics = show_metrics
        self.show_q_values = show_q_values
        self.show_preprocessed = show_preprocessed
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance metrics
        self.rewards_history = []
        self.episode_lengths = []
        self.q_values_history = []
        
        # For calculating FPS
        self.frame_times = deque(maxlen=100)

    def record_episode(self, filename=None, max_steps=10000, render=True):
        """
        Record a full episode of gameplay.
        
        Parameters:
        -----------
        filename : str
            Output filename (without extension)
        max_steps : int
            Maximum number of steps per episode
        render : bool
            Whether to render frames during recording
            
        Returns:
        --------
        dict
            Episode statistics
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.env_name}_{timestamp}"
        
        # Create environment
        env = gym.make(self.env_name, render_mode='rgb_array')
        observation, _ = env.reset()
        
        # Reset agent
        if hasattr(self.agent, 'reset'):
            self.agent.reset()
        
        # Process initial frame(s)
        state = self.agent.process_observation(observation)
        
        # Wait until we have a valid state
        while state is None:
            action = env.action_space.sample()
            observation, _, _, _, _ = env.step(action)
            state = self.agent.process_observation(observation)
        
        # Initialize video writer
        video_path = os.path.join(self.output_dir, f"videos/{filename}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.video_size)
        
        # Episode tracking
        episode_reward = 0
        step_count = 0
        done = False
        q_values_episode = []
        
        # For FPS calculation
        start_time = time.time()
        
        while not done and step_count < max_steps:
            # Get action from agent
            action = self.agent.select_action(state)
            
            # Get Q-values if available
            q_values = None
            if self.show_q_values and hasattr(self.agent, 'get_q_values'):
                q_values = self.agent.get_q_values(state)
                q_values_episode.append(q_values)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = self.agent.process_observation(observation)
            
            # Update episode stats
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Prepare visualization frame
            viz_frame = self.create_visualization_frame(
                observation, 
                state, 
                action, 
                reward, 
                episode_reward,
                step_count,
                q_values,
                info
            )
            
            # Write frame to video
            video_writer.write(cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
            
            if render:
                if not self.running_in_colab():
                    # Display frame if requested
                    cv2.imshow('DQN Visualization', cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            
            # Update timekeeping
            self.frame_times.append(time.time() - start_time)
            start_time = time.time()
            
            # Move to next state
            state = next_state
        
        if self.running_in_colab():
            from IPython.display import Video, display
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            os.chdir('/content/drive/My Drive/projects/ece176_final_project')
            display(Video(video_path, embed=True, width=600, height=400))
            
        # Clean up
        video_writer.release()
        if render:
            cv2.destroyAllWindows()
        env.close()
        
        # Update metrics
        self.rewards_history.append(episode_reward)
        self.episode_lengths.append(step_count)
        if q_values_episode:
            self.q_values_history.append(np.mean(q_values_episode, axis=0))
        
        print(f"Episode recorded to {video_path}")
        print(f"Episode reward: {episode_reward}")
        print(f"Episode length: {step_count} steps")
        
        return {
            'reward': episode_reward,
            'length': step_count,
            'video_path': video_path
        }
    
    def create_visualization_frame(self, 
                                   raw_frame, 
                                   processed_state, 
                                   action, 
                                   reward, 
                                   total_reward,
                                   step_count,
                                   q_values=None,
                                   info=None):
        """
        Create a visualization frame with the game screen and metrics.
        
        Parameters:
        -----------
        raw_frame : numpy.ndarray
            Raw RGB frame from the environment
        processed_state : numpy.ndarray
            Processed state (stacked frames)
        action : int
            Action taken
        reward : float
            Reward received
        total_reward : float
            Cumulative reward for the episode
        step_count : int
            Current step count
        q_values : numpy.ndarray
            Q-values for each action
        info : dict
            Additional information from the environment
            
        Returns:
        --------
        numpy.ndarray
            Visualization frame in RGB format
        """
        # Create a blank canvas
        canvas = np.ones((self.video_size[1], self.video_size[0], 3), dtype=np.uint8) * 255
        
        # Calculate layout
        game_display_height = int(self.video_size[1] * 0.7)
        game_display_width = int(game_display_height * raw_frame.shape[1] / raw_frame.shape[0])
        
        # Resize game frame
        game_frame = cv2.resize(raw_frame, (game_display_width, game_display_height))
        
        # Place game frame in the center
        x_offset = (self.video_size[0] - game_display_width) // 2
        canvas[0:game_display_height, x_offset:x_offset+game_display_width] = game_frame
        
        # Add preprocessed frames if requested
        if self.show_preprocessed and processed_state is not None:
            # Calculate layout for preprocessed frames
            small_frame_size = (84, 84)  # Original preprocessed size
            
            # Upscale slightly for better visibility
            display_size = (84*2, 84*2)
            
            # Stack location - below the main game display
            stack_y = game_display_height + 20
            stack_x = (self.video_size[0] - (display_size[0] * 4 + 30)) // 2  # Center the stack
            
            # Display each frame in the stack
            for i in range(min(4, processed_state.shape[2])):
                # Get the frame and normalize for display
                frame = processed_state[:, :, i]
                
                
                # 1) Convert from PyTorch to NumPy
                frame_np = frame.detach().cpu().numpy()  # shape might be (1, 84, 84)

                # 2) Squeeze any leading dimension
                frame_np = np.squeeze(frame_np)          # => shape (84, 84)

                # 3) Scale up to 0–255 and cast
                frame_display = (frame_np * 255).astype(np.uint8)

                # 4) Now shape is (84, 84), so cv2.COLOR_GRAY2RGB is valid
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2RGB)
                
                # Resize for display
                frame_resized = cv2.resize(frame_rgb, display_size)
                
                # Calculate position
                x = stack_x + i * (display_size[0] + 10)
                
                # Place on canvas
                canvas[stack_y:stack_y+display_size[1], x:x+display_size[0]] = frame_resized
                
                # Add frame number
                cv2.putText(canvas, f"Frame {i+1}", (x, stack_y + display_size[1] + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add metrics if requested
        if self.show_metrics:
            metrics_y = 30
            
            # Game info
            cv2.putText(canvas, f"Game: {self.env_name}", (20, metrics_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Reward info
            metrics_y += 30
            cv2.putText(canvas, f"Step: {step_count} | Reward: {reward:.1f} | Total: {total_reward:.1f}", 
                       (20, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Action info
            metrics_y += 30
            cv2.putText(canvas, f"Action: {action}", (20, metrics_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # FPS calculation
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                metrics_y += 30
                cv2.putText(canvas, f"FPS: {fps:.1f}", (20, metrics_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Environment info
            if info:
                metrics_y += 30
                info_str = " | ".join([f"{k}: {v}" for k, v in info.items() 
                                      if k in ['lives', 'score', 'health']])
                cv2.putText(canvas, info_str, (20, metrics_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add Q-values bar chart if available
        if self.show_q_values and q_values is not None:
            # Create a simple bar chart using OpenCV instead of Matplotlib
            chart_width, chart_height = 400, 200
            chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
            
            # Draw bars
            num_actions = len(q_values)
            bar_width = int(chart_width / (num_actions * 2))
            max_q = max(abs(max(q_values)), abs(min(q_values)), 1.0)  # Avoid division by zero
            
            for i, q in enumerate(q_values):
                # Normalize q value
                bar_height = int((abs(q) / max_q) * (chart_height - 40))
                color = (0, 0, 255) if i == action else (0, 0, 0)  # Red for selected action
                
                # Bar position
                x = i * bar_width * 2 + bar_width
                y = chart_height - 20 - bar_height
                
                # Draw bar
                cv2.rectangle(chart, (x, y), (x + bar_width, chart_height - 20), color, -1)
                
                # Add value text
                cv2.putText(chart, f"{q:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Add action label
                cv2.putText(chart, str(i), (x, chart_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Add title
            cv2.putText(chart, "Q-values by Action", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Calculate position (right side of screen)
            plot_x = self.video_size[0] - chart_width - 20
            plot_y = 20
            
            # Place on canvas
            canvas[plot_y:plot_y+chart_height, plot_x:plot_x+chart_width] = chart
        
        return canvas
    
    def plot_performance_metrics(self, filename=None, show=True):
        """
        Plot performance metrics from recorded episodes.
        
        Parameters:
        -----------
        filename : str
            Output filename (without extension)
        show : bool
            Whether to display the plot
            
        Returns:
        --------
        str
            Path to the saved plot
        """
        if not self.rewards_history:
            print("No episodes recorded yet.")
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.env_name}_metrics_{timestamp}"
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        axes[0].plot(self.rewards_history)
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].grid(True)
        
        # Plot episode lengths
        axes[1].plot(self.episode_lengths)
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].grid(True)
        
        # Add Q-values plot if available
        if self.q_values_history:
            fig.set_size_inches(10, 12)
            fig.add_subplot(3, 1, 3)
            ax = plt.gca()
            
            # Convert to numpy array
            q_values_array = np.array(self.q_values_history)
            
            # Plot mean Q-value for each action
            for i in range(q_values_array.shape[1]):
                ax.plot(q_values_array[:, i], label=f'Action {i}')
            
            ax.set_title('Mean Q-values per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Q-value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.output_dir, f"metrics/{filename}.png")
        plt.savefig(plot_path)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"Performance metrics saved to {plot_path}")
        return plot_path

    def running_in_colab(self):
        import sys
        return 'google.colab' in sys.modules

######################################################################
############################# Example Usage
######################################################################

# Example usage
def example_usage():
    """
    Demonstrate how to use the DQN visualizer.
    """
    from preprocessor import DQNAgent
    import ale_py
    
    # Create environment
    env_name = 'ALE/Breakout-v5'
    
    # Create agent
    env = gym.make(env_name)
    agent = DQNAgent(action_size=env.action_space.n)
    env.close()
    
    # Create visualizer
    visualizer = DQNVisualizer(
        env_name=env_name,
        agent=agent,
        output_dir='../runs',
        show_metrics=True,
        show_q_values=False,  # Set to True if your agent implements get_q_values()
        show_preprocessed=True
    )
    
    # Record episodes
    for episode in range(3):
        print(f"\nRecording episode {episode+1}...")
        episode_stats = visualizer.record_episode(
            filename=f"breakout_episode_{episode+1}",
            max_steps=500,
            render=True  # Set to False for faster recording
        )
    
    # Plot performance metrics
    visualizer.plot_performance_metrics(
        filename="breakout_performance",
        show=True
    )


if __name__ == "__main__":
    example_usage()