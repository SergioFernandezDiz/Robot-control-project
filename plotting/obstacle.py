"""
Plotting utilities for obstacle visualization in obstacle avoidance simulation.
"""

from typing import Optional, Sequence, Tuple, Union, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Try to import Vector3, but handle if not available
try:
    from core.algebra import Vector3
except ImportError:
    try:
        from algebra import Vector3
    except ImportError:
        Vector3 = None


def plot_obstacles_2d(
    obstacles: List,
    world_bounds: Tuple[float, float, float, float],
    base_trajectory: Optional[np.ndarray] = None,
    start_pos: Optional[Tuple[float, float]] = None,
    bulb_pos_xy: Optional[Tuple[float, float]] = None,
    wall_stop_pos: Optional[Tuple[float, float]] = None,
    robot_name: str = "Frankie"
) -> go.Figure:
    """
    Create a 2D plot showing obstacles and robot trajectory.
    
    Args:
        obstacles: List of Obstacle objects with position, radius, height attributes
        world_bounds: Tuple of (min_x, max_x, min_y, max_y)
        base_trajectory: Optional array of base positions (N, 2) or (N, 3)
        start_pos: Optional start position (x, y)
        bulb_pos_xy: Optional bulb position (x, y)
        wall_stop_pos: Optional wall stop position (x, y)
        robot_name: Name of the robot for title
    
    Returns:
        Plotly figure with obstacles and trajectory
    """
    min_x, max_x, min_y, max_y = world_bounds
    
    fig = go.Figure()
    
    # Plot obstacles as circles
    for i, obs in enumerate(obstacles):
        # Extract position and radius
        if hasattr(obs, 'position'):
            pos = obs.position
            if isinstance(pos, (tuple, list, np.ndarray)):
                obs_x, obs_y = float(pos[0]), float(pos[1])
            else:
                obs_x, obs_y = float(pos.x), float(pos.y)
        else:
            continue
            
        radius = float(obs.radius) if hasattr(obs, 'radius') else 0.2
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = obs_x + radius * np.cos(theta)
        circle_y = obs_y + radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            fill='toself',
            fillcolor='rgba(153, 51, 102, 0.5)',  # Purple/pink color for obstacles
            line=dict(color='rgba(153, 51, 102, 0.8)', width=2),
            name=f'Obstacle {i+1}' if i < 5 else None,  # Only show first 5 in legend
            showlegend=(i < 5),
            hoverinfo='skip'
        ))
        
        # Add center marker
        fig.add_trace(go.Scatter(
            x=[obs_x],
            y=[obs_y],
            mode='markers',
            marker=dict(size=6, color='rgba(153, 51, 102, 1.0)', symbol='circle'),
            name=None,
            showlegend=False,
            hovertext=f'Obstacle {i+1}: r={radius:.2f}m',
            hoverinfo='text'
        ))
    
    # Plot base trajectory if provided
    if base_trajectory is not None:
        traj = np.asarray(base_trajectory)
        if traj.shape[1] >= 2:
            traj_x = traj[:, 0]
            traj_y = traj[:, 1]
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode='lines',
                name='Robot Trajectory',
                line=dict(color='blue', width=2),
                hoverinfo='skip'
            ))
    
    # Helper function to convert position to tuple
    def pos_to_tuple(pos) -> Optional[Tuple[float, float]]:
        if pos is None:
            return None
        if Vector3 is not None and isinstance(pos, Vector3):
            return (pos.x, pos.y)
        elif isinstance(pos, (tuple, list, np.ndarray)):
            return (float(pos[0]), float(pos[1]))
        else:
            try:
                return (float(pos.x), float(pos.y))
            except AttributeError:
                return (float(pos[0]), float(pos[1]))
    
    # Add key points
    key_points = [
        (start_pos, 'Start Position', 'green', 'circle'),
        (bulb_pos_xy, 'Bulb Position', 'yellow', 'circle'),
        (wall_stop_pos, 'Wall Stop Position', 'red', 'square')
    ]
    
    for pos, label, color, symbol in key_points:
        if pos is not None:
            pos_tuple = pos_to_tuple(pos)
            if pos_tuple is not None:
                x, y = pos_tuple
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    name=f"{label} ({x:.2f}, {y:.2f})",
                    marker=dict(size=12, color=color, symbol=symbol),
                    hoverinfo='name'
                ))
    
    # Update layout
    fig.update_layout(
        title=f"{robot_name}: Obstacle Map with Robot Trajectory",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        height=700,
        width=1200,
        showlegend=True,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        xaxis_range=[min_x - 0.5, max_x + 0.5],
        yaxis_range=[min_y - 0.5, max_y + 0.5]
    )
    
    return fig

