from typing import Optional, Sequence, Tuple, Union
import time
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift
import matplotlib as mtb
import matplotlib.pyplot as plt
import msvcrt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import plotly.express as px
import plotly.io as pio
import pandas as pd

# Try to import Vector3, but handle if not available
try:
    from core.algebra import Vector3
except ImportError:
    try:
        from algebra import Vector3
    except ImportError:
        Vector3 = None

def plot_episode_metrics(time_log,state_log,base_pos_log,ee_pos_log,dist_to_bulb_log,dist_to_fixture_log,base_vel_log,arm_vel_norm_log,filename,start_pos, bulb_pos_xy, wall_stop_pos, bulb_pos_3d, fixture_pos_3d, TaskState, robot_name: str, obstacles=None, world_bounds=None):
    
    times = list(np.asarray(time_log).tolist())
    base_x = list(np.asarray(base_pos_log)[:, 0].tolist()) if len(base_pos_log) else []
    base_y = list(np.asarray(base_pos_log)[:, 1].tolist()) if len(base_pos_log) else []
    base_z = list(np.asarray(base_pos_log)[:, 2].tolist()) if len(base_pos_log) else []
    dist_bulb = list(np.asarray(dist_to_bulb_log).tolist())
    dist_fixture = list(np.asarray(dist_to_fixture_log).tolist())
    base_vel = list(np.asarray(base_vel_log).tolist())
    arm_vel = list(np.asarray(arm_vel_norm_log).tolist())
    labels = [s.value for s in TaskState]
    state_numeric = []
    
    for s in state_log:
        try: state_numeric.append(labels.index(s))
        except ValueError:
            try: state_numeric.append(labels.index(str(s)))
            except Exception: state_numeric.append(None)

    state_colors = {
        TaskState.NAVIGATE_TO_BULB.value: 'rgba(173, 216, 230, 0.3)', 
        TaskState.APPROACH_BULB.value: 'rgba(0, 255, 255, 0.3)',      
        TaskState.GRASP_BULB.value: 'rgba(144, 238, 144, 0.3)',      
        TaskState.TRANSPORT_TO_WALL.value: 'rgba(255, 165, 0, 0.3)', 
        TaskState.APPROACH_WALL.value: 'rgba(255, 99, 71, 0.3)',     
        TaskState.RETURN_TO_START.value: 'rgba(147, 112, 219, 0.3)', 
        TaskState.IDLE.value: 'rgba(192, 192, 192, 0.3)'             
        }

    
    time_segments = []
    if len(time_log) > 0:
        current_state = state_log[0]
        start_time = times[0]
        
        for i in range(1, len(times)):
            if state_log[i] != current_state or i == len(times) - 1:
                end_time = times[i] if i < len(times) - 1 else times[-1]
                time_segments.append({
                    'state': current_state,
                    'start': start_time,
                    'end': end_time
                })
                current_state = state_log[i]
                start_time = times[i]

    
    shapes = []
    annotations = []
    max_time = times[-1] if times else 0
    max_metric = max(dist_bulb + dist_fixture + base_x + base_y + base_vel + arm_vel) if len(times) else 1

    for seg in time_segments:
        state_val = seg['state']
        color = state_colors.get(state_val, 'rgba(128, 128, 128, 0.3)')
        
        
        shapes.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper', 
            'x0': seg['start'],
            'y0': 0,
            'x1': seg['end'],
            'y1': 1,
            'fillcolor': color,
            'opacity': 0.5,
            'line_width': 0,
            'layer': 'below' 
        })
        
        
        annotations.append({
            'x': (seg['start'] + seg['end']) / 2,
            'y': 1.05, 
            'xref': 'x',
            'yref': 'paper',
            'text': state_val,
            'showarrow': False,
            'font': {'size': 10, 'color': 'black'},
            'bgcolor': 'white',
            'opacity': 0.7,
            'textangle': -45 if (seg['end'] - seg['start']) < 5 else 0
        })
        
    fig_metrics = go.Figure()
    scatter_args = {'mode': 'lines'}
        
    fig_metrics.add_trace(go.Scatter(x=times, y=dist_bulb, name="EE→Bulb [m]", **scatter_args))
    fig_metrics.add_trace(go.Scatter(x=times, y=dist_fixture, name="EE→Fixture [m]", **scatter_args))
    fig_metrics.add_trace(go.Scatter(x=times, y=base_x, name="Base x [m]", **scatter_args))
    fig_metrics.add_trace(go.Scatter(x=times, y=base_y, name="Base y [m]", **scatter_args))
    fig_metrics.add_trace(go.Scatter(x=times, y=base_vel, name="Base velocity [m/s]", **scatter_args))
    fig_metrics.add_trace(go.Scatter(x=times, y=arm_vel, name="Arm velocity norm [rad/s]", **scatter_args))


    fig_metrics.update_yaxes(title_text="Metric Value")

    fig_metrics.update_layout(
        title=f"{robot_name}: Robot Metrics Over Time", 
        height=600, 
        width=1200, 
        xaxis_title="Time [s]",
        legend_tracegroupgap=20,
        hovermode='x unified',
        shapes=shapes,
        annotations=annotations 
    )
    
    # Third- plot 2D for base trajectory OR obstacles
    # If obstacles are provided, plot obstacle map instead of base trajectory
    if obstacles is not None and world_bounds is not None:
        from plotting.obstacle import plot_obstacles_2d
        base_traj = np.column_stack([base_x, base_y]) if len(base_x) > 0 and len(base_y) > 0 else None
        fig_2 = plot_obstacles_2d(
            obstacles=obstacles,
            world_bounds=world_bounds,
            base_trajectory=base_traj,
            start_pos=start_pos,
            bulb_pos_xy=bulb_pos_xy,
            wall_stop_pos=wall_stop_pos,
            robot_name=robot_name
        )
    else:
        # Default: plot base trajectory
        state_to_segment = {TaskState.NAVIGATE_TO_BULB.value: 1,TaskState.TRANSPORT_TO_WALL.value: 2,TaskState.RETURN_TO_START.value: 3 }

        segment = [state_to_segment.get(s, 0) for s in state_log]

        df_traj = pd.DataFrame({'x': base_x, 'y': base_y,'segment': segment})

        fig_2 = go.Figure()

        segments_info = [
            (1, 'Start -> Bulb'),
            (2, 'Bulb -> Wall Stop'),
            (3, 'Wall Stop -> Start')
        ]

        for seg_id, name in segments_info:
            seg_df = df_traj[df_traj['segment'] == seg_id]
            if not seg_df.empty:
                fig_2.add_trace(go.Scatter(x=seg_df['x'], y=seg_df['y'], mode='lines', name=name, line=dict(width=3)))

                
        # Helper function to convert position to tuple (handles Vector3 and tuples/lists)
        def pos_to_tuple(pos) -> Tuple[float, float]:
            if pos is None:
                return None
            if Vector3 is not None and isinstance(pos, Vector3):
                return (pos.x, pos.y)
            elif isinstance(pos, (tuple, list, np.ndarray)):
                return (float(pos[0]), float(pos[1]))
            else:
                # Try to access as attributes
                try:
                    return (float(pos.x), float(pos.y))
                except AttributeError:
                    # Fallback to indexing
                    return (float(pos[0]), float(pos[1]))
        
        key_points = [
            (start_pos, 'Start Position'),
            (bulb_pos_xy, 'Bulb Position'),
            (wall_stop_pos, 'Wall Stop Position')
        ]

        for pos, label in key_points:
            if pos is not None:
                pos_tuple = pos_to_tuple(pos)
                if pos_tuple is not None:
                    x, y = pos_tuple
                    fig_2.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name=f"{label} ({x:.2f}, {y:.2f})", marker=dict(size=12, symbol='circle')))

        fig_2.update_layout(title=f"{robot_name} Base Trajectory", xaxis_title="X [m]", yaxis_title="Y [m]", height=700, width=1200, showlegend=True, yaxis_scaleanchor="x", yaxis_scaleratio=1)
    
    # third-plot 2-3D plots for EE trajectory
    ee_x = np.asarray(ee_pos_log)[:, 0] if len(ee_pos_log) else []
    ee_y = np.asarray(ee_pos_log)[:, 1] if len(ee_pos_log) else []
    ee_z = np.asarray(ee_pos_log)[:, 2] if len(ee_pos_log) else []

    df_ee_traj = pd.DataFrame({'x': ee_x, 'y': ee_y,'z': ee_z,'state': state_log})
    ee_phases = [{'state': TaskState.APPROACH_BULB.value, 'title': 'EE Approach Bulb (Grasp)', 'target': bulb_pos_3d},{'state': TaskState.APPROACH_WALL.value, 'title': 'EE Approach Wall (Pre-Screw)', 'target': fixture_pos_3d},]

    fig_3 = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],subplot_titles=[p['title'] for p in ee_phases])

    for i, phase in enumerate(ee_phases):
        col = i + 1
        df_phase = df_ee_traj[df_ee_traj['state'] == phase['state']]
        if df_phase.empty:
            scene_name = 'scene' if i == 0 else f'scene{i+1}'
            fig_3.layout[scene_name].update(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data')
            continue

        start_pos_3d = df_phase.iloc[0][['x', 'y', 'z']].to_numpy()
        # Convert target to numpy array, handling Vector3 objects
        target = phase['target']
        if Vector3 is not None and isinstance(target, Vector3):
            target_pos_3d = np.array([target.x, target.y, target.z])
        else:
            target_pos_3d = np.array(target)

        # trajectory line
        fig_3.add_trace(go.Scatter3d(x=df_phase['x'], y=df_phase['y'], z=df_phase['z'],
                                    mode='lines', line=dict(width=4), name='EE Trajectory',
                                    showlegend=(i == 0)), row=1, col=col)

        # start marker
        fig_3.add_trace(go.Scatter3d(x=[start_pos_3d[0]], y=[start_pos_3d[1]], z=[start_pos_3d[2]],
                                    mode='markers', marker=dict(size=6), name='Start',
                                    showlegend=(i == 0), text=[f"Start: ({start_pos_3d[0]:.2f}, {start_pos_3d[1]:.2f}, {start_pos_3d[2]:.2f})"],
                                    hoverinfo='text+name'), row=1, col=col)

        # target marker
        fig_3.add_trace(go.Scatter3d(x=[target_pos_3d[0]], y=[target_pos_3d[1]], z=[target_pos_3d[2]],
                                    mode='markers', marker=dict(size=6, symbol='x'), name='Target',
                                    showlegend=(i == 0), text=[f"Target: ({target_pos_3d[0]:.2f}, {target_pos_3d[1]:.2f}, {target_pos_3d[2]:.2f})"],
                                    hoverinfo='text+name'), row=1, col=col)

        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig_3.layout[scene_name].update(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data')

    fig_3.update_layout(height=500, width=1200, title_text=f"{robot_name} EE Trajectory Phases")



    # prepare html
    html_parts = []
    html_parts.append(pio.to_html(fig_metrics, full_html=False, include_plotlyjs='cdn')) 
    html_parts.append(pio.to_html(fig_2, full_html=False, include_plotlyjs=False)) 
    html_parts.append(pio.to_html(fig_3, full_html=False, include_plotlyjs=False)) 
    
    html_content = "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>Frankie Visualizations</title>\n</head>\n<body>\n"
    html_content += "\n<hr/>\n".join(html_parts)
    html_content += "\n</body>\n</html>"
    
    if filename is not None:
        folder = "visualizations"
        import os
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[Plots Utils] Interactive plot(s) saved as {filepath}")
        return filepath

    return html_content
