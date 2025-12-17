"""
Frankie Simulation Launcher
A simple GUI to launch either the basic pick-and-place or obstacle avoidance simulation.

Developed by:
- Federico Panico
- Sergio Fernandez Diz
- Alessandro Carnio

Institution: SUPSI - University of Applied Sciences and Arts of Southern Switzerland
Course: M-D5320ZE - Robot Control Applications
Professor: Prof. Antonio Paolillo
Academic Year: 2025/2026
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Paths to the simulation scripts
PICK_AND_PLACE_SCRIPT = SCRIPT_DIR / "scripts" / "pick_and_place.py"
OBSTACLE_AVOIDANCE_SCRIPT = SCRIPT_DIR / "scripts" / "obstacle_avoidance.py"
MAZE_SCRIPT = SCRIPT_DIR / "scripts" / "maze.py"


def run_script(script_path: Path):
    """Execute a Python script in a new process (cross-platform)."""
    if not script_path.exists():
        messagebox.showerror(
            "File Not Found",
            f"Could not find the script:\n{script_path}\n\nPlease ensure the file exists."
        )
        return
    
    try:
        if sys.platform == "win32":
            # Windows: open in new console window
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(SCRIPT_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        elif sys.platform == "darwin":
            # macOS: open in new Terminal window using AppleScript
            # Escape quotes properly for AppleScript
            script_dir_escaped = str(SCRIPT_DIR).replace('"', '\\"')
            script_path_escaped = str(script_path).replace('"', '\\"')
            python_escaped = sys.executable.replace('"', '\\"')
            script_cmd = f'cd "{script_dir_escaped}" && "{python_escaped}" "{script_path_escaped}"'
            osascript_cmd = f'tell application "Terminal" to do script "{script_cmd}"'
            subprocess.Popen(
                ["osascript", "-e", osascript_cmd]
            )
        else:
            # Linux: try to open in new terminal (gnome-terminal, xterm, etc.)
            # Fallback to running in background if no terminal found
            try:
                # Try gnome-terminal first (most common on Linux)
                subprocess.Popen(
                    ["gnome-terminal", "--", sys.executable, str(script_path)],
                    cwd=str(SCRIPT_DIR)
                )
            except FileNotFoundError:
                try:
                    # Try xterm
                    subprocess.Popen(
                        ["xterm", "-e", f"{sys.executable} {script_path}"],
                        cwd=str(SCRIPT_DIR)
                    )
                except FileNotFoundError:
                    # Fallback: run in background (output goes to launcher console)
                    subprocess.Popen(
                        [sys.executable, str(script_path)],
                        cwd=str(SCRIPT_DIR)
                    )
        
        print(f"Launched: {script_path.name}")
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to launch {script_path.name}:\n{str(e)}"
        )


def launch_pick_and_place():
    """Launch the basic pick-and-place simulation."""
    run_script(PICK_AND_PLACE_SCRIPT)


def launch_obstacle_avoidance():
    """Launch the obstacle avoidance simulation."""
    run_script(OBSTACLE_AVOIDANCE_SCRIPT)


def launch_maze():
    """Launch the maze simulation."""
    run_script(MAZE_SCRIPT)


def main():
    """Create and show the launcher GUI."""
    root = tk.Tk()
    root.title("Frankie Simulation Launcher")
    root.geometry("500x250")
    root.resizable(False, False)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Main frame
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title label
    title_label = tk.Label(
        main_frame,
        text=f"Mobile Manipulation\nFrankie Robot Simulation",
        font=("Arial", 16, "bold"),
        pady=10
    )
    title_label.pack()
    
    # Subtitle
    subtitle_label = tk.Label(
        main_frame,
        text="Select a simulation to launch:",
        font=("Arial", 10),
        pady=5
    )
    subtitle_label.pack()
    
    # Buttons frame (single row)
    buttons_frame = tk.Frame(main_frame, pady=10)
    buttons_frame.pack()
    
    # Button 1: Pick and Place
    btn_pick_place = tk.Button(
        buttons_frame,
        text="Pick and Place\n(Basic)",
        font=("Arial", 10),
        width=16,
        height=3,
        command=launch_pick_and_place,
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        cursor="hand2",
        relief=tk.RAISED,
        bd=3
    )
    btn_pick_place.pack(side=tk.LEFT, padx=8)
    
    # Button 2: Obstacle Avoidance
    btn_obstacle = tk.Button(
        buttons_frame,
        text="Obstacle Avoidance\n(Advanced)",
        font=("Arial", 10),
        width=16,
        height=3,
        command=launch_obstacle_avoidance,
        bg="#2196F3",
        fg="white",
        activebackground="#0b7dda",
        cursor="hand2",
        relief=tk.RAISED,
        bd=3
    )
    btn_obstacle.pack(side=tk.LEFT, padx=8)
    
    # Button 3: Maze Simulation
    btn_maze = tk.Button(
        buttons_frame,
        text="Maze Simulation\n(Two Robots)",
        font=("Arial", 10),
        width=16,
        height=3,
        command=launch_maze,
        bg="#FF9800",
        fg="white",
        activebackground="#e68900",
        cursor="hand2",
        relief=tk.RAISED,
        bd=3
    )
    btn_maze.pack(side=tk.LEFT, padx=8)
    
    # Status label
    status_label = tk.Label(
        main_frame,
        text="Click a button to start the simulation",
        font=("Arial", 9),
        fg="gray",
        pady=10
    )
    status_label.pack()
    
    # Run the GUI
    root.mainloop()


if __name__ == "__main__":
    main()

