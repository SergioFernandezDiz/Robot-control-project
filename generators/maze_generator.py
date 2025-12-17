import random
import numpy as np

EMPTY = 0
WALL = 1
NORTH, SOUTH, EAST, WEST = "n", "s", "e", "w"


def _visit(maze, visited, x, y, width, height):
    maze[(x, y)] = EMPTY

    while True:
        unvisited_neighbors = []

        if y > 1 and (x, y - 2) not in visited:
            unvisited_neighbors.append(NORTH)
        if y < height - 2 and (x, y + 2) not in visited:
            unvisited_neighbors.append(SOUTH)
        if x > 1 and (x - 2, y) not in visited:
            unvisited_neighbors.append(WEST)
        if x < width - 2 and (x + 2, y) not in visited:
            unvisited_neighbors.append(EAST)

        if not unvisited_neighbors:
            return

        direction = random.choice(unvisited_neighbors)

        if direction == NORTH:
            next_x = x
            next_y = y - 2
            maze[(x, y - 1)] = EMPTY
        elif direction == SOUTH:
            next_x = x
            next_y = y + 2
            maze[(x, y + 1)] = EMPTY
        elif direction == WEST:
            next_x = x - 2
            next_y = y
            maze[(x - 1, y)] = EMPTY
        else:  # EAST
            next_x = x + 2
            next_y = y
            maze[(x + 1, y)] = EMPTY

        visited.append((next_x, next_y))
        _visit(maze, visited, next_x, next_y, width, height)


def _generate_base_maze(base_w: int, base_h: int) -> np.ndarray:
    if base_w < 3 or base_h < 3:
        raise ValueError("Base maze too small")

    start_x = random.randrange(1, base_w, 2)
    start_y = random.randrange(1, base_h, 2)

    visited = [(start_x, start_y)]
    maze_dict = {}

    for x in range(base_w):
        for y in range(base_h):
            maze_dict[(x, y)] = WALL

    _visit(maze_dict, visited, start_x, start_y, base_w, base_h)

    arr = np.zeros((base_w, base_h), dtype=int)
    for (x, y), v in maze_dict.items():
        if v == WALL:
            arr[x, y] = 1
        else:
            arr[x, y] = 0

    return arr


def _scale_maze_thin_walls(base: np.ndarray, scale: int) -> np.ndarray:
    h0, w0 = base.shape

    h = h0 * scale
    w = w0 * scale
    world = np.zeros((h, w), dtype=int)

    wall_positions = []

    for x in range(h0):
        for y in range(w0):
            if base[x, y] == 1:
                cx = x * scale + scale // 2
                cy = y * scale + scale // 2
                wall_positions.append((cx, cy))

    for (cx, cy) in wall_positions:
        world[cx, cy] = 1

    for (cx, cy) in wall_positions:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = cx + dx * scale
            ny = cy + dy * scale
            if 0 <= nx < h and 0 <= ny < w and world[nx, ny] == 1:
                if dx != 0:
                    step = 1 if nx > cx else -1
                    for k in range(cx, nx + step, step):
                        world[k, cy] = 1
                if dy != 0:
                    step = 1 if ny > cy else -1
                    for k in range(cy, ny + step, step):
                        world[cx, k] = 1

    return world


def _add_outer_walls(world: np.ndarray) -> np.ndarray:
    h, w = world.shape
    world[0, :] = 1
    world[h - 1, :] = 1
    world[:, 0] = 1
    world[:, w - 1] = 1
    return world


def _add_barrier_around_walls(world: np.ndarray) -> np.ndarray:
    h, w = world.shape
    dirs = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    result = world.copy()
    for x in range(h):
        for y in range(w):
            if world[x, y] == 1:
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if result[nx, ny] == 0:
                            result[nx, ny] = 2

    return result


def get_maze(
    total_width: int = 41,
    total_height: int = 47,
    scale: int = 3,
    lower_room_size: int = 6,
) -> np.ndarray:
    if total_width < 9 or total_height < 9:
        raise ValueError("total_width and total_height must be reasonably large")

    margin = 1
    inner_w = total_width - 2 * margin
    inner_h = total_height - margin - lower_room_size

    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("total size too small for maze with given scale/margins/room")

    base_w = max(3, inner_h // scale)
    base_h = max(3, inner_w // scale)

    if base_w % 2 == 0:
        base_w -= 1
    if base_h % 2 == 0:
        base_h -= 1

    if base_w < 3 or base_h < 3:
        raise ValueError("Base maze too small after adjustment")

    base_maze = _generate_base_maze(base_w, base_h)
    base_maze[:,0] = 0
    base_maze[0,:] = 0
    base_maze[len(base_maze)-1,:] = 0
    base_maze[:, len(base_maze) - 1] = 0
 
    scaled = _scale_maze_thin_walls(base_maze, scale)
    sh, sw = scaled.shape

    if sh > inner_h or sw > inner_w:
        raise ValueError("Scaled maze unexpectedly larger than available inner area")

    world = np.zeros((total_height, total_width), dtype=int)

    offset_x = 1 + margin
    free_w = inner_w
    extra_w = free_w - sw
    offset_y = 1 + margin + max(0, extra_w // 2)

    world[offset_x : offset_x + sh, offset_y : offset_y + sw] = scaled

    world = _add_outer_walls(world)

    world = _add_barrier_around_walls(world)

    return world


def print_maze(grid: np.ndarray):
    h, w = grid.shape
    wall_char = "██"
    near_char = "░░"
    empty_char = "  "

    lines = []
    for x in range(h):
        row_chars = []
        for y in range(w):
            val = grid[x, y]
            if val == 1:
                row_chars.append(wall_char)
            elif val == 2:
                row_chars.append(near_char)
            else:
                row_chars.append(empty_char)
        lines.append("".join(row_chars))
    print("\n".join(lines))
