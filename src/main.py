import pygame
import sys
import math
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple
from shapely.geometry import Polygon, Point
from collections import defaultdict
import asyncio
from queue import Queue
from heapq import heappush, heappop

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
VERTEX_COLOR = (100, 149, 237)        # Cornflower blue
TRIANGLE_COLOR = (240, 240, 240)      # Very light grey for base triangles
EDGE_COLOR = (180, 180, 180)          # Grey for edges
GRID_COLOR = (220, 220, 220)          # Light grey
START_COLOR = (50, 205, 50)           # Lime green for start
END_COLOR = (220, 20, 60)             # Crimson for end
BFS_PATH_COLOR = (135, 206, 235, 128) # Semi-transparent sky blue
DIJKSTRA_PATH_COLOR = (221, 160, 221, 128)  # Semi-transparent plum
OVERLAPPING_PATH_COLOR = (152, 251, 152, 128)  # Semi-transparent pale green
ASTAR_PATH_COLOR = (255, 165, 0, 128)  # Semi-transparent orange
VISIBILITY_PATH_COLOR = (255, 20, 147, 128)  # Semi-transparent deep pink

# Screen dimensions and grid
WIDTH, HEIGHT = 1400, 1100
GRID_SIZE = 20

# Define legend width
LEGEND_WIDTH = 300

# Update grid drawing to exclude the legend area
GRID_DRAW_WIDTH = WIDTH - LEGEND_WIDTH

class DelaunayDemo:
    def __init__(self):
        pygame.init()
    
        # Check if running in Streamlit
        self.is_streamlit = 'streamlit' in sys.modules
        
        if not self.is_streamlit:
            # Normal Pygame initialization for PyScript
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
            self.screen = pygame.display.get_surface()
        else:
            # Streamlit initialization
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        
        pygame.display.set_caption("Delaunay Triangulation Demo")
        
        self.polygons = []          # List of polygons (first is boundary)
        self.current_polygon = []    # Current polygon being drawn
        self.building_polygon = False
        self.triangulation = None    # Store Delaunay triangulation
        self.valid_triangles = []    # Store only the valid triangles
        self.triangle_graph = defaultdict(list)  # Store adjacency information
        self.start_point = None
        self.end_point = None
        self.bfs_path = []  # Rename existing path to bfs_path
        self.dijkstra_path = []  # Add new path for Dijkstra's
        self.placing_start = False
        self.placing_end = False
        self.astar_path = []  # Add new path for A*
        self.bfs_explored_count = 0
        self.dijkstra_explored_count = 0
        self.astar_explored_count = 0
        self.visibility_path = []
        self.visibility_explored_count = 0
        self.show_visibility = False
        self.visibility_path_length = 0
        
        # Add bounding box
        self.add_bounding_box()

    def add_bounding_box(self):
        """Add initial bounding box."""
        margin = GRID_SIZE
        box = [
            (margin, margin),
            (GRID_DRAW_WIDTH - margin, margin),
            (GRID_DRAW_WIDTH - margin, HEIGHT - margin),
            (margin, HEIGHT - margin)
        ]
        self.polygons.append(box)

    def snap_to_grid(self, pos):
        """Snap position to grid."""
        x = round(pos[0] / GRID_SIZE) * GRID_SIZE
        y = round(pos[1] / GRID_SIZE) * GRID_SIZE
        return (x, y)

    def is_near_first_point(self, pos, threshold=20):
        """Check if position is near the first point of current polygon."""
        if not self.current_polygon:
            return False
        first_point = self.current_polygon[0]
        dist = math.hypot(pos[0] - first_point[0], pos[1] - first_point[1])
        return dist < threshold

    def compute_triangulation(self):
        """Compute triangulation using scipy's Delaunay."""
        if len(self.polygons) < 1:
            return

        # Collect vertices from boundary
        points = []
        for polygon in self.polygons:
            points.extend(polygon)
        points = np.array(points)

        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Filter triangles
        boundary = Polygon(self.polygons[0])
        obstacles = [Polygon(poly) for poly in self.polygons[1:]]
        
        self.valid_triangles = []
        self.triangle_centroids = []  # Add storage for centroids
        valid_indices = []
        
        for i, simplex in enumerate(tri.simplices):
            triangle_points = points[simplex]
            centroid = np.mean(triangle_points, axis=0)
            
            if boundary.contains(Point(centroid)):
                valid = True
                for obstacle in obstacles:
                    if obstacle.contains(Point(centroid)):
                        valid = False
                        break
                if valid:
                    self.valid_triangles.append(triangle_points)
                    self.triangle_centroids.append(centroid)  # Store centroid
                    valid_indices.append(i)

        # Compute adjacency graph
        self.compute_triangle_graph(tri, valid_indices)
        
        # Recompute path if start and end points exist
        if self.start_point and self.end_point:
            self.find_path()

    def compute_triangle_graph(self, tri, valid_indices):
        """Create adjacency graph of valid triangles."""
        self.triangle_graph = defaultdict(list)
        
        # Create a mapping from original indices to valid triangle indices
        index_map = {orig: new for new, orig in enumerate(valid_indices)}
        
        # Get neighbors from Delaunay triangulation
        neighbors = tri.neighbors
        
        for new_idx, orig_idx in enumerate(valid_indices):
            # Get neighbors of current triangle
            for neighbor_idx in neighbors[orig_idx]:
                # Check if neighbor is valid and not -1 (which indicates no neighbor)
                if neighbor_idx in index_map:
                    self.triangle_graph[new_idx].append(index_map[neighbor_idx])
        
        print("\nTriangle Adjacency Graph:")
        for tri_idx, neighbors in self.triangle_graph.items():
            print(f"Triangle {tri_idx} is adjacent to triangles: {neighbors}")

    def find_containing_triangle(self, point):
        """Find which triangle contains the given point."""
        point = Point(point[0], point[1])
        for i, triangle in enumerate(self.valid_triangles):
            triangle_poly = Polygon(triangle)
            if triangle_poly.contains(point):
                return i
        return None

    def find_path(self):
        """Find paths using all algorithms."""
        # Clear existing paths first
        self.bfs_path = []
        self.dijkstra_path = []
        self.astar_path = []
        self.visibility_path = []
        self.visibility_path_length = 0

        if not self.start_point or not self.end_point:
            print("Need both start and end points!")
            return

        start_tri = self.find_containing_triangle(self.start_point)
        end_tri = self.find_containing_triangle(self.end_point)

        if start_tri is None or end_tri is None:
            print("Start or end point not in any triangle!")
            return

        # Reset exploration counts
        self.bfs_explored_count = 0
        self.dijkstra_explored_count = 0
        self.astar_explored_count = 0
        
        # Find paths using different algorithms
        self.bfs_path = self.find_bfs_path(start_tri, end_tri)
        self.dijkstra_path = self.find_dijkstra_path(start_tri, end_tri)
        self.astar_path = self.find_astar_path(start_tri, end_tri)
        
        # Verify all paths were found successfully
        if not self.bfs_path or not self.dijkstra_path or not self.astar_path:
            print("Could not find complete path!")
            self.bfs_path = []
            self.dijkstra_path = []
            self.astar_path = []
            return

        print(f"BFS path: {self.bfs_path}")
        print(f"Dijkstra path: {self.dijkstra_path}")
        print(f"A* path: {self.astar_path}")

        # Only compute visibility path if flag is set
        if self.show_visibility:
            self.visibility_path = self.find_visibility_path(self.start_point, self.end_point)
            if len(self.visibility_path) > 1:
                self.visibility_path_length = sum(
                    math.hypot(b[0]-a[0], b[1]-a[1]) 
                    for a, b in zip(self.visibility_path[:-1], self.visibility_path[1:])
                )

    def find_bfs_path(self, start_tri, end_tri):
        """Original BFS implementation."""
        self.bfs_explored_count = 0  # Reset counter
        queue = Queue()
        queue.put(start_tri)
        came_from = {start_tri: None}

        while not queue.empty():
            current = queue.get()
            self.bfs_explored_count += 1  # Increment counter
            if current == end_tri:
                break
            for next_tri in self.triangle_graph[current]:
                if next_tri not in came_from:
                    queue.put(next_tri)
                    came_from[next_tri] = current

        # Reconstruct path
        path = []
        current = end_tri
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return list(reversed(path))

    def find_dijkstra_path(self, start_tri, end_tri):
        """Find path using Dijkstra's algorithm with Euclidean distance."""
        self.dijkstra_explored_count = 0  # Reset counter
        distances = {i: float('infinity') for i in self.triangle_graph}
        distances[start_tri] = 0
        came_from = {start_tri: None}
        pq = [(0, start_tri)]

        while pq:
            current_dist, current = heappop(pq)
            self.dijkstra_explored_count += 1  # Increment counter
            
            if current == end_tri:
                break

            if current_dist > distances[current]:
                continue

            for next_tri in self.triangle_graph[current]:
                dist = np.linalg.norm(
                    self.triangle_centroids[current] - self.triangle_centroids[next_tri]
                )
                new_dist = distances[current] + dist

                if new_dist < distances[next_tri]:
                    distances[next_tri] = new_dist
                    came_from[next_tri] = current
                    heappush(pq, (new_dist, next_tri))

        # Reconstruct path
        path = []
        current = end_tri
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return list(reversed(path))

    def find_astar_path(self, start_tri, end_tri):
        """Find path using A* algorithm with Euclidean distance heuristic."""
        print("\n=== A* Debug ===")
        
        def heuristic(tri_idx):
            h = np.linalg.norm(
                self.triangle_centroids[tri_idx] - self.triangle_centroids[end_tri]
            )
            print(f"  Heuristic for triangle {tri_idx}: {h:.2f}")
            return h

        g_score = {i: float('infinity') for i in self.triangle_graph}
        g_score[start_tri] = 0
        
        open_set = set([start_tri])
        closed_set = set()
        came_from = {}
        
        f_score = {start_tri: heuristic(start_tri)}
        explored_order = []  # Track exploration order
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('infinity')))
            self.astar_explored_count += 1  # Increment counter
            explored_order.append(current)
            
            print(f"\nExploring triangle {current}")
            print(f"  g_score: {g_score[current]:.2f}")
            print(f"  f_score: {f_score[current]:.2f}")
            
            if current == end_tri:
                print(f"Found end! Exploration order: {explored_order}")
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_tri)
                return list(reversed(path))
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.triangle_graph[current]:
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + np.linalg.norm(
                    self.triangle_centroids[current] - self.triangle_centroids[neighbor]
                )
                
                print(f"  Checking neighbor {neighbor}")
                print(f"    Current g_score: {g_score[neighbor]:.2f}")
                print(f"    Tentative g_score: {tentative_g:.2f}")
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g >= g_score[neighbor]:
                    print("    Skip: Not a better path")
                    continue
                    
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                print(f"    Updated f_score: {f_score[neighbor]:.2f}")
            
        print("No path found!")
        return []

    def find_visibility_path(self, start_point, end_point):
        """Find true shortest path using visibility graph approach."""
        if not start_point or not end_point:
            return []

        # Collect vertices but skip bounding box vertices
        vertices = [start_point, end_point]
        for polygon in self.polygons[1:]:  # Skip bounding box vertices
            vertices.extend(polygon)
        
        # Build visibility graph
        visible_edges = []
        # Only check each pair once (j > i)
        for i, v1 in enumerate(vertices):
            for j in range(len(vertices)):
                if i == j:  # Skip same vertex
                    continue
                v2 = vertices[j]
                
                # Check if line segment intersects any polygon edges
                is_visible = True
                for polygon in self.polygons[1:]:  # Skip bounding box for intersection tests
                    for k in range(len(polygon)):
                        p1 = polygon[k]
                        p2 = polygon[(k + 1) % len(polygon)]
                        
                        # Skip if checking against own vertices or if line runs along edge
                        if (v1 in (p1, p2) and v2 in (p1, p2)):
                            continue
                            
                        if self.line_segments_intersect(v1, v2, p1, p2):
                            is_visible = False
                            break
                    if not is_visible:
                        break
                
                if is_visible:
                    dist = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
                    visible_edges.append((i, j, dist))

        # Build graph for Dijkstra's algorithm
        graph = defaultdict(list)
        for start_idx, end_idx, dist in visible_edges:
            graph[start_idx].append((end_idx, dist))

        # Run Dijkstra's algorithm
        distances = {i: float('infinity') for i in range(len(vertices))}
        distances[0] = 0  # Start point is at index 0
        pq = [(0, 0)]  # (distance, vertex_index)
        came_from = {0: None}
        
        while pq:
            current_dist, current = heappop(pq)
            self.visibility_explored_count += 1
            
            if current == 1:  # End point is at index 1
                break
                
            if current_dist > distances[current]:
                continue
                
            for next_vertex, edge_dist in graph[current]:
                new_dist = distances[current] + edge_dist
                
                if new_dist < distances[next_vertex]:
                    distances[next_vertex] = new_dist
                    came_from[next_vertex] = current
                    heappush(pq, (new_dist, next_vertex))
        
        # Reconstruct path
        path = []
        current = 1  # End point index
        while current is not None:
            path.append(vertices[current])
            current = came_from.get(current)
        
        return list(reversed(path))

    def handle_click(self, pos):
        """Handle mouse clicks for polygon creation and point placement."""
        if self.placing_start:
            self.start_point = self.snap_to_grid(pos)
            self.placing_start = False
            if self.start_point and self.end_point:
                self.find_path()
        elif self.placing_end:
            self.end_point = self.snap_to_grid(pos)
            self.placing_end = False
            if self.start_point and self.end_point:
                self.find_path()
        elif self.building_polygon:
            snapped_pos = self.snap_to_grid(pos)
            
            if len(self.current_polygon) >= 3 and self.is_near_first_point(snapped_pos):
                self.polygons.append(self.current_polygon[:])
                self.current_polygon = []
                self.building_polygon = False
                self.compute_triangulation()
            else:
                self.current_polygon.append(snapped_pos)

    def draw(self):
        """Draw everything to the screen."""
        self.screen.fill(WHITE)
        
        # Draw grid excluding legend area
        self.draw_grid()

        # Draw triangles with path coloring
        for i, triangle in enumerate(self.valid_triangles):
            color = TRIANGLE_COLOR
            if i in self.bfs_path and i in self.dijkstra_path and i in self.astar_path:
                color = OVERLAPPING_PATH_COLOR
            elif i in self.bfs_path:
                color = BFS_PATH_COLOR
            elif i in self.dijkstra_path:
                color = DIJKSTRA_PATH_COLOR
            elif i in self.astar_path:
                color = ASTAR_PATH_COLOR
            pygame.draw.polygon(self.screen, color, triangle)
            pygame.draw.polygon(self.screen, EDGE_COLOR, triangle, 2)

        # Draw Dijkstra's centroid path with purple dots
        if len(self.dijkstra_path) > 1:
            # Get all points for the path including start and end points
            path_points = []
            
            # Add start point to first centroid
            if self.start_point:
                start_tri = self.find_containing_triangle(self.start_point)
                if start_tri is not None:
                    path_points.append(self.start_point)
                    path_points.append(tuple(map(int, self.triangle_centroids[start_tri])))
            
            # Add all centroids
            centroid_points = [self.triangle_centroids[i] for i in self.dijkstra_path]
            path_points.extend([(int(p[0]), int(p[1])) for p in centroid_points])
            
            # Add last centroid to end point
            if self.end_point:
                end_tri = self.find_containing_triangle(self.end_point)
                if end_tri is not None:
                    path_points.append(self.end_point)

            # Draw dotted lines between all points
            for i in range(len(path_points) - 1):
                start = path_points[i]
                end = path_points[i + 1]
                
                # Calculate number of dots based on distance
                distance = math.hypot(end[0] - start[0], end[1] - start[1])
                num_dots = max(2, int(distance / 8))  # Ensure at least 2 dots
                
                # Draw dots along the line
                for j in range(num_dots):
                    t = j / (num_dots - 1) if num_dots > 1 else 0
                    x = int(start[0] + t * (end[0] - start[0]))
                    y = int(start[1] + t * (end[1] - start[1]))
                    # Use respective colors for each path
                    pygame.draw.circle(self.screen, (128, 0, 128), (x, y), 3)  # For Dijkstra

        # Draw A* centroid path with orange dots
        if len(self.astar_path) > 1:
            # Get all points for the A* path including start and end points
            path_points = []
            
            # Add start point to first centroid
            if self.start_point:
                start_tri = self.find_containing_triangle(self.start_point)
                if start_tri is not None:
                    path_points.append(self.start_point)
                    path_points.append(tuple(map(int, self.triangle_centroids[start_tri])))
            
            # Add all centroids
            centroid_points = [self.triangle_centroids[i] for i in self.astar_path]
            path_points.extend([(int(p[0]), int(p[1])) for p in centroid_points])
            
            # Add last centroid to end point
            if self.end_point:
                end_tri = self.find_containing_triangle(self.end_point)
                if end_tri is not None:
                    path_points.append(self.end_point)

            # Draw dotted lines between all points
            for i in range(len(path_points) - 1):
                start = path_points[i]
                end = path_points[i + 1]
                
                # Calculate number of dots based on distance
                distance = math.hypot(end[0] - start[0], end[1] - start[1])
                num_dots = max(2, int(distance / 8))
                
                # Calculate offset direction (perpendicular to line)
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                # Normalize and rotate 90 degrees
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    offset_x = -dy/length * 5  # 5 pixel offset
                    offset_y = dx/length * 5
                else:
                    offset_x = offset_y = 0
                
                # Draw dots along the line with offset
                for j in range(num_dots):
                    t = j / (num_dots - 1) if num_dots > 1 else 0
                    x = int(start[0] + t * (end[0] - start[0]) + offset_x)
                    y = int(start[1] + t * (end[1] - start[1]) + offset_y)
                    pygame.draw.circle(self.screen, (255, 140, 0), (x, y), 3)

        # Draw polygons
        for polygon in self.polygons:
            pygame.draw.lines(self.screen, BLACK, True, polygon, 2)
            for point in polygon:
                pygame.draw.circle(self.screen, VERTEX_COLOR, point, 4)

        # Draw current polygon
        if self.current_polygon:
            if len(self.current_polygon) >= 2:
                pygame.draw.lines(self.screen, BLACK, False, self.current_polygon, 2)
            for point in self.current_polygon:
                pygame.draw.circle(self.screen, VERTEX_COLOR, point, 4)

        # Draw start and end points
        if self.start_point:
            pygame.draw.circle(self.screen, START_COLOR, self.start_point, 6)
        if self.end_point:
            pygame.draw.circle(self.screen, END_COLOR, self.end_point, 6)

        # Draw visibility path
        if self.show_visibility and len(self.visibility_path) > 1:
            pygame.draw.lines(self.screen, VISIBILITY_PATH_COLOR, False, self.visibility_path, 4)
            
            # Draw dots along the path
            for i in range(len(self.visibility_path) - 1):
                start = self.visibility_path[i]
                end = self.visibility_path[i + 1]
                
                # Draw dots
                num_dots = int(math.hypot(end[0]-start[0], end[1]-start[1]) / 20)
                for j in range(num_dots):
                    t = j / num_dots
                    x = int(start[0] + t * (end[0] - start[0]))
                    y = int(start[1] + t * (end[1] - start[1]))
                    pygame.draw.circle(self.screen, VISIBILITY_PATH_COLOR, (x, y), 2)

        # Draw legend
        self.draw_legend()

        pygame.display.flip()

    def draw_legend(self):
        """Draw the legend in the reserved legend area."""
        legend_x = GRID_DRAW_WIDTH + 10  # Adjust padding from the grid
        legend_y = 10  # Start 10 pixels from the top
        legend_spacing = 20  # Adjust spacing for better visibility
        section_spacing = 10  # Additional space between sections

        legend_items = [
            ("Instructions:", BLACK),
            ("  P: Place new polygon", BLACK),
            ("  S: Set start point", BLACK),
            ("  E: Set end point", BLACK),
            ("  V: Toggle visibility path", BLACK),
            ("  R: Reset", BLACK),
            ("", BLACK),  # Blank line for spacing
            ("Pathfinding Algorithms:", BLACK),
            ("Start Point", START_COLOR),
            ("End Point", END_COLOR),
            ("Overlapping Paths", OVERLAPPING_PATH_COLOR),
            (f"BFS ({len(self.bfs_path)} nodes, {int(self.calculate_path_length(self.bfs_path))}px)", BFS_PATH_COLOR),
            (f"  Preprocess: O(n log n) [triangulation]", BFS_PATH_COLOR),
            (f"  Runtime: O(n), Explored: {self.bfs_explored_count}", BFS_PATH_COLOR),
            (f"Dijkstra ({len(self.dijkstra_path)} nodes, {int(self.calculate_path_length(self.dijkstra_path))}px)", DIJKSTRA_PATH_COLOR),
            (f"  Preprocess: O(n log n) [triangulation]", DIJKSTRA_PATH_COLOR),
            (f"  Runtime: O(n log n), Explored: {self.dijkstra_explored_count}", DIJKSTRA_PATH_COLOR),
            (f"A* ({len(self.astar_path)} nodes, {int(self.calculate_path_length(self.astar_path))}px)", ASTAR_PATH_COLOR),
            (f"  Preprocess: O(n log n) [triangulation]", ASTAR_PATH_COLOR),
            (f"  Runtime: O(n log n), Explored: {self.astar_explored_count}", ASTAR_PATH_COLOR),
            (f"Visibility ({len(self.visibility_path)} points, {int(self.calculate_path_length(self.visibility_path))}px)", VISIBILITY_PATH_COLOR),
            (f"  Preprocess: O(n²) [visibility graph]", VISIBILITY_PATH_COLOR),
            (f"  Runtime: O(n log n), Explored: {self.visibility_explored_count}", VISIBILITY_PATH_COLOR),
        ]

        # Introductory section
        intro = [
            ("This is a demo for pathfinding in a", BLACK),
            ("2D environment with polygonal obstacles,", BLACK),
            ("using various computational geometry", BLACK),
            ("and algorithmic techniques.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("Concepts:", BLACK)
        ]

        # Add detailed descriptions with empty lines between them
        descriptions = [
            ("Delaunay Triangulation: Creates triangles", BLACK),
            ("  such that no point is inside the", BLACK),
            ("  circumcircle of any triangle. This", BLACK),
            ("  maximizes the minimum angle of", BLACK),
            ("  triangles, avoiding skinny triangles.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("Bowyer-Watson: An incremental algorithm", BLACK),
            ("  for constructing Delaunay triangulations", BLACK),
            ("  by adding points one at a time and", BLACK),
            ("  retriangulating affected areas.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("BFS: Breadth-First Search explores nodes", BLACK),
            ("  layer by layer, ensuring the shortest", BLACK),
            ("  path in unweighted graphs. It uses a", BLACK),
            ("  queue to track exploration.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("Dijkstra: Finds shortest paths from a", BLACK),
            ("  source to all vertices using a priority", BLACK),
            ("  queue. It is optimal for graphs with", BLACK),
            ("  non-negative weights.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("A*: A heuristic-based search algorithm", BLACK),
            ("  that combines Dijkstra's and greedy", BLACK),
            ("  best-first search. It uses heuristics", BLACK),
            ("  to improve pathfinding efficiency.", BLACK),
            ("", BLACK),  # Empty line for spacing
            ("Visibility: Constructs a graph of visible", BLACK),
            ("  vertices and finds shortest paths using", BLACK),
            ("  Dijkstra's algorithm. It is useful for", BLACK),
            ("  pathfinding in polygonal environments.", BLACK)
        ]

        font = pygame.font.Font(None, 20)
        
        # Draw legend items
        for i, (text, color) in enumerate(legend_items):
            y = legend_y + i * legend_spacing
            
            if not text.startswith("  ") and not text.endswith(":") and text != "":
                if len(color) == 4:  # If color has alpha
                    pygame.draw.rect(self.screen, color, (legend_x, y, 15, 15))
                else:
                    pygame.draw.circle(self.screen, color, (legend_x + 7, y + 7), 5)
                text_x = legend_x + 20
            else:
                text_x = legend_x  # No offset for other lines
            
            text_surface = font.render(text, True, BLACK)
            self.screen.blit(text_surface, (text_x, y))

        # Draw introductory section
        intro_y = legend_y + len(legend_items) * legend_spacing + section_spacing
        for i, (text, color) in enumerate(intro):
            y = intro_y + i * legend_spacing
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (legend_x, y))

        # Draw descriptions below the introductory section
        description_y = intro_y + len(intro) * legend_spacing + section_spacing
        for i, (text, color) in enumerate(descriptions):
            y = description_y + i * (legend_spacing - 5)  # Slightly tighter spacing for descriptions
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (legend_x, y))

    async def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:  # Start new polygon
                        self.building_polygon = True
                        self.current_polygon = []
                    elif event.key == pygame.K_s:  # Place start point
                        self.placing_start = True
                        self.placing_end = False
                    elif event.key == pygame.K_e:  # Place end point
                        self.placing_end = True
                        self.placing_start = False
                    elif event.key == pygame.K_v:  # Toggle visibility path
                        self.show_visibility = not self.show_visibility
                        self.visibility_explored_count = 0  # Reset explored count
                        if self.show_visibility and self.start_point and self.end_point:
                            self.find_path()
                    elif event.key == pygame.K_r:  # Reset
                        self.polygons = [self.polygons[0]]  # Keep boundary
                        self.current_polygon = []
                        self.building_polygon = False
                        self.valid_triangles = []
                        self.start_point = None
                        self.end_point = None
                        self.bfs_path = []
                        self.dijkstra_path = []
                        self.astar_path = []
                        self.visibility_path = []
                        self.visibility_explored_count = 0
                        self.visibility_path_length = 0
                        self.show_visibility = False

            self.draw()
            await asyncio.sleep(0.016)  # Add this for browser compatibility

        pygame.quit()

    def calculate_path_distance(self, path):
        """Calculate total Euclidean distance of a path."""
        if not path or len(path) < 2:
            return 0
        
        total_distance = 0
        # Add distance from start point to first centroid
        if self.start_point:
            first_centroid = self.triangle_centroids[path[0]]
            total_distance += np.linalg.norm(
                np.array(self.start_point) - first_centroid
            )
        
        # Add distances between centroids
        for i in range(len(path) - 1):
            current_centroid = self.triangle_centroids[path[i]]
            next_centroid = self.triangle_centroids[path[i + 1]]
            total_distance += np.linalg.norm(current_centroid - next_centroid)
        
        # Add distance from last centroid to end point
        if self.end_point:
            last_centroid = self.triangle_centroids[path[-1]]
            total_distance += np.linalg.norm(
                last_centroid - np.array(self.end_point)
            )
        
        return total_distance

    def line_segments_intersect(self, p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def calculate_path_length(self, path):
        """Calculate the total length of a path in pixels."""
        if not path:
            return 0
        
        if isinstance(path[0], int):  # Triangle-based paths (BFS, Dijkstra, A*)
            # Convert triangle indices to centroid points
            points = [self.start_point]  # Start with start point
            points.extend(self.triangle_centroids[i] for i in path)
            points.append(self.end_point)  # End with end point
        else:  # Visibility path (already points)
            points = path
        
        # Calculate total length
        length = 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length

    def draw_grid(self):
        """Draw the grid excluding the legend area."""
        for x in range(0, GRID_DRAW_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (GRID_DRAW_WIDTH, y))

    def handle_resize(self, event):
        """Handle window resize events."""
        global WIDTH, HEIGHT, GRID_DRAW_WIDTH
        WIDTH, HEIGHT = event.w, event.h
        GRID_DRAW_WIDTH = WIDTH - LEGEND_WIDTH
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.polygons[0] = [
            (GRID_SIZE, GRID_SIZE),
            (GRID_DRAW_WIDTH - GRID_SIZE, GRID_SIZE),
            (GRID_DRAW_WIDTH - GRID_SIZE, HEIGHT - GRID_SIZE),
            (GRID_SIZE, HEIGHT - GRID_SIZE)
        ]
        self.compute_triangulation()  # Recompute triangulation if necessary

if __name__ == "__main__":
    demo = DelaunayDemo()
    
    # Set up and run the asyncio event loop
    async def main():
        await demo.run()
    
    asyncio.run(main())
