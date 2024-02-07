from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pygame
import pygame.font

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1900, 1100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_SIZE = 300

# Set up the display
window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Infinite Grid with Graphs")


def generate_power_law_graph(num_nodes, exponent, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    degrees = np.random.zipf(a=exponent, size=num_nodes)
    degrees = [d for d in degrees if 0 < d < num_nodes]
    if sum(degrees) % 2 == 1:
        degrees[-1] += 1

    # print(f"Max Degree: {np.max(degrees)} - Mean {np.mean(degrees)}")

    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    largest_cc = max(nx.connected_components(G), key=len)
    G_largest_cc = G.subgraph(largest_cc).copy()

    # print(f"Number of nodes in the largest connected component: {G_largest_cc.number_of_nodes()}")

    return G_largest_cc


class Graph:
    def __init__(self, center):
        self.center = center

        min_exp = 1.9
        self.exponent = max(min_exp, min_exp + int(center[1]) * 0.02)
        self.num_nodes = max(20, int(center[0]) * 5)
        seed = (self.num_nodes + int(737331 * self.exponent)) % 2**32
        self.G = generate_power_law_graph(num_nodes=self.num_nodes, exponent=self.exponent, seed=seed)
        self.positions = nx.spring_layout(self.G)  # get positions for all nodes
        self.scale_factor = 1 / 3

    def draw(self, window, offset):
        def to_screen(abs_coord):
            center = self.center * GRID_SIZE + offset
            return center + abs_coord * GRID_SIZE

        for node in self.G.nodes:
            pygame.draw.circle(window, (0, 0, 0), tuple(to_screen(self.positions[node] * self.scale_factor)), 2)

        for edge in self.G.edges:
            pygame.draw.line(
                window,
                (0, 0, 0),
                tuple(to_screen(self.positions[edge[0]] * self.scale_factor)),
                tuple(to_screen(self.positions[edge[1]] * self.scale_factor)),
                1,
            )

        font = pygame.font.Font(None, 25)  # Choose the font, and size
        text = font.render(f"n={self.num_nodes}, e={self.exponent:.3f}", True, (0, 0, 0))  # The color is white
        text_rect = text.get_rect(
            center=(self.center * GRID_SIZE + offset + np.array([0, -GRID_SIZE / 2.4]))
        )  # center the text
        window.blit(text, text_rect)


def draw_grid(offset, graphs):
    window.fill(WHITE)

    # Define the number of cells in the grid
    num_cells_x = WIDTH // GRID_SIZE + 2
    num_cells_y = HEIGHT // GRID_SIZE + 2

    # Calculate grid offset in terms of cells
    offset_cells = offset // GRID_SIZE

    # Iterate over absolute positions on the grid
    for abs_int_x in range(-offset_cells[0] - 1, -offset_cells[0] + num_cells_x):
        for abs_int_y in range(-offset_cells[1] - 1, -offset_cells[1] + num_cells_y):
            abs_int = (abs_int_x, abs_int_y)
            # Calculate the relative position on the display
            rel_pos = np.array(abs_int) * GRID_SIZE + offset

            # Create and draw the grid rectangle
            rect = pygame.Rect(*rel_pos, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(window, BLACK, rect, 1)

            # Check if the graph at this absolute position needs to be generated
            if abs_int not in graphs:
                graphs[abs_int] = Graph(np.array(abs_int))

            # Draw the graph
            graphs[abs_int].draw(window, offset - GRID_SIZE // 2)


def main():
    run = True
    clock = pygame.time.Clock()
    offset = np.array([0, 0])  # Offset as a NumPy array
    mouse_down = False
    last_mouse_pos = None
    graphs = {}
    global GRID_SIZE  # Make the grid size global so we can modify it

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                # Check if Ctrl and C are pressed
                if event.mod & pygame.KMOD_CTRL and event.key == pygame.K_c:
                    run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel scrolled up
                    GRID_SIZE = min(10000, GRID_SIZE + 5)  # Increase the grid size, limit it to 200
                elif event.button == 5:  # Mouse wheel scrolled down
                    GRID_SIZE = max(200, GRID_SIZE - 5)  # Decrease the grid size, limit it to 10

                mouse_down = True
                last_mouse_pos = np.array(pygame.mouse.get_pos())  # Convert to NumPy array
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False

        if mouse_down:
            mouse_pos = np.array(pygame.mouse.get_pos())  # Convert to NumPy array
            if last_mouse_pos is not None:
                offset += mouse_pos - last_mouse_pos  # Simplified with NumPy array operation
            last_mouse_pos = mouse_pos

        draw_grid(offset, graphs)  # Updated to use the NumPy array offset
        pygame.display.update()
        clock.tick(40)

    pygame.quit()


main()
