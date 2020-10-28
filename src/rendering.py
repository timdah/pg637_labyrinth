import pygame
import pygame.draw
import pygame.freetype
import pygame.font
from src import environment
from src.monte_carlo import MonteCarlo, MonteCarloWithoutES, MonteCarloExploringStart

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

cell_size = 96, 96
wall_size = 5
map_size = environment.field_length * 96, environment.field_length * 96

pygame.init()
pygame.display.set_caption('Labyrinth PG637')

screen = pygame.display.set_mode(map_size)
surface = pygame.Surface(map_size)
# font = pygame.freetype.Font(pygame.font.get_default_font(), 16)
font = pygame.font.SysFont("Trebuchet MS", 16)

wall_offsets = {
    'left': (0, 0, wall_size, cell_size[1]),
    'right': (cell_size[0] - wall_size, 0, wall_size, cell_size[1]),
    'up': (0, 0, cell_size[0], wall_size),
    'down': (0, cell_size[1] - wall_size, cell_size[0], wall_size)
}


def render(position_id, value_map=None):
    global screen

    background = surface.copy()
    background.fill(WHITE)

    for pid in environment.position_ids:
        x = (pid % environment.field_length) * cell_size[0]
        y = (pid // environment.field_length) * cell_size[1]

        wall_directions = [d for d in ['left', 'right', 'up', 'down'] if d not in environment.get_valid_directions(pid)]
        
        for direction in wall_directions:
            wall_offset = wall_offsets[direction]
            wall_rect = pygame.Rect(x + wall_offset[0], y + wall_offset[1], wall_offset[2], wall_offset[3])

            pygame.draw.rect(background, BLACK, wall_rect, 0)

        if pid == position_id:
            pygame.draw.circle(background, BLUE, (x + (cell_size[0] // 2), y + (cell_size[1] // 2)), 20)

        if pid == environment.trap_id:
            pygame.draw.circle(background, RED, (x + (cell_size[0] // 2), y + (cell_size[1] // 2)), 20)

        if pid == environment.exit_id:
            pygame.draw.circle(background, GREEN, (x + (cell_size[0] // 2), y + (cell_size[1] // 2)), 20)

        # if value_map is not None:
        #     text = font.render(str(value_map[pid]), False, (0, 0, 0))
        #     background.blit(text, (x,y))

    screen.blit(background, (0, 0))
    pygame.display.flip()


def shutdown():
    pygame.quit()


def get_policy_and_start_position(method: MonteCarlo, episodes: int, random_start: bool) -> tuple:
    policy, values = method.generate_monte_carlo_policy(episodes)
    start_position = MonteCarloExploringStart.get_random_start_position() if random_start else environment.entry_id
    return policy, start_position, values


# BEISPIEL:

clock = pygame.time.Clock()
running = True

# Hard labyrinth
environment.exit_id = 5
environment.trap_id = 2
mc_without_es = MonteCarloWithoutES(epsilon=0.6, gamma=0.9)

# Custom labyrinth
# environment.exit_id = 17
# environment.trap_id = 33
# mc_without_es = MonteCarloWithoutES(epsilon=0.7, gamma=0.9)

# mc_exploring_start = MonteCarloExploringStart(gamma=0.9)

steps = 50
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if steps >= 50:
        mc_policy, position, values = get_policy_and_start_position(mc_without_es, episodes=200, random_start=False)
        print(f"new policy: {mc_policy}")
        environment.prettyprint(values)
        steps = 0
        render(position)
        pygame.time.wait(1000)

    clock.tick(60)
    render(position, values)
    pygame.time.wait(150)
    if position == environment.trap_id:
        print("You lost!")
        steps = 50
        # running = False
    elif position == environment.exit_id:
        print("You won!")
        steps = 50
        # running = False
    next_action = mc_policy[position]
    position = environment.next_position_functions[next_action](position)
    steps = steps + 1


