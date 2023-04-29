# pylint: disable=missing-docstring
# pylint: disable=no-member
# pylint: disable=c-extension-no-member
# pylint: disable=import-error

import itertools
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Protocol

import pygame

from src.coordinate import Coordinate

SCREEN_SIZE = Coordinate(800, 600)


def generate_random_position(screen_offset: int = 0) -> Coordinate:
    """Generates a random position on the screen. The minimum distance to the screen borders
    can be specified using the optional screen_offset parameter.
    """
    return Coordinate(
        random.randint(screen_offset, int(SCREEN_SIZE.x) - screen_offset),
        random.randint(screen_offset, int(SCREEN_SIZE.y) - screen_offset),
    )


@dataclass
class Stat:
    """Encapsulates parameters to draw random values from a normal distribution"""

    average: float
    standard_deviation: float = 0
    min: float = 0

    def __call__(self) -> float:
        """Returns a random number taken from the a normal distribution
        defined by this Stat's average and standard deviation.
        """
        deviation = 3 * self.standard_deviation
        return max(
            self.min,
            random.randint(
                int(self.average - deviation), int(self.average + deviation)
            ),
        )


class Entity(Protocol):
    def update(self):
        ...

    def render(self, screen: pygame.surface.Surface):
        ...


@dataclass
class Node:
    position: Coordinate
    throughput: float

    def __hash__(self):
        return hash(self.position)

    def update(self):
        pass

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(
            *tuple(self.position), int(self.throughput), int(self.throughput)
        )  # type: ignore
        color = (255, 255, 255)
        pygame.draw.rect(screen, color, rect, width=0)


@dataclass
class Connection:
    start: Node
    end: Node

    def update(self):
        pass

    def render(self, screen: pygame.surface.Surface):
        color = (255, 255, 255)
        pygame.draw.line(
            screen, color, tuple(self.start.position), tuple(self.end.position), width=1
        )


@dataclass
class Game:
    nodes: Iterable[Node]
    connections: Iterable[Connection]

    @property
    def entities(self) -> Iterable[Entity]:
        yield from self.nodes
        yield from self.connections


@dataclass
class Params:
    amount: Stat
    throughput: Stat
    connections_per_node: Stat


def init_game(params: Params) -> Game:
    nodes = spawn_nodes(params)
    return Game(nodes=nodes, connections=spawn_connections(nodes, params))


def spawn_connections(nodes: Iterable[Node], params: Params) -> Iterable[Connection]:
    # TODO: no crossing lines

    distances = defaultdict(list)
    for node1, node2 in itertools.combinations(nodes, 2):
        distances[node1].append(
            (node1.position.compute_distance(node2.position), node2)
        )
    for distvalues in distances.values():
        distvalues.sort()

    connections = []
    for node in nodes:
        for _ in range(int(params.connections_per_node())):
            possible_connections = distances[node][:5]
            if not possible_connections:
                break
            target_node = possible_connections.pop(
                random.choice(list(range(len(possible_connections))))
            )[1]
            connection = Connection(start=node, end=target_node)
            connections.append(connection)
    return connections


def spawn_nodes(params: Params) -> Iterable[Node]:
    # more spaced out
    screen_offset = 50
    return [
        Node(
            position=generate_random_position(screen_offset),
            throughput=params.throughput(),
        )
        for _ in range(int(params.amount()))
    ]


def main():
    pygame.init()
    screen = pygame.display.set_mode(tuple(SCREEN_SIZE))
    clock = pygame.time.Clock()

    level_one_params = Params(
        amount=Stat(20, 3, 15),
        throughput=Stat(10, 0, 10),
        connections_per_node=Stat(3, 1, 0),
    )
    game = init_game(level_one_params)

    terminated = False
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        clock.tick(50)
        screen.fill((0, 0, 0))
        for node in game.entities:
            node.render(screen)
        pygame.display.flip()
    pygame.display.quit()


if __name__ == "__main__":
    main()
