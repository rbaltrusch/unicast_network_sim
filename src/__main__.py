# pylint: disable=missing-docstring
# pylint: disable=no-member
# pylint: disable=c-extension-no-member
# pylint: disable=import-error

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Protocol, Tuple

import pygame

from src.coordinate import Coordinate

SCREEN_SIZE = Coordinate(800, 600)
BOARD_SIZE = Coordinate(700, 500)


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
    min: Optional[float] = None

    def __call__(self) -> float:
        """Returns a random number taken from the a normal distribution
        defined by this Stat's average and standard deviation.
        """
        deviation = 3 * self.standard_deviation
        number = random.randint(
            int(self.average - deviation), int(self.average + deviation)
        )
        return max(self.min, number) if self.min is not None else number


class Entity(Protocol):
    def update(self, game: Game):
        ...

    def render(self, screen: pygame.surface.Surface):
        ...


@dataclass
class Node:
    position: Coordinate
    throughput: float
    connections: List[Connection] = field(default_factory=list)

    def __hash__(self):
        return hash(self.position)

    def update(self, game: Game):
        pass

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(
            *tuple(self.position), int(self.throughput), int(self.throughput)
        )  # type: ignore
        color = (255, 255, 255)
        pygame.draw.rect(screen, color, rect, width=0)


def calculate_angle(coord1: Coordinate, coord2: Coordinate) -> float:
    xdiff = coord1.x - coord2.x
    ydiff = coord1.y - coord2.y
    return math.atan(ydiff / xdiff)


@dataclass
class Player:
    position: Coordinate
    target_node: Node
    speed: float

    def update(self, game: Game):
        self._move()

    def _move(self):
        if self.reached_target:
            return

        dist = self.target_node.position.compute_distance(self.position)
        if self.speed >= dist:
            self.position = self.target_node.position.clone()
            return

        angle = calculate_angle(self.position, self.target_node.position)
        self.position.x += (
            abs(math.cos(angle))
            * self.speed
            * (1 if self.position.x <= self.target_node.position.x else -1)
        )
        self.position.y += (
            abs(math.sin(angle))
            * self.speed
            * (1 if self.position.y <= self.target_node.position.y else -1)
        )
        print(angle, self.position, self.target_node.position)

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(*tuple(self.position), 25, 25)  # type: ignore
        color = (255, 0, 0)
        pygame.draw.rect(screen, color, rect, width=0)

        # target node
        rect = pygame.Rect(*tuple(self.target_node.position), 25, 25)  # type: ignore
        color = (100, 255, 100)
        pygame.draw.rect(screen, color, rect, width=0)

    def set_target_connection(self, index: int):
        if len(self.target_node.connections) <= index:
            return
        connection = self.target_node.connections[index]
        self.target_node = (
            connection.end if connection.start is self.target_node else connection.start
        )

    @property
    def reached_target(self) -> bool:
        return bool(tuple(self.position) == tuple(self.target_node.position))


@dataclass
class Connection:
    start: Node
    end: Node

    def __hash__(self):
        return hash((self.start, self.end))

    def update(self, game: Game):
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
    player: Player

    @property
    def entities(self) -> Iterable[Entity]:
        yield from self.nodes
        yield from self.connections
        yield self.player


@dataclass
class Params:
    amount: Stat
    throughput: Stat
    connections_per_node: Stat
    node_offset: Stat
    max_connection_length: float
    player_speed: float


def init_game(params: Params) -> Game:
    nodes = spawn_nodes(params)
    connections = spawn_connections(nodes, params)
    conn_dict = defaultdict(set)
    for connection in connections:
        conn_dict[connection.start].add(connection)
        conn_dict[connection.end].add(connection)

    for node in nodes:
        node.connections = list(conn_dict.get(node, set()))

    nodes = [x for x in nodes if x.connections]

    target_node = min(nodes, key=lambda x: x.position.x)
    player = Player(
        position=target_node.position.clone(),
        target_node=target_node,
        speed=params.player_speed,
    )
    return Game(nodes=nodes, connections=connections, player=player)


def compute_node_distances(
    nodes: Iterable[Node],
) -> Mapping[Node, List[Tuple[float, Node]]]:
    distances: Mapping[Node, List[Tuple[float, Node]]] = defaultdict(list)
    for node1, node2 in itertools.combinations(nodes, 2):
        distances[node1].append(
            (node1.position.compute_distance(node2.position), node2)
        )
    print(distances)
    for distvalues in distances.values():
        distvalues.sort(key=lambda x: x[0])
    return distances


def spawn_connections(nodes: Iterable[Node], params: Params) -> Iterable[Connection]:
    # TODO: no crossing lines
    # TODO: min connection angle difference
    distances = compute_node_distances(nodes)
    connections = []
    for node in nodes:
        for _ in range(int(params.connections_per_node())):
            possible_connections = [
                (dist, node)
                for dist, node in distances[node][:5]
                if dist < params.max_connection_length
            ]
            if not possible_connections:
                break
            target_node = possible_connections.pop(
                random.choice(list(range(len(possible_connections))))
            )[1]
            connection = Connection(start=node, end=target_node)
            connections.append(connection)
    return connections


def spawn_nodes(params: Params) -> Iterable[Node]:
    screen_offset = 20

    min_step = 100
    grid = list(
        itertools.product(
            range(screen_offset, int(BOARD_SIZE.x), min_step),
            range(screen_offset, int(BOARD_SIZE.y), min_step),
        )
    )

    return [
        Node(
            position=Coordinate(x + params.node_offset(), y + params.node_offset()),
            throughput=params.throughput(),
        )
        for x, y in random.sample(grid, k=min(len(grid), int(params.amount())))
    ]

    # def saturate(num, screen_max):
    #     return max(screen_offset, min(num, screen_max - screen_offset))

    # area = BOARD_SIZE.x * BOARD_SIZE.y
    # area /
    # amount = round(
    #     math.sqrt(params.amount())
    # )  # HACK: only works correctly for square screen
    # spacing = (SCREEN_SIZE.x - screen_offset * 1) / amount

    # nodes = []
    # for x, y in itertools.product(range(amount), range(amount)):
    #     position = Coordinate(
    #         x=saturate((x + 1) * spacing + params.node_offset(), SCREEN_SIZE.x),
    #         y=saturate((y + 0) * spacing + params.node_offset(), SCREEN_SIZE.y),
    #     )
    #     node = Node(position, throughput=params.throughput())
    #     nodes.append(node)
    # return nodes

    # return [
    #     Node(
    #         position=generate_random_position(screen_offset),
    #         throughput=params.throughput(),
    #     )
    #     for _ in range(int(params.amount()))
    # ]


def main():
    pygame.init()
    screen = pygame.display.set_mode(tuple(SCREEN_SIZE))
    board = pygame.Surface(tuple(BOARD_SIZE))
    clock = pygame.time.Clock()

    level_one_params = Params(
        amount=Stat(30, 3, 25),
        throughput=Stat(10, 0, 10),
        connections_per_node=Stat(1, 1, 1),
        node_offset=Stat(0, 5),
        max_connection_length=350,
        player_speed=10,
    )
    game = init_game(level_one_params)

    in_game = True
    terminated = False
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and in_game:
                if game.player.reached_target:
                    if event.key == pygame.K_1:
                        game.player.set_target_connection(0)
                    if event.key == pygame.K_2:
                        game.player.set_target_connection(1)
                    if event.key == pygame.K_3:
                        game.player.set_target_connection(2)
                    if event.key == pygame.K_4:
                        game.player.set_target_connection(3)
                    if event.key == pygame.K_5:
                        game.player.set_target_connection(4)

        for entity in game.entities:
            entity.update(game)

        screen.fill((0, 0, 0))
        board.fill((100, 100, 100))
        for entity in game.entities:
            entity.render(board)
        screen.blit(board, (100, 100))
        pygame.display.flip()
        clock.tick(50)

    pygame.display.quit()


if __name__ == "__main__":
    main()
