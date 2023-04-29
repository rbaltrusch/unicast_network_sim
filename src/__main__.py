# pylint: disable=missing-docstring
# pylint: disable=no-member
# pylint: disable=c-extension-no-member
# pylint: disable=import-error
# pylint: disable=global-statement
# pylint: disable=invalid-name

from __future__ import annotations

import itertools
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Optional, Protocol, Tuple, Union

import pygame

from src.coordinate import Coordinate

SCREEN_SIZE = Coordinate(800, 600)
BOARD_SIZE = Coordinate(700, 500)
MAX_CONNECTIONS = 5
MAX_THROUGHPUT = 25
MAX_COMBO = 5
MIN_COMBO = 0
MAIL_OFFSET = Coordinate(25, 0)

font: pygame.font.Font = None  # type: ignore
gui_font: pygame.font.Font = None  # type: ignore


def chance(factor: float) -> bool:
    return random.random() < factor


def saturate(num: float, min_: float, max_: float):
    return max(min_, min(num, max_))


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
class Mail:
    parent: Union[Node, Player]
    target_node: Node
    size: float

    def update(self, game: Game):
        pass

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(
            *tuple(self.position), int(self.size), int(self.size)
        )  # type: ignore
        color = (0, 0, 255)
        pygame.draw.rect(screen, color, rect, width=0)

    @property
    def reached_target(self) -> bool:
        return bool(tuple(self.parent.position) == tuple(self.target_node.position))

    @property
    def position(self) -> Coordinate:
        return self.parent.position + MAIL_OFFSET

    @property
    def points(self) -> float:
        return self.size


@dataclass
class Node:
    position: Coordinate
    throughput: float
    mail: Optional[Mail] = None
    connections: List[Connection] = field(default_factory=list)

    def __hash__(self):
        return hash(self.position)

    def get_mail(self, player: Player):
        if self.mail and not player.mail:
            player.mail = self.mail
            self.mail.parent = player
        self.mail = None

    def update(self, game: Game):
        spawn_chance = self.throughput / MAX_THROUGHPUT * game.mail_spawn_factor
        if not self.mail and chance(spawn_chance):
            self.mail = Mail(
                parent=self,
                size=random.randint(5, 15),
                target_node=random.choice(list(x for x in game.nodes if x is not self)),
            )

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(*tuple(self.position), self.size, self.size)  # type: ignore
        color = (255, 255, 255)
        pygame.draw.rect(screen, color, rect, width=0)
        if self.mail:
            self.mail.render(screen)

    @property
    def size(self) -> int:
        return int(self.throughput)


def calculate_angle(coord1: Coordinate, coord2: Coordinate) -> float:
    xdiff = coord1.x - coord2.x
    ydiff = coord1.y - coord2.y
    return math.atan(ydiff / xdiff if xdiff else math.inf)


@dataclass
class Player:
    position: Coordinate
    target_node: Node
    speed: float
    mail: Optional[Mail] = None

    def update(self, game: Game):
        self._move()
        if self.mail and self.mail.reached_target:
            game.score.update_score(self.mail)
            self.mail = None

    def _move(self):
        if self.reached_target:
            return

        dist = self.target_node.position.compute_distance(self.position)
        if self.speed >= dist:
            self.position = self.target_node.position.clone()
            self.target_node.get_mail(self)
            return

        travel_coord = self._calculate_travel_coordinates(
            self.speed, self.position, self.target_node.position
        )
        self.position.x += travel_coord.x
        self.position.y += travel_coord.y

    @staticmethod
    def _calculate_travel_coordinates(speed, position, target_position) -> Coordinate:
        angle = calculate_angle(position, target_position)
        x = (
            abs(math.cos(angle))
            * speed
            * (1 if position.x <= target_position.x else -1)
        )
        y = (
            abs(math.sin(angle))
            * speed
            * (1 if position.y <= target_position.y else -1)
        )
        return Coordinate(x, y)

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(*tuple(self.position), 25, 25)  # type: ignore
        color = (255, 0, 0)
        pygame.draw.rect(screen, color, rect, width=0)

        # target node
        rect = pygame.Rect(*tuple(self.target_node.position), 25, 25)  # type: ignore
        color = (200, 255, 200)
        pygame.draw.rect(screen, color, rect, width=0)

        # mail target
        self._render_mail_target(screen)

        # connected node numbers
        global font
        if font is None:
            return
        for i, node in enumerate(self.connected_nodes, 1):
            travel_coord = self._calculate_travel_coordinates(
                self.speed * 3, self.position, node.position
            )
            surf = font.render(str(i), True, (255, 255, 255))
            screen.blit(surf, tuple(travel_coord + self.position))

    def _render_mail_target(self, screen: pygame.surface.Surface):
        if self.mail is None:
            return
        node_size = self.mail.target_node.size
        rect = pygame.Rect(*tuple(self.mail.target_node.position), node_size, node_size)  # type: ignore
        color = (100, 255, 100)
        pygame.draw.rect(screen, color, rect, width=0)

    def set_target_connection(self, index: int):
        if len(self.target_node.connections) <= index:
            return
        self.target_node = self.connected_nodes[index]

    @property
    def connected_nodes(self) -> List[Node]:
        return sorted(
            [
                x.end if x.start is self.target_node else x.start
                for x in self.target_node.connections
            ],
            key=lambda x: calculate_angle(self.position, x.position),
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
class Score:
    game: Game
    combo_factor_decrease_per_tick: float  # negative
    combo_factor: float = 1
    points: int = 0

    def update(self, game: Game):
        self.combo_factor = saturate(
            self.combo_factor + self.combo_factor_decrease_per_tick,
            min_=MIN_COMBO,
            max_=MAX_COMBO,
        )

    def update_score(self, mail: Mail):
        points = mail.points
        self.points += round(points * self.combo_factor)
        self.combo_factor = saturate(
            self.combo_factor + 1, min_=MIN_COMBO, max_=MAX_COMBO
        )

    def render(self, screen: pygame.surface.Surface):
        global gui_font
        if gui_font is None:
            return
        text = f"Points: {self.points}     Combo: {self.combo_factor:.2f}"
        surf = gui_font.render(text, True, (255, 255, 255))
        screen.blit(surf, (0, 0))


def game_over_by_zero_combo(game: Game) -> bool:
    return game.score.combo_factor <= 0


@dataclass
class Game:
    nodes: Iterable[Node]
    connections: Iterable[Connection]
    player: Player
    mail_spawn_factor: float
    game_over_strategy: Callable[[Game], bool]
    score: Score = None  # type: ignore

    @property
    def entities(self) -> Iterable[Entity]:
        yield from self.nodes
        yield from self.connections
        yield self.player

    @property
    def over(self) -> bool:
        return self.game_over_strategy(self)

    @property
    def gui_entities(self) -> Iterable[Entity]:
        yield self.score


@dataclass
class Params:
    amount: Stat
    throughput: Stat
    connections_per_node: Stat
    node_offset: Stat
    max_connection_length: float
    player_speed: float
    mail_size: Stat
    mail_spawn_factor: float
    initial_mail: int
    combo_factor: float
    combo_factor_decrease_per_tick: float


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
    for node in nodes:
        node.connections = node.connections[:MAX_CONNECTIONS]

    target_node = min(nodes, key=lambda x: x.position.x)
    player = Player(
        position=target_node.position.clone(),
        target_node=target_node,
        speed=params.player_speed,
    )

    game = Game(
        nodes=nodes,
        connections=connections,
        player=player,
        mail_spawn_factor=params.mail_spawn_factor,
        game_over_strategy=game_over_by_zero_combo,
    )
    game.score = Score(
        game,
        combo_factor=params.combo_factor,
        combo_factor_decrease_per_tick=params.combo_factor_decrease_per_tick,
    )
    return game


def compute_node_distances(
    nodes: Iterable[Node],
) -> Mapping[Node, List[Tuple[float, Node]]]:
    distances: Mapping[Node, List[Tuple[float, Node]]] = defaultdict(list)
    for node1, node2 in itertools.combinations(nodes, 2):
        distances[node1].append(
            (node1.position.compute_distance(node2.position), node2)
        )
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
                for dist, node in distances[node][:MAX_CONNECTIONS]
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print([calculate_angle(Coordinate(0, 0), Coordinate(-10, -10))])
    screen = pygame.display.set_mode(tuple(SCREEN_SIZE))
    board = pygame.Surface(tuple(BOARD_SIZE))
    clock = pygame.time.Clock()

    # TODO: expose as ini params
    font_size = 25
    gui_font_size = 40

    global font, gui_font
    try:
        font = pygame.font.SysFont(name=pygame.font.get_default_font(), size=font_size)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init font due to exception %s", str(exc))

    try:
        gui_font = pygame.font.SysFont(
            name=pygame.font.get_default_font(), size=gui_font_size
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init gui font due to exception %s", str(exc))

    level_one_params = Params(
        amount=Stat(30, 3, 25),
        throughput=Stat(10, 0, 10),
        connections_per_node=Stat(1, 1, 1),
        node_offset=Stat(0, 5),
        max_connection_length=350,
        player_speed=10,
        mail_size=Stat(10, 3, 2),
        mail_spawn_factor=0.005,
        initial_mail=1,
        combo_factor=1,
        combo_factor_decrease_per_tick=-0.001,
    )
    params = level_one_params

    game = init_game(params)
    in_game = True
    terminated = False
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and in_game:
                if game.player.reached_target and not game.over:
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
                if game.over and event.key == pygame.K_RETURN:  # restart
                    game = init_game(params)

        for entity in itertools.chain(game.entities, game.gui_entities):
            entity.update(game)

        screen.fill((0, 0, 0))
        board.fill((100, 100, 100))
        for entity in game.entities:
            entity.render(board)
        for entity in game.gui_entities:
            entity.render(screen)
        screen.blit(board, (100, 100))
        if game.over:
            surf = gui_font.render(
                "Game Over! Press ENTER to restart.", True, (255, 255, 255)
            )
            *_, width, height = surf.get_rect()
            screen.blit(
                surf, ((SCREEN_SIZE.x - width) / 2, (SCREEN_SIZE.y - height) / 2)
            )
        pygame.display.flip()
        clock.tick(50)

    pygame.display.quit()


if __name__ == "__main__":
    main()
