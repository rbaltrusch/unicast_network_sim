# pylint: disable=missing-docstring
# pylint: disable=no-member
# pylint: disable=c-extension-no-member
# pylint: disable=import-error
# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=line-too-long

from __future__ import annotations

import enum
import itertools
import logging
import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import pygame

from src.coordinate import Coordinate
from src.particle import CircleParticle, Colour, ParticleSystem, RectParticle

SCREEN_SIZE = Coordinate(800, 600)
BOARD_SIZE = Coordinate(700, 500)
MAX_CONNECTIONS = 5
MAX_THROUGHPUT = 25
MAX_COMBO = 5
MIN_COMBO = 0
MIN_MAIL_SIZE = 5
MAIL_OFFSET = Coordinate(25, 0)
MAIL_TARGET_COLOR = Colour(100, 255, 100, 0)
SAVE_FILEPATH = "mailgame.sav"

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
        self.size -= game.mail_size_decay

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(
            *tuple(self.position), int(self.size), int(self.size)
        )  # type: ignore
        color = (100, 100, 255)
        # pygame.draw.circle(screen, color, tuple(self.position), self.size)
        pygame.draw.rect(screen, color, rect, width=0)

    @property
    def reached_target(self) -> bool:
        return bool(tuple(self.parent.position) == tuple(self.target_node.position))

    @property
    def position(self) -> Coordinate:
        return self.parent.position + MAIL_OFFSET

    @property
    def points(self) -> float:
        return self.size * 2

    @property
    def expired(self) -> bool:
        return self.size <= MIN_MAIL_SIZE


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
            player.take_mail(self)
        self.mail = None

    def update(self, game: Game):
        spawn_chance = self.throughput / MAX_THROUGHPUT * game.mail_spawn_factor
        if self.mail:
            self.mail.update(game)
            if self.mail.expired:
                self.mail = None

        if not self.mail and chance(spawn_chance):
            self.spawn_mail(game)

    def spawn_mail(self, game: Game):
        self.mail = Mail(
            parent=self,
            size=random.randint(7, 14),
            target_node=random.choice(list(x for x in game.nodes if x is not self)),
        )

    def render(self, screen: pygame.surface.Surface):
        # rect = pygame.Rect(*tuple(self.position), self.size, self.size)  # type: ignore
        # color = (255, 255, 255)
        color = (255, 255, 255)
        # pygame.draw.rect(screen, color, rect, width=0)
        pygame.draw.circle(screen, color, tuple(self.position), self.size)
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
class Particle:
    color: Tuple[float, float, float, float]


@dataclass
class Player:
    position: Coordinate
    target_node: Node
    speed: float
    mail: Optional[Mail] = None

    def __post_init__(self):
        self.origin_node: Node = self.target_node
        self.target_node_particle_system = ParticleSystem(
            CircleParticle,
            position=Coordinate(),
            colour=MAIL_TARGET_COLOR,
            spawn_rate=0.3,
            expired=True,
        )
        self.target_node_particle_system.add_kwargs(
            size_drift=1, width=2, alpha_drift=20
        )

    def take_mail(self, node: Node):
        if not node.mail:
            return
        self.mail = node.mail
        self.mail.parent = self
        self.target_node_particle_system = self.target_node_particle_system.clone(
            self.mail.target_node.position.clone()
        )
        self.target_node_particle_system.add_kwargs(size=self.mail.target_node.size)

    def update(self, game: Game):
        self._move()
        if self.mail:
            self.mail.update(game)
            if self.mail.expired:
                self.mail = None
            if not self.target_node_particle_system.expired:
                self.target_node_particle_system.update()

        if self.mail and self.mail.reached_target and not game.over:
            game.score.update_score(self.mail)
            self.mail = None

    def _move(self):
        if self.reached_target:
            return

        speed = (
            self.speed
            * min(self.origin_node.throughput, self.target_node.throughput)
            / 2
        )
        dist = self.target_node.position.compute_distance(self.position)
        if speed >= dist:
            self.position = self.target_node.position.clone()
            self.origin_node = self.target_node
            if self.mail is None:
                self.target_node.get_mail(self)
            return

        travel_coord = self._calculate_travel_coordinates(
            speed, self.position, self.target_node.position
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
        color = (255, 200, 200)
        size = 14
        pygame.draw.circle(screen, color, tuple(self.position), size)
        self._render_mail_target(screen)
        self._render_connected_node_numbers(screen)

    def _render_connected_node_numbers(self, screen: pygame.surface.Surface):
        global font
        if font is None:
            return
        # color = (50, 50, 50)
        color = (255, 255, 255)
        for i, node in enumerate(self.connected_nodes, 1):
            travel_coord = self._calculate_travel_coordinates(
                self.speed * 3.5, self.position, node.position
            )
            surf = font.render(str(i), True, color)
            screen.blit(surf, tuple(travel_coord + self.position))

    def _render_mail_target(self, screen: pygame.surface.Surface):
        if self.mail is None:
            return
        node_size = max(self.mail.target_node.size, 7)
        color = MAIL_TARGET_COLOR.colour
        pygame.draw.circle(
            screen, color, tuple(self.mail.target_node.position), node_size, width=0
        )
        if not self.target_node_particle_system.expired:
            self.target_node_particle_system.render(screen)

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

    def remove(self):
        if self in self.start.connections:
            self.start.connections.remove(self)
        if self in self.end.connections:
            self.end.connections.remove(self)

    def render(self, screen: pygame.surface.Surface):
        # color = (200, 200, 200)
        color = (36, 36, 36)
        for diff in range(self.width):
            pygame.draw.aaline(
                screen,
                color,
                tuple(self.start.position + Coordinate(x=diff)),
                tuple(self.end.position + Coordinate(x=diff)),
            )

    @property
    def speed(self) -> float:
        return min(self.start.throughput, self.end.throughput)

    @property
    def width(self) -> int:
        return int(self.speed)


@dataclass
class Score:
    game: Game
    combo_factor_decrease_per_tick: float  # negative
    combo_factor_decrease_delta_per_tick: float  # negative
    combo_factor: float = 1
    points: int = 0

    def update(self, game: Game):
        self.combo_factor = saturate(
            self.combo_factor + self.combo_factor_decrease_per_tick,
            min_=MIN_COMBO,
            max_=MAX_COMBO,
        )
        self.combo_factor_decrease_per_tick += self.combo_factor_decrease_delta_per_tick

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
        combo = f"{self.combo_factor:.2f}" if self.combo_factor != 0 else 0
        text = f"Points: {self.points}     Combo: {combo}"
        surf = gui_font.render(text, True, (255, 255, 255))
        screen.blit(surf, (0, 0))


def game_over_by_zero_combo(game: Game) -> bool:
    return game.score.combo_factor <= 0


def endless_game_win_strategy(game: Game) -> bool:
    return False


@dataclass
class Game:
    nodes: Iterable[Node]
    connections: Iterable[Connection]
    player: Player
    mail_spawn_factor: float
    mail_size_decay: float
    seed: int
    level: Level
    game_over_strategy: Callable[[Game], bool]
    game_win_strategy: Callable[[Game], bool] = endless_game_win_strategy
    score: Score = None  # type: ignore

    @property
    def entities(self) -> Iterable[Entity]:
        yield from self.connections
        yield from self.nodes
        yield self.player

    @property
    def over(self) -> bool:
        return self.game_over_strategy(self)

    @property
    def won(self) -> bool:
        return self.game_win_strategy(self)

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
    mail_spawn_factor: float
    initial_mail: int
    combo_factor: float
    combo_factor_decrease_per_tick: float
    combo_factor_decrease_delta_per_tick: float
    level: Level
    mail_size_decay: float


def init_game(params: Params, seed: Optional[int] = None) -> Game:
    seed = seed or random.randint(0, 100_000_000)
    random.seed(seed)

    nodes = spawn_nodes(params)
    connections = spawn_connections(nodes, params)
    conn_dict = defaultdict(set)
    for connection in connections:
        conn_dict[connection.start].add(connection)
        conn_dict[connection.end].add(connection)

    for node in nodes:
        node.connections = list(conn_dict.get(node, set()))

    nodes = [x for x in nodes if x.connections]
    removed_connections = set()
    for node in nodes:
        for connection in node.connections[MAX_CONNECTIONS:]:
            connection.remove()
            removed_connections.add(connection)
        node.connections = node.connections[:MAX_CONNECTIONS]
    nodes = [x for x in nodes if x.connections]
    connections = [x for x in connections if x not in removed_connections]

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
        seed=seed,
        level=params.level,
        game_over_strategy=game_over_by_zero_combo,
        mail_size_decay=params.mail_size_decay,
    )
    game.score = Score(
        game,
        combo_factor=params.combo_factor,
        combo_factor_decrease_per_tick=params.combo_factor_decrease_per_tick,
        combo_factor_decrease_delta_per_tick=params.combo_factor_decrease_delta_per_tick,
    )

    for node in random.choices(nodes, k=min(len(nodes), params.initial_mail)):
        node.spawn_mail(game)
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
    screen_offset = 50

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


class Level(enum.Enum):
    ONE = 0


@dataclass
class SaveData:
    highscore: int = 0
    cleared_levels: List[Level] = field(default_factory=list)

    def save(self):
        try:
            with open(SAVE_FILEPATH, "wb") as file:
                pickle.dump(self.__dict__, file)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Could not save data due to exception %s", str(exc))


def load_save_data() -> SaveData:
    try:
        with open(SAVE_FILEPATH, "rb") as file:
            dict_: Dict[str, Any] = pickle.load(file)
        save_data = SaveData()
        save_data.__dict__ = dict_
        return save_data
    except Exception:  # pylint: disable=broad-except
        return SaveData()


def save_game_data(game: Game):
    save_data = load_save_data()
    save_data.highscore = max(save_data.highscore, game.score.points)
    if game.won:
        save_data.cleared_levels.append(game.level)
    save_data.save()


def main():
    pygame.init()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # TODO: expose as ini params
    font_size = 25
    gui_font_size = 40
    full_screen = True
    seed = None

    flags = pygame.FULLSCREEN if full_screen else 0
    screen = pygame.display.set_mode(tuple(SCREEN_SIZE), flags=flags)
    board = pygame.Surface(tuple(BOARD_SIZE))
    clock = pygame.time.Clock()

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
        throughput=Stat(4, 1, 1),
        connections_per_node=Stat(1, 1, 1),
        node_offset=Stat(0, 5),
        max_connection_length=350,
        player_speed=10,
        mail_spawn_factor=0.005,
        initial_mail=5,
        combo_factor=1,
        combo_factor_decrease_per_tick=-0.00025,
        combo_factor_decrease_delta_per_tick=-0.000001,
        level=Level.ONE,
        mail_size_decay=0.003,  # per tick
    )
    params = level_one_params

    print(load_save_data())

    game = init_game(params, seed)
    in_game = True
    terminated = False
    saved = False
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and in_game:
                if event.key == pygame.K_ESCAPE:
                    terminated = True
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
                    if event.key == pygame.K_6:
                        game.player.set_target_connection(5)
                    if event.key == pygame.K_7:
                        game.player.set_target_connection(6)
                    if event.key == pygame.K_8:
                        game.player.set_target_connection(7)
                    if event.key == pygame.K_9:
                        game.player.set_target_connection(8)
                if game.over and event.key == pygame.K_RETURN:  # restart
                    game = init_game(params)
                    saved = False

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
            if not saved:
                save_game_data(game)
                saved = True

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
