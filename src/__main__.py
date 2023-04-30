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
import json
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Protocol, Tuple, Union)

import pygame
from pygame import gfxdraw

from src.coordinate import Coordinate
from src.particle import (CircleParticle, Colour, ParticleSystem,
                          check_min_size_expired)

MIN_NODE_SPAWN_STEP = 100
BOARD_SCREEN_OFFSET = 40

PLAYER_COLOR = (255, 51, 153)
RETRO_GREEN = (51, 255, 51)
SCREEN_SIZE = Coordinate(800, 600)
BOARD_SIZE = Coordinate(600, 500)
GUI_HEIGHT = SCREEN_SIZE.y - BOARD_SIZE.y
MAX_CONNECTIONS = 5
MAX_THROUGHPUT = 25
MAX_COMBO = 2
MIN_COMBO = 0
MIN_MAIL_SIZE = 5
MAIL_OFFSET = Coordinate(10, -5)
MAIL_TARGET_COLOR = Colour(255, 153, 51, 255)
SAVE_FILEPATH = "mailgame.sav"
FPS = 50

font: Optional[pygame.font.Font] = None
gui_font: Optional[pygame.font.Font] = None


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

    def __post_init__(self):
        self.previous_decay: float = 0 # positive

    def update(self, game: Game):
        self.previous_decay = game.mail_size_decay * self.decay_factor
        self.size -= self.previous_decay

    def render(self, screen: pygame.surface.Surface):
        rect = pygame.Rect(
            *tuple(self.position), int(self.size), int(self.size)
        )  # type: ignore
        #color = (100, 100, 255)
        color = (51, 51, 255)
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

    @property
    def decay_factor(self) -> float:
        return 0.4 if isinstance(self.parent, Player) else 1

    @property
    def ticks_until_expiry(self) -> float:
        if self.expired:
            return 0
        return (self.size - MIN_MAIL_SIZE) / self.previous_decay if self.previous_decay != 0 else math.inf

@dataclass
class Node:
    position: Coordinate
    throughput: float
    mail: Optional[Mail] = None
    connections: List[Connection] = field(default_factory=list)

    def __post_init__(self):
        self.occupied: bool = False

    def __hash__(self):
        return hash(self.position)

    def get_mail(self, player: Player):
        if self.mail and not player.mail:
            player.take_mail(self)
            self.mail = None
        self.occupied = True

    def update(self, game: Game):
        spawn_chance = self.throughput / MAX_THROUGHPUT * game.mail_spawn_factor
        if self.mail:
            self.mail.update(game)
            if self.mail.expired:
                self.mail = None

        if not self.mail and not self.occupied and chance(spawn_chance):
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
        #color = (255, 255, 255)
        color = (255, 247, 251)
        # pygame.draw.rect(screen, color, rect, width=0)
        # pygame.draw.circle(screen, color, tuple(self.position), self.size)
        gfxdraw.aacircle(screen, int(self.position.x), int(self.position.y), self.size, color)
        gfxdraw.filled_circle(screen, int(self.position.x), int(self.position.y), self.size, color)
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
class Animation:
    values: Iterable[Any]
    tick: int

    def __post_init__(self):
        self._iterator: Optional[Iterator[Any]] = None
        self.start_tick: Optional[int] = None
        self.current_tick: int = 0
        self.ongoing: bool = False
        self.current_value: Optional[Any] = None

    def start(self):
        self.start_tick = None
        self.current_tick = 0
        self.ongoing = True
        self._iterator = None
        self.current_value = None

    def update(self, game: Game):
        if not self.ongoing:
            return

        if self.start_tick is None:
            self.start_tick = game.tick

        if self.current_tick % self.tick == 0:
            value = self._next()
            if value is not None:
                self.current_value = value

        if self.start_tick != game.tick:
            self.current_tick += 1

    def _next(self) -> Optional[Any]:
        if self._iterator is None:
            self._iterator = iter(self.values)

        try:
            return next(self._iterator)
        except StopIteration:
            self.ongoing = False
            return None


@dataclass
class Player:
    position: Coordinate
    target_node: Node
    speed: float
    mail: Optional[Mail] = None
    color = PLAYER_COLOR
    size = 10

    def __post_init__(self):
        self.target_node.occupied = True
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
        self.target_animation = Animation(
            [0, 1, 1, 2, 3, 3, 3, 2, 1, 0, -1, -1, 0], tick=2
        )
        self.path_particle_system = ParticleSystem(CircleParticle, position=Coordinate(), colour=Colour(*self.color), spawn_rate=0.05, expired=True)
        self.path_particle_system.add_kwargs(size_drift=-0.2, size=round(self.size * 0.7), expiration_algorithm=check_min_size_expired)
        self.path_particle_systems: List[ParticleSystem] = []


    def take_mail(self, node: Node):
        if not node.mail:
            return
        self.target_animation.start()
        self.mail = node.mail
        self.mail.parent = self
        self.target_node_particle_system = self.target_node_particle_system.clone(
            self.mail.target_node.position.clone()
        )
        self.target_node_particle_system.add_kwargs(size=self.mail.target_node.size)

    def update(self, game: Game):
        for particle_system in self.path_particle_systems:
            particle_system.position = self.position.clone()
            particle_system.update()

        self._move()
        if self.mail:
            self.mail.update(game)
            self.target_animation.update(game)
            if self.mail.expired:
                self.mail = None
            if not self.target_node_particle_system.expired:
                self.target_node_particle_system.update()

        if self.mail and self.mail.reached_target and not game.over:
            game.score.update_score(self.mail)
            self.mail = None
            if self.target_node.mail:
                self.target_node.get_mail(self)

    def _move(self):
        if self.reached_target:
            self._expire_path_particle_systems()
            return

        self.origin_node.occupied = False
        speed = (
            self.speed
            * min(self.origin_node.throughput, self.target_node.throughput)
            / 2
        )
        dist = self.target_node.position.compute_distance(self.position)
        if speed >= dist:
            self.position = self.target_node.position.clone()
            self.origin_node = self.target_node
            self.target_node.occupied = True
            if self.mail is None:
                self.target_node.get_mail(self)
            self._expire_path_particle_systems()
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

    def _expire_path_particle_systems(self):
        for particle_system in self.path_particle_systems:
            particle_system.expired = True

    def render(self, screen: pygame.surface.Surface):
        # pygame.draw.circle(screen, color, tuple(self.position), size)
        gfxdraw.aacircle(screen, int(self.position.x), int(self.position.y), self.size, self.color)
        gfxdraw.filled_circle(screen, int(self.position.x), int(self.position.y), self.size, self.color)
        self._render_mail_target(screen)
        self._render_connected_node_numbers(screen)
        self._render_particle_systems(screen)

    def _render_particle_systems(self, screen: pygame.surface.Surface):
        for particle_system in self.path_particle_systems:
            particle_system.render(screen)
        self.path_particle_systems = [x for x in self.path_particle_systems if not x.fully_expired]


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
        size_offset = self.target_animation.current_value or 0
        node_size = max(self.mail.target_node.size, 7) + size_offset
        color = MAIL_TARGET_COLOR.colour
        x, y = self.mail.target_node.position
        gfxdraw.aacircle(screen, int(x), int(y), node_size, color)
        gfxdraw.filled_circle(screen, int(x), int(y), node_size, color)
        if not self.target_node_particle_system.expired:
            self.target_node_particle_system.render(screen)

    def set_target_connection(self, index: int):
        if len(self.target_node.connections) <= index:
            return
        self.target_node = self.connected_nodes[index]

        new_particle_system = self.path_particle_system.clone(position=self.position.clone())
        self.path_particle_systems.append(new_particle_system)

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
    color = RETRO_GREEN

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
        #color = (36, 36, 36)
        for diff in range(self.width):
            pygame.draw.aaline(
                screen,
                self.color,
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

    def __post_init__(self):
        self.max_combo_animation = Animation(values=[0, 1, 1, 2, 2, 2, 2, 1, 1, 0, -1, -1, -2, -2, -2, -2, -1, -1] * 3 + [0], tick=3)
        self.expired_mail_animation = Animation(values=[1, 1.05, 1.1, 1.1, 1.05, 1, 0.9, 0.95, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0], tick=4)
        self.expired_mail_offset_animation = Animation(values=[Coordinate(0, 0), Coordinate(3, -10), Coordinate(6, -20), Coordinate(9, -20), Coordinate(12, 0), Coordinate(15, 35), Coordinate(20, 70), Coordinate(25, 105)], tick=5)
        self.mail: Optional[Mail] = None

    def update(self, game: Game):
        self.combo_factor = saturate(
            self.combo_factor + self.combo_factor_decrease_per_tick,
            min_=MIN_COMBO,
            max_=MAX_COMBO + 1, # give some leeway to show MAX COMBO text
        )
        self.combo_factor_decrease_per_tick += self.combo_factor_decrease_delta_per_tick
        self.max_combo_animation.update(game)
        self.expired_mail_animation.update(game)
        self.expired_mail_offset_animation.update(game)
        self.set_mail(game.player.mail)

    def set_mail(self, mail: Optional[Mail]):
        if mail is None and self.mail is not None and not self.mail.reached_target:
            self.expired_mail_animation.start()
            self.expired_mail_offset_animation.start()
        self.mail = mail

    def update_score(self, mail: Mail):
        points = mail.points
        self.points += round(points * min(self.combo_factor, MAX_COMBO))
        new_combo_factor = saturate(
            self.combo_factor + 1, min_=MIN_COMBO, max_=MAX_COMBO + 1
        )
        if new_combo_factor >= MAX_COMBO and self.combo_factor < MAX_COMBO:
            self._reach_max_combo()
        self.combo_factor = new_combo_factor

    def _reach_max_combo(self):
        self.max_combo_animation.start()

    def render(self, screen: pygame.surface.Surface):
        self._render_score(screen)
        self._render_remaining_time(screen)

    def _render_score(self, screen: pygame.surface.Surface):
        global gui_font
        if gui_font is None:
            return
        combo = f"{self.combo_factor:.2f}" if self.combo_factor != 0 else 0
        combo_text = f"Combo: {combo}" if self.combo_factor < MAX_COMBO else "MAX COMBO"
        text = f"Points: {self.points}  {combo_text}"
        surf = gui_font.render(text, True, RETRO_GREEN)
        *_, width, height = surf.get_rect()
        # x = int((SCREEN_SIZE.x - width) / 2)
        x = BOARD_SCREEN_OFFSET
        y = int((GUI_HEIGHT - height) / 2)
        y_offs = self.max_combo_animation.current_value or 0
        screen.blit(surf, (x, y + y_offs))

    def _render_remaining_time(self, screen: pygame.surface.Surface):
        global font
        if font is None:
            return
        if self.mail is None and not self.expired_mail_animation.ongoing:
            return
        ticks = self.mail.ticks_until_expiry if self.mail else 0
        time_text = f"{ticks / FPS:.1f}" if ticks != math.inf else "never"
        text = f"Mail expires in: {time_text}" if not self.expired_mail_animation.ongoing else "EXPIRED"
        surf = font.render(text, True, RETRO_GREEN)
        *_, width, height = surf.get_rect()
        x_offs, y_offs = (0, 0)
        if self.expired_mail_animation.ongoing:
            *_, scaled_width, scaled_height = surf.get_rect()
            factor = self.expired_mail_animation.current_value or 0
            surf = pygame.transform.scale(surf, (scaled_width * factor, scaled_height * factor))
            offs = self.expired_mail_offset_animation.current_value or Coordinate(45, 135)
            x_offs = (width - scaled_width) + offs.x
            y_offs =  offs.y #(height - scaled_height)
        x = SCREEN_SIZE.x - width - BOARD_SCREEN_OFFSET
        y = int((GUI_HEIGHT - height) / 2)
        screen.blit(surf, (x + x_offs, y + y_offs))


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

    def __post_init__(self):
        self.tick: int = 0

    def update(self):
        self.tick += 1
        self.score.update(self)

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
    logging.info("Seed: %s", seed)

    Connection.color = RETRO_GREEN
    Player.color = PLAYER_COLOR

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
        if node.occupied:
            continue
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
    grid = list(
        itertools.product(
            range(BOARD_SCREEN_OFFSET, int(BOARD_SIZE.x), MIN_NODE_SPAWN_STEP),
            range(BOARD_SCREEN_OFFSET, int(BOARD_SIZE.y), MIN_NODE_SPAWN_STEP),
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

def init_font(font_size: int) -> Optional[pygame.font.Font]:
    path = os.path.join(os.path.dirname(__file__), "media", "VT323-Regular.ttf")
    try:
        return pygame.font.Font(path, font_size)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init retro font due to exception %s", str(exc))
        logging.info("Falling back to default font...")

    try:
        return pygame.font.SysFont(name=pygame.font.get_default_font(), size=font_size)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init font due to exception %s", str(exc))

def init_gui_font(font_size: int) -> Optional[pygame.font.Font]:
    path = os.path.join(os.path.dirname(__file__), "media", "VT323-Regular.ttf")
    try:
        return pygame.font.Font(path, font_size)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init retro font due to exception %s", str(exc))
        logging.info("Falling back to default font...")

    try:
        return pygame.font.SysFont(
            name=pygame.font.get_default_font(), size=font_size
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not init gui font due to exception %s", str(exc))

@dataclass
class Config:
    font_size: int = 25
    gui_font_size: int = 40
    full_screen: bool = False
    seed: Optional[int] = None
    muted: bool = False
    log_enabled: bool = True

def read_config_json() -> Config:
    try:
        with open("config.json", "r", encoding="utf-8") as file:
            dict_ = json.load(file)
        config = Config()
        config.__dict__ = dict_
        return config
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Could not read config json due to exception %s", str(exc))
        return Config()


def main():
    pygame.init()

    config = read_config_json()
    log_enabled = config.log_enabled
    font_size = config.font_size
    gui_font_size = config.gui_font_size
    full_screen = config.full_screen
    seed = config.seed
    muted = config.muted

    handlers = [logging.NullHandler()] if not log_enabled else [logging.FileHandler("mailgame.log")]
    logging.basicConfig(handlers=handlers, level=logging.INFO, format="%(asctime)s %(message)s")



    flags = pygame.FULLSCREEN if full_screen else 0
    screen = pygame.display.set_mode(tuple(SCREEN_SIZE), flags=flags)
    board = pygame.Surface(tuple(BOARD_SIZE))
    clock = pygame.time.Clock()

    global gui_font, font
    font = init_font(font_size)
    gui_font = init_gui_font(gui_font_size)
    
    level_one_params = Params(
        amount=Stat(30, 3, 25),
        throughput=Stat(4, 0.8, 1.5),
        connections_per_node=Stat(2, 1, 1),
        node_offset=Stat(0, 10),
        max_connection_length=350,
        player_speed=10,
        mail_spawn_factor=0.005,
        initial_mail=5,
        combo_factor=1,
        combo_factor_decrease_per_tick=-0.00025,
        combo_factor_decrease_delta_per_tick=-0.0000001,
        level=Level.ONE,
        #mail_size_decay=0.003,  # per tick
        mail_size_decay=0.03,  # per tick
    )
    params = level_one_params

    logging.info("Save data: %s", load_save_data())

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
                if event.key == pygame.K_f:
                    full_screen = not full_screen
                    flags = pygame.FULLSCREEN if full_screen else 0
                    screen = pygame.display.set_mode(tuple(SCREEN_SIZE), flags=flags)
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
                if event.key == pygame.K_r or event.key == pygame.K_RETURN:  # restart
                    game = init_game(params)
                    saved = False
                if event.key == pygame.K_m:
                    muted = not muted

        for entity in itertools.chain(game.entities, game.gui_entities):
            entity.update(game)
        game.update()

        if game.over:
            Connection.color = (17, 150, 17, 90) # type: ignore
            Player.color = (178, 71, 125) # type: ignore
        screen.fill((12, 12, 12))
        #board.fill((36, 36, 36))
        board.fill((12, 12, 12))
        for entity in game.entities:
            entity.render(board)
        for entity in game.gui_entities:
            entity.render(screen)
        if game.over:
            board.fill((0, 0, 0, 90), special_flags=pygame.BLEND_RGBA_SUB)
        screen.blit(board, (100, 100))
        if game.over:
            if not saved:
                save_game_data(game)
                saved = True

            surf = gui_font.render(
                "Game Over! Press R to restart.", True, RETRO_GREEN
            )
            *_, width, height = surf.get_rect()
            screen.blit(
                surf, ((SCREEN_SIZE.x - width) / 2, (SCREEN_SIZE.y - height) / 2)
            )
        pygame.display.flip()
        clock.tick(FPS)

    pygame.display.quit()


if __name__ == "__main__":
    main()
