# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=c-extension-no-member
# pylint: disable=import-error

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Type, Union

import pygame

from coordinate import Coordinate


@dataclass
class Colour:

    r: int
    g: int
    b: int
    alpha: int = 0
    max_: int = 255
    min_: int = 0

    def __iadd__(self, value: float):
        self.r = self.saturate(self.r + value)
        self.g = self.saturate(self.g + value)
        self.b = self.saturate(self.b + value)
        self.alpha = self.saturate(self.alpha + value)
        return self

    def __isub__(self, value: float):
        self.r = self.saturate(self.r - value)
        self.g = self.saturate(self.g - value)
        self.b = self.saturate(self.b - value)
        self.alpha = self.saturate(self.alpha - value)
        return self

    @property
    def colour(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.alpha)

    def saturate(self, value: float) -> int:
        return max(min(round(value), self.max_), self.min_)


def check_max_size_expired(particle: Particle) -> bool:
    return particle.size >= particle.max_size

def check_min_size_expired(particle: Particle) -> bool:
    return particle.size <= particle.min_size

@dataclass
class RectParticle:

    position: Coordinate
    colour: Colour
    spread: int = 0
    width: int = 0
    size: int = 25
    size_drift: int = -1
    colour_spread: int = 0
    alpha_drift: int = 0
    colour_drift: int = 0
    min_size: int = 0
    max_size: int = 255
    x_drift: int = 0
    y_drift: int = 0
    expiration_algorithm: Callable[[Particle], bool] = field(default=check_max_size_expired)

    def __post_init__(self):
        self.position.x += random.randint(-self.spread, self.spread)
        self.position.y += random.randint(-self.spread, self.spread)
        self.colour += random.randint(-self.colour_spread, self.colour_spread)
        self.expired = False

    def update(self):
        if self.expiration_algorithm(self):
            self.expired = True
            return

        self._update_size()
        self._update_colour()
        self._update_position()

    def render(self, screen: pygame.surface.Surface):
        if self.expired:
            return

        rect = pygame.Rect(*tuple(self.position), self.size, self.size)  # type: ignore
        pygame.draw.rect(screen, self.colour.colour, rect, self.width)

    def _update_size(self):
        self.size += self.size_drift

    def _update_colour(self):
        self.colour += self.colour_drift
        self.colour.alpha = self.colour.saturate(self.colour.alpha + self.alpha_drift)

    def _update_position(self):
        self.position.x += self.x_drift
        self.position.y += self.y_drift

    @property
    def expired(self):
        return self._expired or self.size == 0 or not any(self.colour.colour)

    @expired.setter
    def expired(self, value):
        self._expired = value


class CircleParticle(RectParticle):
    def render(self, screen: pygame.surface.Surface):
        if not self.expired:
            pygame.draw.circle(
                screen, self.colour.colour, (int(self.position.x), int(self.position.y)), self.size, self.width
            )


Particle = Union[RectParticle, CircleParticle]


@dataclass
class ParticleSystem:

    particle_type: Type[Particle]
    position: Coordinate
    spawn_rate: float
    colour: Colour
    colour_drift: int = 0
    lifetime: float = 1
    expired: bool = False

    def __post_init__(self):
        self.particles: List[Particle] = []
        self.start_time = time.time()
        self.spawn_time = time.time()
        self.kwargs = {}

    def add_kwargs(self, **kwargs):
        self.kwargs.update(kwargs)

    def render(self, screen: pygame.surface.Surface):
        if self.fully_expired:
            return

        for particle in self.particles:
            particle.render(screen)

    def clone(self, position: Coordinate) -> ParticleSystem:
        copied = copy.deepcopy(self)
        copied.position = position.clone()
        copied.__post_init__()
        copied.expired = False
        copied.kwargs = self.kwargs
        return copied

    def update(self):
        if self.fully_expired:
            return

        self.colour += self.colour_drift
        for particle in self.particles:
            particle.update()

        if time.time() - self.start_time > self.lifetime:
            self.expired = True

        if time.time() - self.spawn_time > self.spawn_rate and not self.expired:
            self.spawn_time = time.time()
            self.particles.append(self.create_new_particle())

    def create_new_particle(self) -> Particle:
        new_particle = self.particle_type(
            self.position.clone(), copy.deepcopy(self.colour), **self.kwargs
        )
        if new_particle.expired:
            self.expired = True
        return new_particle

    @property
    def fully_expired(self) -> bool:
        return self.expired and all(x.expired for x in self.particles)
