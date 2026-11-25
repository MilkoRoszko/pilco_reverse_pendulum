"""
Model obiektu: odwrócone wahadło przymocowane do stałego przegubu (napęd momentem w przegubie).

Wahadło to jednorodny pręt o masie m i długości l, obracający się wokół jednego końca.
Kąt theta mierzymy od pionu w GÓRĘ (theta = 0 -> wahadło stoi w pionie, theta = +-pi -> wisi w dół),
dodatni kierunek przeciwnie do ruchu wskazówek zegara.

Równanie ruchu (Euler-Lagrange):
    J * theta_dd = m*g*(l/2)*sin(theta) - b*theta_d + u,    J = m*l^2/3
czyli
    theta_dd = 3g/(2l) * sin(theta) - 3b/(m l^2) * theta_d + 3/(m l^2) * u

Przy b = 0 i domyślnych parametrach to dokładnie dynamika gym 'Pendulum-v0',
której używa przykład examples/pendulum_swing_up.py z repozytorium nrontsis/PILCO.
Dzięki temu klasę PendulumEnv można podstawić w miejsce środowiska gym bez zmian w kodzie PILCO.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class PendulumParams:
    m: float = 1.0          # masa pręta [kg]
    l: float = 1.0          # długość pręta [m]
    g: float = 10.0         # przyspieszenie ziemskie [m/s^2] (gym używa 10)
    b: float = 0.0          # tarcie wiskotyczne w przegubie [N m s/rad]
    u_max: float = 2.0      # ograniczenie momentu [N m]
    dt: float = 0.05        # okres próbkowania / sterowania [s]
    max_speed: float = 8.0  # ograniczenie prędkości kątowej [rad/s] (jak w gym)

    @property
    def J(self):
        return self.m * self.l ** 2 / 3.0


def angle_normalize(th):
    """Sprowadza kąt do przedziału [-pi, pi)."""
    return ((th + np.pi) % (2 * np.pi)) - np.pi


def dynamics(x, u, p: PendulumParams):
    """Prawa strona ODE: x = [theta, theta_d] -> [theta_d, theta_dd]."""
    th, thd = x
    thdd = (p.m * p.g * p.l / 2 * np.sin(th) - p.b * thd + u) / p.J
    return np.array([thd, thdd])


def rk4_step(x, u, p: PendulumParams, dt=None, substeps=10):
    """Całkowanie RK4 przy stałym sterowaniu w okresie dt (ZOH)."""
    dt = p.dt if dt is None else dt
    h = dt / substeps
    for _ in range(substeps):
        k1 = dynamics(x, u, p)
        k2 = dynamics(x + h / 2 * k1, u, p)
        k3 = dynamics(x + h / 2 * k2, u, p)
        k4 = dynamics(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def gym_step(x, u, p: PendulumParams):
    """Krok dokładnie jak w gym Pendulum-v0 (półjawny Euler) - do porównania."""
    th, thd = x
    thd = thd + (3 * p.g / (2 * p.l) * np.sin(th) + 3.0 / (p.m * p.l ** 2) * u) * p.dt
    thd = np.clip(thd, -p.max_speed, p.max_speed)
    th = th + thd * p.dt
    return np.array([th, thd])


def energy(x, p: PendulumParams):
    """Energia całkowita (zero potencjału w przegubie)."""
    th, thd = x
    return 0.5 * p.J * thd ** 2 + p.m * p.g * p.l / 2 * np.cos(th)


def linearize(p: PendulumParams, theta0=0.0):
    """Macierze A, B układu zlinearyzowanego w punkcie równowagi theta0 (0 = góra, pi = dół)."""
    A = np.array([[0.0, 1.0],
                  [p.m * p.g * p.l / 2 * np.cos(theta0) / p.J, -p.b / p.J]])
    B = np.array([[0.0], [1.0 / p.J]])
    return A, B


class PendulumEnv:
    """
    Środowisko w stylu gym, zgodne z tym, czego oczekuje examples/utils.py:rollout() z nrontsis/PILCO.
    Obserwacja: [cos(theta), sin(theta), theta_d] (jak w Pendulum-v0).
    """

    def __init__(self, params: PendulumParams = None, integrator="rk4", seed=None):
        self.p = params or PendulumParams()
        self.integrator = integrator
        self.rng = np.random.default_rng(seed)
        self.state = np.array([np.pi, 0.0])
        self.action_space_high = np.array([self.p.u_max])

    def _obs(self):
        th, thd = self.state
        return np.array([np.cos(th), np.sin(th), thd])

    def reset(self, state=None, noise=0.01):
        """Domyślnie start z wahadłem wiszącym w dół (jak w pendulum_swing_up.py)."""
        if state is None:
            state = np.array([-np.pi, 0.0]) + self.rng.uniform(0, noise, size=2) * np.array([np.pi, 1.0])
        self.state = np.asarray(state, dtype=float)
        return self._obs()

    def step(self, action):
        u = float(np.clip(np.ravel(action)[0], -self.p.u_max, self.p.u_max))
        th, thd = self.state
        cost = angle_normalize(th) ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2
        if self.integrator == "rk4":
            x = rk4_step(self.state, u, self.p)
            x[1] = np.clip(x[1], -self.p.max_speed, self.p.max_speed)
        else:
            x = gym_step(self.state, u, self.p)
        self.state = x
        return self._obs(), -cost, False, {}

    def simulate(self, x0, policy, T):
        """Symulacja T kroków: policy(x, t) -> u. Zwraca czasy, stany, sterowania."""
        self.state = np.asarray(x0, dtype=float)
        xs, us = [self.state.copy()], []
        for t in range(T):
            u = float(np.clip(policy(self.state, t), -self.p.u_max, self.p.u_max))
            self.step(u)
            xs.append(self.state.copy())
            us.append(u)
        return np.arange(T + 1) * self.p.dt, np.array(xs), np.array(us)
