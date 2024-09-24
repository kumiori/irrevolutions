import logging
import random

import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Implementing a custom time-stepping iterator, CustomSimpleIterator, to iterate over a list of loads parametrising the evolution of an incremental system, providing the flexibility to pause time when needed.
# The iterator maintains an index to track the current load, along with a boolean flag to indicate whether time should be paused.
# If the flag is set, the iterator returns the current load without incrementing the index; otherwise, it increments the index and returns the next load.
# This design allows for a simple implementation of an energetic variational statement for time-dependent processes that require performing variations at fixed load. Such computations involve equilibrium, bifurcation, and stability checks, offering a clear and efficient mechanism for following the evolution of states for systems which are time-parametrised.


class CustomSimpleIterator:
    def __init__(self, loads):
        self.i = 0
        self.stop_time = False
        self.loads = loads

    def __iter__(self):
        return self

    def __next__(self):
        logger.info(f"\n\nCalled next, can time be stopped? {self.stop_time}")

        if self.stop_time:
            self.stop_time = False
            index = self.i
        else:
            # If pause_time_flag is False, check if there are more items to return
            if self.i < len(self.loads):
                # If there are more items, increment the index
                self.i += 1
                index = self.i
            else:
                raise StopIteration

        return index

    def pause_time(self):
        self.stop_time = True
        logger.info(f"Called pause, stop_time is {self.stop_time}")


# Example functions
def update_loads(t):
    return t


class EquilibriumSolver:
    def solve(self):
        return (random.uniform(-1, 1), random.uniform(0, 1))


class StabilitySolver:
    def solve(self, y_t, t):
        if t == 0:
            return True
        else:
            return random.choice([True, False])


def perturb_state(y_t):
    return y_t[0] + 0.1, y_t[1] + 0.1


# Example usage
loads = np.linspace(0, 10, 11)
iterator = CustomSimpleIterator(loads)
equilibrium = EquilibriumSolver()
stability = StabilitySolver()

while True:
    try:
        t = next(iterator)
        # next increments the self index
    except StopIteration:
        break

    # Perform your time step with t
    update_loads(t)

    # Log current load
    logger.info(f"Current load: {t:.2f}")

    y_t = equilibrium.solve()
    stable = stability.solve(y_t, t)

    logger.info(f"Equilibrium state at load {t:.2f}")
    # logger.info(f"Equilibrium state at load {t:.2f}: {y_t}")
    logger.info(f"Stability of state at load {t:.2f}: {stable}")

    if not stable:
        iterator.pause_time()
        y_t = perturb_state(y_t)
        logger.info(f"State perturbed at load {t:.2f}: {y_t}")
