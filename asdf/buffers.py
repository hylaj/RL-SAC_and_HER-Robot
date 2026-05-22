from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )




class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        
        # TODO: fill this in
        # You can put additional attributes here if needed.
        # Also: There is a number of methods in the base class that could be useful to override.

        self.episode_indices = []

    def start_episode(self):
        self.episode_indices = []


    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_observation: dict[str, torch.Tensor],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        # TODO: fill this in
        # Just a suggestion: it may make sense to modify this method
        
        # Store the transition
        
        idx = self._ptr
        super().store(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.episode_indices.append(idx) # lista indeksów przejść aktualnego epizodu

        # TODO: fill this in
        # Or maybe here?


    def end_episode(self):
        # w tym nowym algorytmie to każda akcja z epizodu dostaje nowy goal, a w zasadzie 4 nowe goale (n_smapled_goal)



        # w zależności czy strategia to future czy final, wybieramy indeksy odpowiednich kandydatów do losowania. Zabezpieczenie przed brakiem kandydatów. 

        episode_indices = self.episode_indices
        n = len(episode_indices)

        if n == 0:
            return
        
        for i, idx in enumerate(episode_indices):
            if self.selection_strategy == "final":
                candidates = episode_indices[-1:]  
            elif self.selection_strategy == "future":
                candidates = episode_indices[i+1:]
            else:
                raise ValueError(f"Unknown goal selection strategy: {self.selection_strategy}")
            
            if not candidates:
                continue

        # Samplujemy n_sampled_goal razy (lub mniej) - czyli losujemy spośród kandydatów

            n_goals = min(self.n_sampled_goal, len(candidates))
            sampled_indices = np.random.choice(candidates, size=n_goals, replace=False)

            # pobieramy obserwacje, next_obs, akcje i wszystkie inne dane dla tych indeksów. 

            obs = {k: self.observations[k][idx].cpu().numpy() for k in self.observations}

            next_obs = {k: self.next_observations[k][idx].cpu().numpy() for k in self.next_observations}

            action = self.actions[idx].cpu().numpy()
            terminated = self.terminations[idx].item()  # czy castowac na bool?
            truncated = self.truncations[idx].item()
            info = self.infos[idx]        


            # Dla każdego nowego goal tworzymy jedno nowe przejście: Pobieramy osiągnięty goal z obserwacji z bufora pod tym indeksem goalu. 

            for sampled_idx in sampled_indices:
                new_goal = self.observations["achieved_goal"][sampled_idx].cpu().numpy()

                # Kopiujemy te obserwacje i next obs  i zastępujemy w nich desired goal tym achieved goal.
                new_obs = {k: v.copy() for k, v in obs.items()}
                new_next_obs = {k: v.copy() for k, v in next_obs.items()}

                new_obs["desired_goal"] = new_goal
                new_next_obs["desired_goal"] = new_goal

                # liczymy nową reward (specjalna funkcja env.compute_reward)

                new_reward = self.env.unwrapped.compute_reward(new_next_obs["achieved_goal"], new_goal, info)

                # sprawszmy czy jest termindated (jest jesli nagorda nowa jest rowna zero) i truncated nie jest (czylu false), bo w her nie ma limitu czasu

                new_terminated = (new_reward == 0.0)
                new_truncated = False

                # zapisujemy za pomoca funkcji store do buffera

                DictReplayBuffer.store(
                    self,
                    observation=new_obs,
                    action=action,
                    reward=new_reward,
                    next_observation=new_next_obs,
                    terminated=new_terminated,
                    truncated=new_truncated,
                    info=info,
                )








