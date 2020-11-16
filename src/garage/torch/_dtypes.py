"""Data structures used in garage.torch."""
from dataclasses import dataclass
import enum

import torch
from torch import nn


class ShuffledOptimizationNotSupported(ValueError):
    """Raised by recurrent policies if they're passed a shuffled batch."""


class PolicyMode(enum.IntEnum):
    """Defines what mode a PolicyInput is being used in.

    See :class:`PolicyInput` for detailed documentation.

    """
    # Policy is being used to run a rollout.
    # observations contains the last observations, and all_observations
    # contains partial episodes batched using lengths.
    ROLLOUT = 0
    # Policy is being used to do an optimization with timesteps from different
    # episodes. Recurrent policies must raise
    # ShuffledOptimizationNotSupported if they encounter this mode.
    SHUFFLED = 1
    # Policy is being used to do an optimization on complete episodes.
    FULL = 2


@dataclass
class PolicyInput:
    r"""The (differentiable) input to all pytorch policies.

    Args:
        mode (PolicyMode): The mode this batch is being used in. Determines the
            shape of observations.
        observations (torch.Tensor): A torch tensor containing flattened
            observations in a batch. Stateless policies should always operate
            on this input. Shape depends on the mode:
             * If `mode == ROLLOUT`, has shape :math:`(V, O)` (where V is the
                vectorization level).
             * If `mode == SHUFFLED`, has shape :math:`(B, O)` (where B is the
                mini-batch size).
             * If mode == FULL, has shape :math:`(N \bullet [T], O)` (where N
                is the number of episodes, and T is the episode lengths).
        lengths (torch.Tensor or None): Integer tensor containing the lengths
            of each episode. Only has a value if `mode == FULL`.

    """

    mode: PolicyMode
    observations: torch.Tensor
    lengths: torch.Tensor = None

    def __post_init__(self):
        """Check that lengths is consistent with the rest of the fields.

        Raises:
            ValueError: If lengths is not consistent with another field.

        """
        if self.mode == PolicyMode.FULL:
            if self.lengths is None:
                raise ValueError(
                    'lengths is None, but must be a torch.Tensor when '
                    'mode == PolicyMode.FULL')
            assert self.lengths is not None
            if self.lengths.dtype not in (torch.uint8, torch.int8, torch.int16,
                                          torch.int32, torch.int64):
                raise ValueError(
                    f'lengths has dtype {self.lengths.dtype}, but must have '
                    f'an integer dtype')
            total_size = sum(self.lengths)
            if self.observations.shape[0] != total_size:
                raise ValueError(
                    f'observations has batch size '
                    f'{self.observations.shape[0]}, but must have batch '
                    f'size {total_size} to match lengths')
            assert self.observations.shape[0] == total_size
        elif self.lengths is not None:
            raise ValueError(
                f'lengths has value {self.lengths}, but must be None '
                f'when mode == {self.mode}')

    def to_packed_sequence(self):
        """Turn full observations into a torch.nn.utils.rnn.PackedSequence.

        Raises:
            ShuffledOptimizationNotSupported: If called when `mode != FULL`

        Returns:
            torch.nn.utils.rnn.PackedSequence: The sequence of flattened
                observations.

        """
        if self.mode != PolicyMode.FULL:
            raise ShuffledOptimizationNotSupported(
                f'mode has value {self.mode} but must have mode '
                f'{PolicyMode.FULL} to use to_packed_sequence')
        sequence = []
        start = 0
        for length in self.lengths:
            stop = start + length
            sequence.append(self.observations[start:stop])
            start = stop
        pack_sequence = nn.utils.rnn.pack_sequence
        return pack_sequence(sequence, enforce_sorted=False)
