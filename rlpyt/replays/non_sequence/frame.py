
import numpy as np

from rlpyt.utils.collections import is_namedarraytuple
from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.frame import FrameBufferMixin
from rlpyt.replays.non_sequence.uniform import UniformReplay
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplay
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class NStepFrameBuffer(FrameBufferMixin, NStepReturnBuffer):
    """Special method for re-assembling observations from frames."""

    def extract_observation(self, T_idxs, B_idxs):
        """Assembles multi-frame observations from frame-wise buffer.  Frames
        are ordered OLDEST to NEWEST along C dim: [B,C,H,W].  Where
        ``done=True`` is found, the history is not full due to recent
        environment reset, so these frames are zero-ed.
        """
        # Begin/end frames duplicated in samples_frames so no wrapping here.
        # return np.stack([self.samples_frames[t:t + self.n_frames, b]
        #     for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        if is_namedarraytuple(self.samples_frames):
            stacked = {}
            for k, arr in self.samples_frames.items():
                stacked[k] = np.stack([arr[t:t + self.n_frames, b]
                    for t, b in zip(T_idxs, B_idxs)], axis=0)
            observation = type(self.samples_frames)(**stacked)
        else:
            observation = np.stack([self.samples_frames[t:t + self.n_frames, b]
                for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.samples.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation

    def load_state_dict(self, state_dict):
        assert self.T == state_dict["T"]
        assert self.B == state_dict["B"]
        assert self.size == state_dict["size"]
        assert self.discount == state_dict["discount"]
        assert self.n_step_return == state_dict["n_step_return"]
        self.t = t
        assert self.samples.shape == state_dict["samples"].shape
        self.samples = state_dict["samples"]
        self.samples_return_ = state_dict["samples_return_"]
        self.samples_done_n = state_dict["samples_done_n"]
        self._buffer_full = state_dict["_buffer_full"]
        self.off_backward = state_dict["off_backward"]

        self.n_frames = state_dict["n_frames"]
        assert self.samples_frames.shape == state_dict["samples_frames"].shape
        self.samples_frames = state_dict["samples_frames"]
        assert (
            self.samples_new_frames.shape
            == state_dict["samples_new_frames"].shape
        )
        self.samples_new_frames = state_dict["samples_new_frames"]
        self.off_forward = state_dict["off_forward"]

    def state_dict(self):
        return dict(
            # NStepReturnBuffer
            T=self.T,
            B=self.B,
            size=self.size,
            discount=self.discount,
            n_step_return=self.n_step_return,
            t=self.t,
            samples=self.samples,
            samples_return_=self.samples_return_,
            samples_done_n=self.samples_done_n,
            _buffer_full=self._buffer_full,
            off_backward=self.off_backward,
            # FrameBufferMixin
            n_frames=self.n_frames,
            samples_frames=self.samples_frames,
            samples_new_frames=self.samples_new_frames,
            off_forward=self.off_forward,
        )


class UniformReplayFrameBuffer(UniformReplay, NStepFrameBuffer):
    pass


class PrioritizedReplayFrameBuffer(PrioritizedReplay, NStepFrameBuffer):
    pass


class AsyncUniformReplayFrameBuffer(AsyncReplayBufferMixin,
        UniformReplayFrameBuffer):
    pass


class AsyncPrioritizedReplayFrameBuffer(AsyncReplayBufferMixin,
        PrioritizedReplayFrameBuffer):
    pass
