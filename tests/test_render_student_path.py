import torch
import pytest

from inference.compare_teacher_student import predict_pair
from inference.render_student import predict_student_action


class _Normalizer:
    def unnormalize(self, value):
        return value


class _ImageEncoder:
    def __init__(self):
        self.calls = 0

    def encode_image_context(self, images, *, input_grad):
        self.calls += 1
        assert input_grad is False
        # Keep a context-dependent scalar so the test catches accidental use
        # of the original observation or the teacher force path.
        return torch.cat([images["side"].mean(dim=(1, 2, 3, 4), keepdim=False).view(-1, 1)], dim=1)


class _Policy:
    action_dim = 7
    horizon = 3
    n_obs_steps = 2
    n_action_steps = 2
    action_start_index = 1
    image_keys = ("side", "wrist")
    relative_pose_actions = False
    action_reference_key = "action_reference"

    def __init__(self, action_dim=7):
        self.action_dim = action_dim
        self.obs_encoder = _ImageEncoder()
        self.normalizer = {"action": _Normalizer()}
        self.predict_action_calls = 0
        self.force_fusion_calls = 0

    def conditional_sample(self, shape, context, generator=None):
        # Diffusion noise is the only random source in this fake policy.
        value = torch.randn(shape, generator=generator)
        value[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return value

    def predict_action(self, obs, generator=None):
        self.predict_action_calls += 1
        # Teacher's full path is deliberately separate from the student path.
        self.force_fusion_calls += 1
        context = torch.zeros((obs["side"].shape[0], 1))
        value = self.conditional_sample((obs["side"].shape[0], self.horizon, self.action_dim), context, generator)
        return {"action": value[:, self.action_start_index:self.action_start_index + self.n_action_steps]}


class _Preprocessor:
    n_obs_steps = 2

    def __init__(self):
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return obs["future_wrench"], None, None


class _RenderNet:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, wrench, *, use_attention_mask=None):
        self.calls += 1
        assert use_attention_mask is True
        return {key: value + (0.1 if key == "side" else 0.0) for key, value in images.items()}


def _obs():
    return {
        "side": torch.zeros((1, 2, 3, 4, 4)),
        "wrist": torch.zeros((1, 2, 3, 4, 4)),
        "future_wrench": torch.ones((1, 2, 8, 6)),
        "wrench": torch.ones((1, 2, 8, 6)),
    }


def test_student_path_encodes_rendered_rgb_without_teacher_force_fusion():
    policy, render_net, preprocessor, obs = _Policy(), _RenderNet(), _Preprocessor(), _obs()
    result = predict_student_action(policy, render_net, preprocessor, obs,
                                    generator=torch.Generator().manual_seed(4),
                                    use_attention_mask=True)
    assert result["action"].shape == (1, 2, 7)
    assert render_net.calls == 1 and preprocessor.calls == 1
    assert policy.obs_encoder.calls == 1
    # Student action construction never calls policy.predict_action / force fusion.
    assert policy.predict_action_calls == 0 and policy.force_fusion_calls == 0
    assert torch.allclose(result["action"][..., 3:].norm(dim=-1), torch.ones((1, 2)))


@pytest.mark.parametrize("action_dim", [7, 8])
def test_pair_restores_diffusion_generator_and_returns_postprocessed_actions(action_dim):
    policy, render_net, preprocessor, obs = _Policy(action_dim), _RenderNet(), _Preprocessor(), _obs()
    generator = torch.Generator().manual_seed(9)
    teacher, student = predict_pair(policy, render_net, preprocessor, obs, generator,
                                     amp_enabled=False, use_attention_mask=True)
    assert teacher.shape == student.shape == (2, action_dim)
    # The fake paths use identical diffusion noise; only the image context is
    # different, but this fake denoiser intentionally ignores context.
    assert (teacher == student).all()
    assert policy.predict_action_calls == 1
    assert policy.force_fusion_calls == 1
    assert policy.obs_encoder.calls == 1
