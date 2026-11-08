"""The RenderNet student action path used by distillation.

Wrench enters RenderNet only. Rendered RGB is encoded by the frozen visual
path and passed directly to DiT, bypassing the teacher's wrench fusion.
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def predict_student_action(policy, render_net, preprocessor, obs, *, generator=None,
                           use_attention_mask=None):
    """Return the teacher-compatible action dictionary for the RGB student.

    ``policy`` supplies the frozen DINO visual path, DiT, scheduler and action
    normalizer. Calling ``policy.predict_action`` here would incorrectly run
    its force encoder / cross-attention on top of the learned RGB rendering.
    The caller owns autocast and the diffusion generator, just as for FDP.
    """
    # Import after the caller has loaded its configured diffusion_policy repo.
    from diffusion_policy.common.pose_util import (
        normalize_quaternion_xyzw,
        relative_pose_to_absolute_pose,
    )

    if "past_action" in obs:
        raise NotImplementedError("past_action conditioning is not implemented")
    if int(policy.action_dim) not in (7, 8):
        raise ValueError("RenderNet FDP adapter requires xyz + xyzw with an optional gripper command")
    if policy.relative_pose_actions and int(policy.action_dim) != 7:
        raise ValueError("Relative pose actions require seven dimensions")
    if int(preprocessor.n_obs_steps) != int(policy.n_obs_steps):
        raise ValueError("RenderNet preprocessing and FDP observation lengths differ")
    images = {key: obs[key][:, :policy.n_obs_steps] for key in policy.image_keys}
    wrench, _, _ = preprocessor(obs)
    rendered = render_net(images, wrench, use_attention_mask=use_attention_mask)
    context = policy.obs_encoder.encode_image_context(rendered, input_grad=False)
    batch_size = context.shape[0]
    normalized_prediction = policy.conditional_sample(
        (batch_size, policy.horizon, policy.action_dim), context=context,
        generator=generator,
    )
    model_prediction = policy.normalizer["action"].unnormalize(normalized_prediction)
    model_prediction = torch.cat((model_prediction[..., :3],
                                  normalize_quaternion_xyzw(model_prediction[..., 3:7]),
                                  model_prediction[..., 7:]), dim=-1)
    action_prediction = model_prediction
    if policy.relative_pose_actions:
        key = policy.action_reference_key
        if key not in obs:
            raise KeyError(f"relative pose prediction requires obs[{key!r}]")
        reference = obs[key]
        if reference.ndim == 3:
            reference = reference[:, -1]
        if reference.ndim != 2 or reference.shape != (batch_size, 7):
            raise ValueError(f"obs[{key!r}] must have shape [B, 7] or [B, T, 7]")
        action_prediction = relative_pose_to_absolute_pose(model_prediction, reference)
    start = policy.action_start_index
    end = start + policy.n_action_steps
    selected = action_prediction[:, start:end]
    if selected.shape[1] != policy.n_action_steps:
        raise RuntimeError("student action chunk does not match the configured horizon")
    return {"action": selected, "action_pred": action_prediction,
            "model_action_pred": model_prediction}
