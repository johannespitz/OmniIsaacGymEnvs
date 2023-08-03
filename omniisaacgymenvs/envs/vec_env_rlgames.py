# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np

from datetime import datetime

import wandb

# VecEnv Wrapper for RL training
class VecEnvRLGames(VecEnvBase):

    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

        from omni.isaac.core.simulation_context import SimulationContext
        self.sim = SimulationContext._instance

        # https://github.com/NVIDIA-Omniverse/Orbit/blob/f5b24bba8218444a04c0ac7b47ac697ada2d7580/source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/isaac_env.py
        # check if flatcache is enabled
        # this is needed to flush the flatcache data into Hydra manually when calling `env.render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        if self._render is False and self.sim.get_physics_context().use_flatcache:
            from omni.physxflatcache import get_physx_flatcache_interface
            # acquire flatcache interface
            self._flatcache_iface = get_physx_flatcache_interface()

        self.num_cameras = self._task.num_cameras
        self.frames_list = [ [] for _ in range(self.num_cameras) ]

    def step(self, actions):
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        self._task.pre_physics_step(actions)
        
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        if self._render is False:
            if self.sim.get_physics_context().use_flatcache:
                self._flatcache_iface.update(0.0, 0.0)
            self.sim.render()

        curr_frames = self._task.render()
        for idx in range(self.num_cameras):
            if isinstance(curr_frames[idx], np.ndarray):
                if curr_frames[idx].shape[0] > 0:
                    self.frames_list[idx].append(curr_frames[idx])
                else:
                    print(f"{idx} is empty: {curr_frames[idx]}")
            else:
                print(f"{idx} is not an np.ndarray: {curr_frames[idx]}")
            if len(self.frames_list[idx]) >= 10:
                print("[saving video]")
                wandb.log(
                    {
                        f"camera_{idx}": wandb.Video(
                            np.array(self.frames_list[idx]).transpose(0, 3, 1, 2),
                        )
                    }
                )
                self.frames_list[idx] = []

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()
        
        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        # TODO: Figure out why
        # Skips first few steps. Necessary when using domain randomization
        # Probably need to wait for the replicator to initialize?
        self._world.step(render=self._render)
        if self._render is False:
            if self.sim.get_physics_context().use_flatcache:
                self._flatcache_iface.update(0.0, 0.0)
            self.sim.render()
        while not self._world.is_playing():
            self._world.step(render=self._render)
            if self._render is False:
                if self.sim.get_physics_context().use_flatcache:
                    self._flatcache_iface.update(0.0, 0.0)
                self.sim.render()

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.rl_device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict
