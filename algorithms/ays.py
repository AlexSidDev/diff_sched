import numpy as  np
import math
import os
import torch
import torch.distributed as dist
import time
from metrics import METRICS


class StepsOptimizer:
    def __init__(self,
                 denoiser,
                 scheduler,
                 prompts: list,
                 metric: str,
                 teacher_paths: str,
                 traj_paths: str = None,
                 device: str = 'cuda',
                 num_candidates: int = 10,
                 save_dir='./mu_schedule'
                 ):
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.traj_paths = traj_paths
        self.prompts = prompts
        self.use_trajs = traj_paths is not None
        self.metric = METRICS[metric]
        self.teacher_paths = teacher_paths
        self.device = device
        self.num_candidates = num_candidates
        self.save_dir = save_dir

        self.sample_inds = list(range(len(self.traj_paths)))

    def load_teacher_traj(self, traj_ind: int):
        traj = torch.load(self.teacher_paths[traj_ind], map_location=self.device)
        x_end = traj[-1]
        x_start = traj[0]
        return x_end, x_start

    def get_student_traj(self, traj_ind: int, step_ind: int):
        traj = torch.load(self.traj_paths[traj_ind], map_location=self.device)
        x_cur = traj[step_ind]
        return x_cur

    def sigma(self, step_ind):
        return self.scheduler.sigmas[step_ind]

    def uniform_neighbourhood(self, t: float, tmin: float, tmax: float):
        step = (tmax - tmin) / (self.num_candidates - 1)
        neighbourhood = torch.arrange(tmin + step, tmax - step, step)
        return torch.cat([neighbourhood, torch.tensor([t])])

    def calculate_metric(self, t, step_ind, schedule: list, num_samples):
        metrics = []
        samples = np.random.choice(self.sample_inds, size=(num_samples,), replace=False).tolist()
        reversed_step_ind = len(schedule) - (step_ind + 1) if not self.use_trajs else step_ind
        for i in range(num_samples):

            x_end, x_start = self.load_teacher_traj(samples[i])
            prompt = self.prompts[samples[i]]

            schedule[reversed_step_ind] = t
            self.set_schedule(schedule)

            if self.use_trajs:
                x_cur = self.get_student_traj(samples[i], step_ind - 1)
                x_cur = self.denoiser(x_cur, schedule[step_ind - 1], prompt)
            else:
                x_cur = x_end + torch.randn_like(x_end) * self.sigma(reversed_step_ind)

            denoised = self.denoiser(x_cur, schedule[reversed_step_ind], prompt,
                                      return_orig=self.metric == 'cosine')

            metric = self.metric(x_end, x_cur if self.metric == 'cosine' else x_start, denoised)
            metrics.append(metric)

        return sum(metrics) / len(metrics)

    def update_trajs(self, step_ind: int, schedule):
        for i, traj_path in enumerate(self.traj_paths):
            traj = torch.load(traj_path, map_location=self.device)
            new_xt = self.denoiser(traj[step_ind - 1], schedule[step_ind - 1], step_ind, self.prompts[i])
            traj[step_ind] = new_xt
            torch.save(traj, traj_path)

    def set_schedule(self, schedule: list):
        schedule = schedule[::-1] if self.use_trajs else schedule
        self.scheduler.set_timesteps(timesteps=schedule)
        return schedule

    @torch.no_grad()
    def optimization(self, schedule: list, num_samples: int):
        optimized_schedule = (schedule[::-1] if self.use_trajs else schedule).copy()
        for i in range(1, len(schedule) - 1):
            print(f"Processing {i}-th step")
            neighbours = self.uniform_neighbourhood(optimized_schedule[i], optimized_schedule[i + 1],
                                                    optimized_schedule[i - 1]).tolist()
            metrics = []
            for j in range(self.num_candidates):
                metric = self.calculate_metric(neighbours[j], i, optimized_schedule, num_samples)
                print(f"Metric estimated for {j}-th candidate")
                metrics.append(metric)

            min_ind = np.argmin(metrics)

            if min_ind != self.num_candidates:
                optimized_schedule[i] = neighbours[min_ind]
                print(f'Changed {i}-th step')

            schedule = self.set_schedule(optimized_schedule)

            if self.use_trajs:
                self.update_trajs(i, optimized_schedule)

            with open(os.path.join(self.save_dir, 'schedule.txt'), 'r') as fout:
                print(schedule, file=fout)

        return schedule











