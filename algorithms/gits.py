import csv
import copy
import os.path
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# dp_list is a list of indices to be selected from the longer teacher time schedule

def get_dp_list(denoiser, prompts: list, guidance_scale: float, device, **solver_kwargs):
    model_kwargs = dict()

    try:
        is_main_process = dist.get_rank() == 0
        world_size = dist.get_world_size()
    except:
        is_main_process = True
        world_size = 1

    print0 = print if is_main_process else lambda *args, **kwargs: None

    num_warmup = solver_kwargs['num_warmup']
    max_batch_size = solver_kwargs['max_batch_size']
    num_steps = solver_kwargs['num_steps']
    num_steps_tea = solver_kwargs['num_steps_tea']
    metric = solver_kwargs['metric']
    coeff = solver_kwargs['coeff']
    traj_cache_dir = solver_kwargs['cache_dir']

    model_kwargs['output_type'] = 'trajs'
    model_kwargs['num_inference_steps'] = num_steps_tea
    model_kwargs['guidance_scale'] = guidance_scale

    denoiser.scheduler.set_timesteps(num_steps_tea + 1)

    sigmas = denoiser.scheduler.sigmas[:-1]
    denoiser.scheduler.set_timesteps(sigmas=sigmas)

    num_accumulation_rounds = num_warmup // (max_batch_size + 1) + 1
    batch_gpu = max_batch_size // world_size
    print0(f'Accumulate {num_accumulation_rounds} rounds to collect {num_warmup} trajectories...')
    cost_mat = torch.zeros((num_steps_tea + 1, num_steps_tea + 1), device=device)

    prompts_inds = np.random.choice(list(range(len(prompts))), num_accumulation_rounds * max_batch_size, replace=False)
    os.makedirs(traj_cache_dir, exist_ok=True)
    is_cache_empty = len(os.listdir(traj_cache_dir)) == 0
    order = 4

    used_prompts = []

    for r in range(num_accumulation_rounds):
        with torch.no_grad():

            ind_len = len(str(r))
            add_zeros = '0' * (order - ind_len)
            if is_cache_empty:
                print0(f'Round {r + 1}/{num_accumulation_rounds} | Generating the teacher trajectory...')
                round_prompts = [prompts[prompts_ind] for prompts_ind in prompts_inds[r * batch_gpu: (r + 1) * batch_gpu]]
                teacher_traj, eps_traj = denoiser(round_prompts, **model_kwargs)

                used_prompts.extend(round_prompts)
                torch.save(teacher_traj, os.path.join(traj_cache_dir, f'traj_{add_zeros}{r}.pt'))
                torch.save(eps_traj, os.path.join(traj_cache_dir, f'eps_{add_zeros}{r}.pt'))
            else:
                teacher_traj = torch.load(os.path.join(traj_cache_dir, f'traj_{r}.pt')).to(device)
                eps_traj = torch.load(os.path.join(traj_cache_dir, f'eps_{r}.pt')).to(device)

            dev_tea = cal_deviation(teacher_traj, bs=batch_gpu).mean(dim=0)
            dev_tea = torch.cat([dev_tea, torch.zeros_like(dev_tea[:1])])

            print0(f'Round {r + 1}/{num_accumulation_rounds} | Calculating the cost matrix...')
            for i, sigma_cur in enumerate(sigmas[:-1]):
                x_cur = teacher_traj[i]
                d_cur = eps_traj[i]

                for j in range(i + 1, num_steps_tea):
                    sigma_next = sigmas[j]
                    x_next = x_cur + (sigma_next - sigma_cur) * d_cur
                    if metric == 'l1':
                        cost_mat[i][j] += torch.norm(x_next - teacher_traj[j], p=1, dim=(1, 2, 3)).mean()
                    elif metric == 'l2':
                        cost_mat[i][j] += torch.norm(x_next - teacher_traj[j], p=2, dim=(1, 2, 3)).mean()
                    elif metric == 'dev':
                        temp = torch.cat(
                            (teacher_traj[0].unsqueeze(0), x_next.unsqueeze(0), teacher_traj[-1].unsqueeze(0)), dim=0)
                        dev_stu = cal_deviation(temp, bs=batch_gpu).mean(dim=0)
                        cost_mat[i][j] += (dev_stu - dev_tea[j - 1]).mean()
                    else:
                        raise NotImplementedError(f"Unknown metric: {metric}")

    used_prompts = pd.Series(used_prompts, name='Prompt')
    used_prompts.to_csv(os.path.join(traj_cache_dir, 'used_prompts.csv'))

    try:
        torch.distributed.all_reduce(cost_mat)
    except:
        pass

    cost_mat /= world_size * num_accumulation_rounds
    cost_mat = cost_mat.detach().cpu().numpy()

    # Description string.
    desc = f"{denoiser.scheduler}-{num_steps_tea}-warmup{num_warmup}-{metric}"

    # dynamic programming
    dp_list = dp(cost_mat, num_steps, num_steps_tea, coeff, True, desc, sigmas)

    return dp_list


# ----------------------------------------------------------------------------
# Dynamic programming

def dp(cost_mat, num_steps, num_steps_tea, coeff, multiple_coeff=False, desc=None, t_steps=None):
    K = num_steps - 1
    V = np.full((num_steps_tea, K + 1), np.inf)
    for i in range(num_steps_tea):
        V[i][1] = cost_mat[i][-1]
    for k in range(2, K + 1):
        for j in range(num_steps_tea - 1):
            for i in range(j + 1, num_steps_tea - 1):
                V[j][k] = min(V[j][k], cost_mat[j][i] + coeff * V[i][k - 1])
    phi, w = [0], 0
    for temp in range(K):
        k = K - temp
        for j in range(w + 1, num_steps_tea):
            if V[w][k] == cost_mat[w][j] + coeff * V[j][k - 1]:
                phi.append(j)
                w = j
                break
    phi.append(num_steps_tea - 1)
    dp_list = phi

    if multiple_coeff:
        # Output multiple dp_list and time schedule to a txt file with a list of coeffs for efficiency
        K = num_steps - 1
        for coeff in [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.10, 1.15, 1.2]:
            V = np.full((num_steps_tea, K + 1), np.inf)
            for i in range(num_steps_tea):
                V[i][1] = cost_mat[i][-1]
            for k in range(2, K + 1):
                for j in range(num_steps_tea - 1):
                    for i in range(j + 1, num_steps_tea - 1):
                        V[j][k] = min(V[j][k], cost_mat[j][i] + coeff * V[i][k - 1])

            if dist.get_rank() == 0:
                Note = open('dp_record.txt', mode='a')
                Note.write(f"{desc}-{coeff}\n")
                for K_temp in range(2, K + 1):
                    phi, w = [0], 0
                    for temp in range(K_temp):
                        k = K_temp - temp
                        for j in range(w + 1, num_steps_tea):
                            if V[w][k] == cost_mat[w][j] + coeff * V[j][k - 1]:
                                phi.append(j)
                                w = j
                                break
                    phi.append(num_steps_tea - 1)
                    Note.write(f"{phi} {[round(num.item(), 4) for num in t_steps[phi]]}\n")
                Note.close()
    return dp_list


# ----------------------------------------------------------------------------
# Calculate the deviation of the sampling trajectory

def cal_deviation(traj, bs=1):
    traj = traj.transpose(0, 1)
    # intermedia points, start point, end point
    a, b, c = traj[:, 1:-1], traj[:, 0].unsqueeze(1), traj[:, -1].unsqueeze(1)

    ac = c - a  # (bs, num_steps-1, ch, r, r)
    bc = c - b  # (bs, 1, ch, r, r)
    bc_unit = bc / torch.norm(bc, p=2, dim=(1, 2, 3, 4)).reshape(bs, 1, 1, 1, 1)  # (bs, 1, ch, r, r)

    # Calculate projection vector
    bc_unit_bcasted = bc_unit.expand_as(ac)  # (bs, num_steps-1, ch, r, r)
    temp = torch.sum(ac * bc_unit_bcasted, dim=(2, 3, 4))  # (bs, num_steps-1,)
    temp_expanded = temp.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (bs, num_steps-1, ch, r, r)
    ac_projection = temp_expanded * bc_unit

    # Calculate the deviation
    perp = ac - ac_projection  # (bs, num_steps-1, ch, r, r)
    norm = torch.norm(perp, p=2, dim=(2, 3, 4))
    return norm