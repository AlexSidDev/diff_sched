import torch
from pipelines.sdxl_pipeline import StableDiffusionXLPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from algorithms import StepsOptimizer
import pandas as pd
import os
import argparse
import shutil
from functools import partial


torch.manual_seed(42)


@torch.no_grad()
def encode_prompts(pipe, prompts, bs):
    results = []
    for ind in range(0, len(prompts), bs):
        prompt = prompts[ind, ind + bs]
        pr_emb, neg_pr_emd, pooled_emb, neg_pooled_emd = pipe.encode_prompt(prompt=prompt,
                                                                            device=args.device,
                                                                            do_classifier_free_guidance=True)
        embeds = torch.cat([pr_emb, neg_pr_emd])
        pooled_embeds = torch.cat([pooled_emb, neg_pooled_emd])
        results.append((embeds, pooled_embeds))

    return results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--text_prompt', type=str)
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--cache_dir', type=str, required=False)
    parser.add_argument('--teacher_cache_dir', type=str, required=True)
    parser.add_argument('--use_trajs', action='store_true')

    parser.add_argument('--num_steps', type=int, required=False, default=50)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--metric', type=str, choices=['dev', 'cosine'], required=False, default='dev')
    parser.add_argument('--num_samples', type=int, required=False, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = f'cuda{args.device}'

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, variant='fp16')
    pipe.to(device)
    config = pipe.scheduler.config
    pipe.scheduler = EulerDiscreteScheduler.from_config(config)

    data = pd.read_csv(args.text_prompt)
    prompts = encode_prompts(pipe, data['Prompt'].tolist(), args.batch_size)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(torch.Generator(device=device), None)

    trajs_paths = None
    if args.use_trajs:
        trajs_dir = 'updated_' + args.cache_dir
        shutil.copytree(args.cache_dir, trajs_dir, dir_exist_ok=True)

        trajs_paths = list(map(lambda path: os.path.join(trajs_dir, path),
                               filter(lambda path: path.startswith('traj'), sorted(os.listdir(trajs_dir)))))

    teacher_paths = list(map(lambda path: os.path.join(args.teacher_cache_dir, path),
                               filter(lambda path: path.startswith('traj'), sorted(os.listdir(args.teacher_cache_dir)))))

    denoiser = partial(
        pipe.denoise_step,
        extra_step_kwargs=extra_step_kwargs,
        guidance_scale=5.0
    )
    pipe.scheduler.set_timesteps(args.num_steps, device=device)

    pipe.vae = None
    pipe.text_encoder = None

    torch.cuda.empty_cache()

    os.makedirs(args.save_dir, exist_ok=True)

    optimizer = StepsOptimizer(denoiser, pipe.scheduler, prompts,
                               args.metric, teacher_paths, trajs_paths)

    schedule = pipe.scheduler.timesteps.tolist()
    num_samples = args.num_samples or len(teacher_paths)

    schedule = optimizer.optimization(schedule, num_samples)

    with open(os.path.join(args.save_dir, 'final_schedule.txt'), 'r') as fout:
        print(schedule, file=fout)


