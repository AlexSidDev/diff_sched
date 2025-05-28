import torch


def cal_deviation(x_end, x_start, x_next):
    bs = x_end.shape
    bc = x_end - x_start  # (bs, 1, ch, r, r)
    bc_unit = bc / torch.norm(bc, p=2, dim=(1, 2, 3, 4)).reshape(bs, 1, 1, 1, 1)  # (bs, 1, ch, r, r)

    # Calculate projection vector
    bc_unit_bcasted = bc_unit.expand_as(x_next)  # (bs, num_steps-1, ch, r, r)
    temp = torch.sum(x_next * bc_unit_bcasted, dim=(2, 3, 4))  # (bs, num_steps-1,)
    temp_expanded = temp.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (bs, num_steps-1, ch, r, r)
    ac_projection = temp_expanded * bc_unit

    # Calculate the deviation
    perp = x_next - ac_projection  # (bs, num_steps-1, ch, r, r)
    norm = torch.norm(perp, p=2, dim=(2, 3, 4))
    return norm.mean().item()


def cal_cosine_dist(x_end, x_cur, x_pred):
    cur_path = x_pred - x_cur
    teacher_path = x_end - x_cur
    norms = torch.norm(cur_path, p=2, dim=(1,2,3)) * torch.norm(teacher_path, p=2, dim=(1,2,3))
    cos_sim = (cur_path * teacher_path).sum(dim=(1,2,3)) / norms
    cos_dist = 1 - cos_sim
    return cos_dist.mean().item()


METRICS = {'dev': cal_deviation, 'cosine': cal_cosine_dist}
