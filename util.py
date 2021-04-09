from numba import jit
import numpy as np 

hyper_params = dict(
    total_stride=8,
    context_amount=0.5,
    test_lr=0.52,
    penalty_k=0.04,
    window_influence=0.21,
    windowing="cosine",
    z_size=127,
    x_size=303,
    num_conv3x3=3,
    min_w=10,
    min_h=10,
    phase_init="feature",
    phase_track="track",
)

@jit(nopython=True)
def change(r):
    return np.maximum(r, 1. / r)

@jit(nopython=True)
def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)
    
@jit(nopython=True)
def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)

@jit(nopython=True)
def postprocess_score(score, box_wh, target_sz, scale_x, window):
    r"""
    Perform SiameseRPN-based tracker's post-processing of score
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_sz: previous state (w & h)
    :param scale_x:
    :return:
        best_pscore_id: index of chosen candidate along axis HW
        pscore: (HW, ), penalized score
        penalty: (HW, ), penalty due to scale/ratio change
    """
    # size penalty
    penalty_k = 0.04
    target_sz_in_crop = target_sz * scale_x
    s_c = change(
        sz(box_wh[:, 2], box_wh[:, 3]) /
        (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                    (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
    penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
    pscore = penalty * score

    # ipdb.set_trace()
    # cos window (motion model)
    window_influence = 0.21
    pscore = pscore * (
        1 - window_influence) + window * window_influence
    best_pscore_id = np.argmax(pscore)

    return best_pscore_id, pscore, penalty

@jit(nopython=True)
def postprocess_box(best_pscore_id, score, box_wh, target_pos, target_sz, scale_x, x_size, penalty):
    r"""
    Perform SiameseRPN-based tracker's post-processing of box
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_pos: (2, ) previous position (x & y)
    :param target_sz: (2, ) previous state (w & h)
    :param scale_x: scale of cropped patch of current frame
    :param x_size: size of cropped patch
    :param penalty: scale/ratio change penalty calculated during score post-processing
    :return:
        new_target_pos: (2, ), new target position
        new_target_sz: (2, ), new target size
    """
    pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
    # about np.float32(scale_x)
    # attention!, this casting is done implicitly
    # which can influence final EAO heavily given a model & a set of hyper-parameters

    # box post-postprocessing
    test_lr = 0.52
    lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
    res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
    res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    new_target_pos = np.array([res_x, res_y])
    new_target_sz = np.array([res_w, res_h])

    return new_target_pos, new_target_sz

@jit(nopython=True)
def restrict_box(im_h, im_w, target_pos, target_sz):
    r"""
    Restrict target position & size
    :param target_pos: (2, ), target position
    :param target_sz: (2, ), target size
    :return:
        target_pos, target_sz
    """
    target_pos[0] = max(0, min(im_w, target_pos[0]))
    target_pos[1] = max(0, min(im_h, target_pos[1]))
    target_sz[0] = max(5,
                        min(im_w, target_sz[0]))
    target_sz[1] = max(10,
                        min(im_h, target_sz[1]))

    return target_pos, target_sz