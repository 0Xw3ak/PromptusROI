import time

import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import cv2 as cv

from streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from lossbuilder import LossBuilder
from quantization import QParam, FakeQuantize

VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    value_dict = {}
    for key in keys:
        if key == "txt":
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = ""

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict


def sample(
        model,
        sampler,
        prompt="A scenic landscape with a sky filled with clouds above, a lake with reflections below. In the distance, there are trees and a setting sun.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    value_dict = init_embedder_options(
        keys=get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict={
            "orig_width": W,
            "orig_height": H,
            "target_width": W,
            "target_height": H,
        },
        prompt=prompt,
    )

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    # with torch.no_grad():
    with precision_scope("cuda"):
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(model.conditioner),
            value_dict,
            [1],
        )
        c = model.conditioner(batch)
        uc = None
        randn = seeded_randn(shape, seed)

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        samples_z = sampler(denoiser, randn, cond=c, uc=uc)
        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        if filter is not None:
            samples = filter(samples)
        samples = (
            (255 * samples)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
        )
    return samples

def sample_inv(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        c = torch.load('/home/wjk/workspace/PyProject/generative-models/c.pth')#model.conditioner(batch)
        # c['crossattn'] = torch.randn_like(c['crossattn'])
        # c['vector'] = torch.randn_like(c['vector'])
        c['crossattn'].requires_grad = True
        c['vector'].requires_grad = True
        uc = None
        randn = seeded_randn(shape, seed)
        # randn.requires_grad = True

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def decode(samples_z):
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            if filter is not None:
                samples = filter(samples)
            samples = (
                (255 * samples)
                    .to(dtype=torch.uint8)
                    .permute(0, 2, 3, 1)
                    .detach()
                    .cpu()
                    .numpy()
            )
            return samples

        loss_fn = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam([c['crossattn'], c['vector'], randn], lr=0.1)
        optimizer = torch.optim.Adam([c['crossattn'], c['vector']], lr=0.1)
        gt = torch.load('/home/wjk/workspace/PyProject/generative-models/samples_z.pth')

        for _ in range(10000):
            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
            loss = loss_fn(samples_z, gt)
            print('iter: {}, loss: {}'.format(_, loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
            # model.model.diffusion_model.zero_grad()
            # model.denoiser.zero_grad()
            if _ % 2 == 0:
                samples = decode(samples_z)
                img = samples[0][:, :, ::-1]
                cv2.imwrite('tmp/{:05d}.png'.format(_), img)

    return samples

def sample_inv_sd(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        # token_embedding = torch.load('/home/wjk/workspace/PyProject/generative-models/x_token_embedding.pth')
        # token_embedding.requires_grad = True
        # token_embedding = {'txt':token_embedding}
        # c = model.conditioner(token_embedding)
        # if False:
        # c = torch.load('/home/wjk/workspace/PyProject/generative-models/c_sd.pth')#model.conditioner(batch)
        # c['crossattn'] = torch.randn_like(c['crossattn'])
        # c['crossattn'].requires_grad = True
        rank = 16
        # u, s, v = torch.linalg.svd(c['crossattn'][0],full_matrices=False)
        # U, S, V = u[:, :rank], s[:rank], v[:rank, :]
        # U = U @ torch.diag(S)
        U = torch.rand([77, rank]).float().cuda()
        U.requires_grad = True
        V = torch.rand([rank, 1024]).float().cuda()
        V.requires_grad = True

        uc = None
        sigma_1 = torch.Tensor([0.05]).float().cuda()
        # sigma_1.requires_grad = True
        sigma_2 = torch.Tensor([0.95]).float().cuda()
        # sigma_2.requires_grad = True
        # randn = seeded_randn(shape, seed)
        #randn = (torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth') * sigma + seeded_randn(shape, seed) * (1 - sigma))#.detach()#torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth')
        # randn.requires_grad = False
        # randn.requires_grad = True
        rand_noise = seeded_randn(shape, seed)

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        def standardize(tensor):
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            return (tensor - mean) / std

        # prev_frame = torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth')
        prev_frame = model.encode_first_stage(
            load_img('/data/wjk/sky_timelapse/sky_train/4u-U0bOgs94/4u-U0bOgs94_1/4u-U0bOgs94_frames_00000406.jpg'))
        loss_fn = torch.nn.MSELoss()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
        # optimizer = torch.optim.Adam([token_embedding['txt'], randn], lr=0.1)
        # optimizer = torch.optim.Adam([c['crossattn'], randn], lr=0.1)
        # optimizer = torch.optim.Adam([c['crossattn'], sigma_1, sigma_2], lr=0.1)
        # optimizer = torch.optim.Adam([c['crossattn']], lr=0.1)
        optimizer = torch.optim.Adam([U, V], lr=0.1)
        # optimizer = torch.optim.Adam([U, V, sigma_1, sigma_2], lr=0.1)
        # gt = torch.load('/home/wjk/workspace/PyProject/generative-models/samples_z.pth')
        # gt = torch.load('/home/wjk/workspace/PyProject/generative-models/samples_x.pth')
        # gt = torch.clamp(gt, min=-1.0, max=1.0)
        gt = load_img('/data/wjk/sky_timelapse/sky_train/4u-U0bOgs94/4u-U0bOgs94_1/4u-U0bOgs94_frames_00000412.jpg')
        gt.requires_grad = True
        min_loss = 1e9
        ckpt_prev = torch.load('result/406_rank16/ckpt.pth')
        U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
        # U, V = U_prev, V_prev
        for _ in range(4000):
            # c = {'crossattn': standardize((U @ V).unsqueeze(dim=0))}
            c = {'crossattn': (U @ V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
            randn = (prev_frame * sigma_1 + rand_noise * sigma_2)
            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
            samples_x = model.decode_first_stage(samples_z)
            samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)
            loss = 0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
            smooth = loss_fn(U, U_prev) + loss_fn(V, V_prev)
            loss = loss + 0.01 * smooth
            print('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth))
            # std_loss = loss_fn(c['crossattn'].std(), torch.tensor([1.0]).cuda())
            # loss = loss + 0.1 * std_loss
            if _ < 20:
                loss_regu = torch.mean(torch.abs(c['crossattn']))
                loss = loss + 0.1 * loss_regu
            elif loss < min_loss:
                min_loss = loss
                ckpt = {
                    'U' : U,
                    'V' : V,
                    'randn' : randn,
                    'iter' : _,
                    'loss' : loss,
                }
                torch.save(ckpt, 'tmp/ckpt.pth')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
            # model.model.diffusion_model.zero_grad()
            # model.denoiser.zero_grad()
            if _ % 2 == 0:
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                if filter is not None:
                    samples = filter(samples)
                samples = (
                    (255 * samples)
                        .to(dtype=torch.uint8)
                        .permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                )
                img = samples[0][:, :, ::-1]
                cv2.imwrite('tmp/{:05d}.png'.format(_), img)

    return samples

def sample_inv_sd_sequence(
        model,
        sampler,
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [832, 832])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img
        def TVLoss(U, V):
            UV = torch.concatenate([V, U.T], dim=1)
            UV = UV.reshape(rank, -1, 3)
            dx = UV[:, 1:, :] - UV[:, :-1, :]
            smoothness_reg = torch.mean(torch.abs(dx))# + torch.mean(dy ** 2)
            return smoothness_reg

        rank = 8
        uc = None
        rand_noise = seeded_randn(shape, seed)

        for f_id in range(1, 150):
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            Quant_Param_U = QParam(num_bits=8)
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            Quant_Param_V = QParam(num_bits=8)
            sigma_1 = torch.Tensor([0.05]).float().cuda()
            sigma_2 = torch.Tensor([0.95]).float().cuda()
            prev_frame = model.encode_first_stage(
                # load_img('/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))
                load_img('tmp/images/00000.png').to('cuda:1')).to('cuda:0')
            loss_fn = torch.nn.MSELoss()
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
            builder = LossBuilder('cuda')
            style_layers = [('conv_1', 1), ('conv_3', 1), ('conv_5', 1)]
            # content_layers = [('conv_1',0.026423), ('conv_2', 0.009285), ('conv_3',0.006710), ('conv_4',0.004898),('conv_5',0.003910),('conv_6',0.003956),('conv_7',0.003813),('conv_8',0.002968),('conv_9',0.002997),('conv_10',0.003631),('conv_11',0.004147),('conv_12',0.005765),('conv_13',0.007442),('conv_14',0.009666),('conv_15',0.012586),('conv_16',0.013377)]
            content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                              ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                              ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                              ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                              ('conv_16', 1)]
            pt_loss, style_losses, content_losses = builder.get_style_and_content_loss(dict(content_layers),
                                                                                       dict(style_layers))
            optimizer = torch.optim.Adam([U, V], lr=0.1)
            gt = load_img('tmp/images/{:05d}.png'.format(f_id))#load_img('/data/wph/N3DV_dataset/frame_coffee_martini/{:04d}/images/cam00.png'.format(f_id))
            gt.requires_grad = True
            min_loss = 1e9
            if not os.path.exists('tmp/{:04d}/'.format(f_id)):
                os.mkdir('tmp/{:04d}/'.format(f_id))
            log_output = open('tmp/{:04d}/log.txt'.format(f_id), 'a')
            if f_id > 1:
                ckpt_prev = torch.load('tmp/{:04d}/ckpt.pth'.format(f_id - 1))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
            else:
                ckpt_prev = None
            for _ in range(10000):
                Quant_Param_U.update(U)
                Q_U = FakeQuantize.apply(U, Quant_Param_U)
                Quant_Param_V.update(V)
                Q_V = FakeQuantize.apply(V, Quant_Param_V)
                c = {'crossattn': (Q_U @ Q_V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
                randn = (prev_frame * sigma_1 + rand_noise * sigma_2)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z.to('cuda:1')).to('cuda:0')
                # samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)

                input_images = torch.cat([gt, samples_x], dim=0)#torch.cat([(gt + 1.0) / 2.0, (samples_x + 1.0) / 2.0], dim=0)
                pt_loss(input_images)

                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                style_score = style_score / (len(style_losses) + 1e-9)
                for cl in content_losses:
                    content_score += cl.loss
                content_score = content_score / (len(content_losses) + 1e-9)

                loss = 0.2 * content_score + 0.8 * loss_fn(samples_x, gt)#0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
                # V_TV = TVLoss(Q_U, Q_V)
                # loss = loss + 50 * V_TV
                if f_id > 1:
                    smooth = loss_fn(Q_U, U_prev) + loss_fn(Q_V, V_prev)
                    if smooth > 0.1:
                        loss = loss + 5.0 * smooth
                else:
                    smooth = None
                print('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.write('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}\n'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.flush()
                if _ < 20:
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu
                elif loss < min_loss:
                    min_loss = loss
                    ckpt = {
                        'U' : Q_U,
                        'U_scale' : Quant_Param_U.scale,
                        'U_zero_point' : Quant_Param_U.zero_point,
                        'U_bits' : Quant_Param_U.num_bits,
                        'V' : Q_V,
                        'V_scale' : Quant_Param_V.scale,
                        'V_zero_point' : Quant_Param_V.zero_point,
                        'V_bits' : Quant_Param_V.num_bits,
                        'randn' : randn,
                        'iter' : _,
                        'loss' : loss,
                    }
                    torch.save(ckpt, 'tmp/{:04d}/ckpt.pth'.format(f_id))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.model.zero_grad()
                # model.model.diffusion_model.zero_grad()
                # model.denoiser.zero_grad()
                if _ % 2 == 0:
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    if filter is not None:
                        samples = filter(samples)
                    samples = (
                        (255 * samples)
                            .to(dtype=torch.uint8)
                            .permute(0, 2, 3, 1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    img = samples[0][:, :, ::-1]
                    cv2.imwrite('tmp/{:04d}/{:05d}.png'.format(f_id, _), img)
            log_output.close()

    return samples

def sample_inv_sd_sequence_patch(
        model,
        sampler,
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [1024, 1024])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img
        def TVLoss(U, V):
            UV = torch.concatenate([V, U.T], dim=1)
            UV = UV.reshape(rank, -1, 3)
            dx = UV[:, 1:, :] - UV[:, :-1, :]
            smoothness_reg = torch.mean(torch.abs(dx))# + torch.mean(dy ** 2)
            return smoothness_reg

        rank = 8
        uc = None
        rand_noise = seeded_randn(shape, seed)

        for f_id in range(1, 150):
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            Quant_Param_U = QParam(num_bits=8)
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            Quant_Param_V = QParam(num_bits=8)
            sigma_1 = torch.Tensor([0.05]).float().cuda()
            sigma_2 = torch.Tensor([0.95]).float().cuda()
            prev_frame = model.encode_first_stage(
                load_img('/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))
            loss_fn = torch.nn.MSELoss()
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
            builder = LossBuilder('cuda')
            style_layers = [('conv_1', 1), ('conv_3', 1), ('conv_5', 1)]
            # content_layers = [('conv_1',0.026423), ('conv_2', 0.009285), ('conv_3',0.006710), ('conv_4',0.004898),('conv_5',0.003910),('conv_6',0.003956),('conv_7',0.003813),('conv_8',0.002968),('conv_9',0.002997),('conv_10',0.003631),('conv_11',0.004147),('conv_12',0.005765),('conv_13',0.007442),('conv_14',0.009666),('conv_15',0.012586),('conv_16',0.013377)]
            content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                              ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                              ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                              ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                              ('conv_16', 1)]
            pt_loss, style_losses, content_losses = builder.get_style_and_content_loss(dict(content_layers),
                                                                                       dict(style_layers))
            optimizer = torch.optim.Adam([U, V], lr=0.1)
            gt = load_img('/data/wph/N3DV_dataset/frame_coffee_martini/{:04d}/images/cam00.png'.format(f_id))
            gt.requires_grad = True
            min_loss = 1e9
            if not os.path.exists('tmp/{:04d}/'.format(f_id)):
                os.mkdir('tmp/{:04d}/'.format(f_id))
            log_output = open('tmp/{:04d}/log.txt'.format(f_id), 'a')
            if f_id > 1:
                ckpt_prev = torch.load('tmp/{:04d}/ckpt.pth'.format(f_id - 1))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
            else:
                ckpt_prev = None
            for _ in range(10000):
                Quant_Param_U.update(U)
                Q_U = FakeQuantize.apply(U, Quant_Param_U)
                Quant_Param_V.update(V)
                Q_V = FakeQuantize.apply(V, Quant_Param_V)
                c = {'crossattn': (Q_U @ Q_V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
                randn = (prev_frame * sigma_1 + rand_noise * sigma_2)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                if _ % 4 == 0:
                    z_patch = samples_z[:,:,:64,:64]
                    gt_patch = gt[:,:,:512,:512]
                elif _ % 4 == 1:
                    z_patch = samples_z[:,:,:64,-64:]
                    gt_patch = gt[:,:,:512,-512:]
                elif _ % 4 == 2:
                    z_patch = samples_z[:,:,-64:,:64]
                    gt_patch = gt[:,:,-512:,:512]
                elif _ % 4 == 3:
                    z_patch = samples_z[:,:,-64:,-64:]
                    gt_patch = gt[:,:,-512:,-512:]
                samples_x_patch = model.decode_first_stage(z_patch)
                # samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)

                input_images = torch.cat([gt_patch, samples_x_patch], dim=0)#torch.cat([(gt + 1.0) / 2.0, (samples_x + 1.0) / 2.0], dim=0)
                pt_loss(input_images)

                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                style_score = style_score / (len(style_losses) + 1e-9)
                for cl in content_losses:
                    content_score += cl.loss
                content_score = content_score / (len(content_losses) + 1e-9)

                loss = 0.2 * content_score + 0.8 * loss_fn(samples_x_patch, gt_patch)#0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
                # V_TV = TVLoss(Q_U, Q_V)
                # loss = loss + 50 * V_TV
                if f_id > 1:
                    smooth = loss_fn(Q_U, U_prev) + loss_fn(Q_V, V_prev)
                    if smooth > 0.1:
                        loss = loss + 5.0 * smooth
                else:
                    smooth = None
                print('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.write('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}\n'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.flush()
                if _ < 20:
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu
                elif loss < min_loss:
                    min_loss = loss
                    ckpt = {
                        'U' : Q_U,
                        'U_scale' : Quant_Param_U.scale,
                        'U_zero_point' : Quant_Param_U.zero_point,
                        'U_bits' : Quant_Param_U.num_bits,
                        'V' : Q_V,
                        'V_scale' : Quant_Param_V.scale,
                        'V_zero_point' : Quant_Param_V.zero_point,
                        'V_bits' : Quant_Param_V.num_bits,
                        'randn' : randn,
                        'iter' : _,
                        'loss' : loss,
                    }
                    torch.save(ckpt, 'tmp/{:04d}/ckpt.pth'.format(f_id))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.model.zero_grad()
                # model.model.diffusion_model.zero_grad()
                # model.denoiser.zero_grad()
                if _ % 1 == 0:
                    samples = torch.clamp((samples_x_patch + 1.0) / 2.0, min=0.0, max=1.0)
                    if filter is not None:
                        samples = filter(samples)
                    samples = (
                        (255 * samples)
                            .to(dtype=torch.uint8)
                            .permute(0, 2, 3, 1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    img = samples[0][:, :, ::-1]
                    cv2.imwrite('tmp/{:04d}/{:05d}.png'.format(f_id, _), img)
            log_output.close()

    return samples
def sample_inv_sd_sequence_latent(
        model,
        sampler,
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img
        def TVLoss(U, V):
            UV = torch.concatenate([V, U.T], dim=1)
            UV = UV.reshape(rank, -1, 3)
            dx = UV[:, 1:, :] - UV[:, :-1, :]
            smoothness_reg = torch.mean(torch.abs(dx))# + torch.mean(dy ** 2)
            return smoothness_reg

        rank = 8
        uc = None
        rand_noise = seeded_randn(shape, seed)

        for f_id in range(1, 150):
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            Quant_Param_U = QParam(num_bits=8)
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            Quant_Param_V = QParam(num_bits=8)
            sigma_1 = torch.Tensor([0.05]).float().cuda()
            sigma_2 = torch.Tensor([0.95]).float().cuda()
            prev_frame = model.encode_first_stage(
                load_img('/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))
            loss_fn = torch.nn.MSELoss()
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
            builder = LossBuilder('cuda')
            style_layers = [('conv_1', 1), ('conv_3', 1), ('conv_5', 1)]
            # content_layers = [('conv_1',0.026423), ('conv_2', 0.009285), ('conv_3',0.006710), ('conv_4',0.004898),('conv_5',0.003910),('conv_6',0.003956),('conv_7',0.003813),('conv_8',0.002968),('conv_9',0.002997),('conv_10',0.003631),('conv_11',0.004147),('conv_12',0.005765),('conv_13',0.007442),('conv_14',0.009666),('conv_15',0.012586),('conv_16',0.013377)]
            content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                              ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                              ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                              ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                              ('conv_16', 1)]
            pt_loss, style_losses, content_losses = builder.get_style_and_content_loss(dict(content_layers),
                                                                                       dict(style_layers))
            optimizer = torch.optim.Adam([U, V], lr=0.1)
            gt = model.encode_first_stage(load_img('/data/wph/N3DV_dataset/frame_coffee_martini/{:04d}/images/cam00.png'.format(f_id)))
            gt.requires_grad = True
            min_loss = 1e9
            if not os.path.exists('tmp/{:04d}/'.format(f_id)):
                os.mkdir('tmp/{:04d}/'.format(f_id))
            log_output = open('tmp/{:04d}/log.txt'.format(f_id), 'a')
            if f_id > 1:
                ckpt_prev = torch.load('tmp/{:04d}/ckpt.pth'.format(f_id - 1))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
            else:
                ckpt_prev = None
            for _ in range(10000):
                Quant_Param_U.update(U)
                Q_U = FakeQuantize.apply(U, Quant_Param_U)
                Quant_Param_V.update(V)
                Q_V = FakeQuantize.apply(V, Quant_Param_V)
                c = {'crossattn': (Q_U @ Q_V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
                randn = (prev_frame * sigma_1 + rand_noise * sigma_2)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                # samples_x = model.decode_first_stage(samples_z)
                # # samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)
                #
                # input_images = torch.cat([gt, samples_x], dim=0)#torch.cat([(gt + 1.0) / 2.0, (samples_x + 1.0) / 2.0], dim=0)
                # pt_loss(input_images)

                # style_score = 0
                # content_score = 0
                # for sl in style_losses:
                #     style_score += sl.loss
                # style_score = style_score / (len(style_losses) + 1e-9)
                # for cl in content_losses:
                #     content_score += cl.loss
                # content_score = content_score / (len(content_losses) + 1e-9)
                #
                # loss = 0.2 * content_score + 0.8 * loss_fn(samples_x, gt)#0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
                loss = loss_fn(samples_z, gt)
                if f_id > 1:
                    smooth = loss_fn(Q_U, U_prev) + loss_fn(Q_V, V_prev)
                    if smooth > 0.1:
                        loss = loss + 5.0 * smooth
                else:
                    smooth = None
                print('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.write('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}, V_TV: {}\n'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth, None))
                log_output.flush()
                if _ < 20:
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu
                elif loss < min_loss:
                    min_loss = loss
                    ckpt = {
                        'U' : Q_U,
                        'U_scale' : Quant_Param_U.scale,
                        'U_zero_point' : Quant_Param_U.zero_point,
                        'U_bits' : Quant_Param_U.num_bits,
                        'V' : Q_V,
                        'V_scale' : Quant_Param_V.scale,
                        'V_zero_point' : Quant_Param_V.zero_point,
                        'V_bits' : Quant_Param_V.num_bits,
                        'randn' : randn,
                        'iter' : _,
                        'loss' : loss,
                    }
                    torch.save(ckpt, 'tmp/{:04d}/ckpt.pth'.format(f_id))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.model.zero_grad()
                # model.model.diffusion_model.zero_grad()
                # model.denoiser.zero_grad()
                if _ % 2 == 0:
                    samples_x = model.decode_first_stage(samples_z.detach())
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    if filter is not None:
                        samples = filter(samples)
                    samples = (
                        (255 * samples)
                            .to(dtype=torch.uint8)
                            .permute(0, 2, 3, 1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    img = samples[0][:, :, ::-1]
                    cv2.imwrite('tmp/{:04d}/{:05d}.png'.format(f_id, _), img)
            log_output.close()

    return samples

def sample_inv_sd_sequence_cp(
        model,
        sampler,
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        rank = 64
        uc = None
        rand_noise = seeded_randn(shape, seed)

        for f_id in range(1, 150):
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            Quant_Param_U = QParam(num_bits=8)
            V = torch.rand([32, rank]).float().cuda()
            V.requires_grad = True
            Quant_Param_V = QParam(num_bits=8)
            W = torch.rand([32, rank]).float().cuda()
            W.requires_grad = True
            Quant_Param_W = QParam(num_bits=8)
            sigma_1 = torch.Tensor([0.05]).float().cuda()
            sigma_2 = torch.Tensor([0.95]).float().cuda()
            prev_frame = model.encode_first_stage(
                load_img('/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))
            loss_fn = torch.nn.MSELoss()
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
            builder = LossBuilder('cuda')
            style_layers = [('conv_1', 1), ('conv_3', 1), ('conv_5', 1)]
            # content_layers = [('conv_1',0.026423), ('conv_2', 0.009285), ('conv_3',0.006710), ('conv_4',0.004898),('conv_5',0.003910),('conv_6',0.003956),('conv_7',0.003813),('conv_8',0.002968),('conv_9',0.002997),('conv_10',0.003631),('conv_11',0.004147),('conv_12',0.005765),('conv_13',0.007442),('conv_14',0.009666),('conv_15',0.012586),('conv_16',0.013377)]
            content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                              ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                              ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                              ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                              ('conv_16', 1)]
            pt_loss, style_losses, content_losses = builder.get_style_and_content_loss(dict(content_layers),
                                                                                       dict(style_layers))
            optimizer = torch.optim.Adam([U, V, W], lr=0.1)
            gt = load_img('/data/wph/N3DV_dataset/frame_coffee_martini/{:04d}/images/cam00.png'.format(f_id))
            gt.requires_grad = True
            min_loss = 1e9
            if not os.path.exists('tmp/{:04d}/'.format(f_id)):
                os.mkdir('tmp/{:04d}/'.format(f_id))
            log_output = open('tmp/{:04d}/log.txt'.format(f_id), 'a')
            if f_id > 1:
                ckpt_prev = torch.load('tmp/{:04d}/ckpt.pth'.format(f_id - 1))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
            else:
                ckpt_prev = None
            for _ in range(10000):
                Quant_Param_U.update(U)
                Q_U = FakeQuantize.apply(U, Quant_Param_U)
                Quant_Param_V.update(V)
                Q_V = FakeQuantize.apply(V, Quant_Param_V)
                Quant_Param_W.update(W)
                Q_W = FakeQuantize.apply(W, Quant_Param_W)
                C = torch.einsum('ir,jr,kr->ijk', Q_U, Q_V, Q_W)
                C = (C.reshape(77, 1024) / torch.pow(torch.tensor([rank]).cuda(), 1.0/3.0))
                #c = {'crossattn': (Q_U @ Q_V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
                c = {'crossattn': C.unsqueeze(dim=0).float()}
                randn = (prev_frame * sigma_1 + rand_noise * sigma_2)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)
                # samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)

                input_images = torch.cat([gt, samples_x], dim=0)#torch.cat([(gt + 1.0) / 2.0, (samples_x + 1.0) / 2.0], dim=0)
                pt_loss(input_images)

                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                style_score = style_score / (len(style_losses) + 1e-9)
                for cl in content_losses:
                    content_score += cl.loss
                content_score = content_score / (len(content_losses) + 1e-9)

                loss = 0.2 * content_score + 0.8 * loss_fn(samples_x, gt)#0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
                if f_id > 1:
                    smooth = loss_fn(Q_U, U_prev) + loss_fn(Q_V, V_prev)
                    if smooth > 0.1:
                        loss = loss + 5.0 * smooth
                else:
                    smooth = None
                print('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth))
                log_output.write('iter: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}, smooth: {}\n'.format(_, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std(), smooth))
                log_output.flush()
                if _ < 20:
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu
                elif loss < min_loss:
                    min_loss = loss
                    ckpt = {
                        'U' : Q_U,
                        'U_scale' : Quant_Param_U.scale,
                        'U_zero_point' : Quant_Param_U.zero_point,
                        'U_bits' : Quant_Param_U.num_bits,
                        'V' : Q_V,
                        'V_scale' : Quant_Param_V.scale,
                        'V_zero_point' : Quant_Param_V.zero_point,
                        'V_bits' : Quant_Param_V.num_bits,
                        'randn' : randn,
                        'iter' : _,
                        'loss' : loss,
                    }
                    torch.save(ckpt, 'tmp/{:04d}/ckpt.pth'.format(f_id))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.model.zero_grad()
                # model.model.diffusion_model.zero_grad()
                # model.denoiser.zero_grad()
                if _ % 2 == 0:
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    if filter is not None:
                        samples = filter(samples)
                    samples = (
                        (255 * samples)
                            .to(dtype=torch.uint8)
                            .permute(0, 2, 3, 1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    img = samples[0][:, :, ::-1]
                    cv2.imwrite('tmp/{:04d}/{:05d}.png'.format(f_id, _), img)
            log_output.close()

    return samples

def sample_sd(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        c = torch.load('/home/wjk/workspace/PyProject/generative-models/sky_c_3.4k_prev_frame.pth')
        uc = None
        sigma_1 = torch.Tensor([0.05]).float().cuda()
        sigma_2 = torch.Tensor([0.95]).float().cuda()
        rand_noise = seeded_randn(shape, seed)
        prev_frame = torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth')

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        loss_fn = torch.nn.MSELoss()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
        gt = load_img('/data/wjk/sky_timelapse/sky_train/4u-U0bOgs94/4u-U0bOgs94_1/4u-U0bOgs94_frames_00000412.jpg')

        for t in range(100):
            randn = (prev_frame * sigma_1 + rand_noise * sigma_2).detach()
            # noise_for_c = seeded_randn(c['crossattn'].shape, seed).detach()
            # c_noise = {'crossattn': c['crossattn'].detach() + t * 0.01 * noise_for_c}

            # rand_mask = torch.rand(c['crossattn'].shape).cuda().detach() > t * 0.01
            # c_noise = {'crossattn': c['crossattn'].detach() * rand_mask}

            noise_for_c = seeded_randn(c['crossattn'].shape, seed).detach()
            rand_mask = torch.rand(c['crossattn'].shape).cuda().detach() > t * 0.01
            c_noise = {'crossattn': c['crossattn'].detach() * rand_mask + noise_for_c * (~rand_mask)}

            samples_z = sampler(denoiser, randn, cond=c_noise, uc=uc)
            samples_x = model.decode_first_stage(samples_z)
            samples_x = torch.clamp(samples_x, min=-1.0, max=1.0).detach()
            loss = 0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
            print('noise: {}, loss: {}'.format(t, loss))
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            if filter is not None:
                samples = filter(samples)
                samples = (
                    (255 * samples)
                        .to(dtype=torch.uint8)
                        .permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                )
                img = samples[0][:, :, ::-1]
                cv2.imwrite('tmp/{:05d}.png'.format(t), img)

    return samples

def sample_sd_interplation(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        uc = None
        sigma_1 = torch.Tensor([0.05]).float().cuda()
        sigma_2 = torch.Tensor([0.95]).float().cuda()
        rand_noise = seeded_randn(shape, seed)

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        prev_frame = model.encode_first_stage(
            load_img(
                '/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))  # torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth')
        loss_fn = torch.nn.MSELoss()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
        gt = load_img('/data/wph/N3DV_dataset/frame_coffee_martini/{:04d}/images/cam00.png'.format(0))#load_img('/data/wjk/sky_timelapse/sky_train/4u-U0bOgs94/4u-U0bOgs94_1/4u-U0bOgs94_frames_00000412.jpg')
        ckpt1 = torch.load('tmp/0001_our_lpips_mse_[-1,1]_0.2_0.8_8bits_rank8/ckpt.pth')#torch.load('result/406_rank16/ckpt.pth')
        ckpt2 = torch.load('tmp/0005_our_lpips_mse_[-1,1]_0.2_0.8_8bits_rank8/ckpt.pth')#torch.load('result/412_rank16/ckpt.pth')
        U1, V1, U2, V2 = ckpt1['U'], ckpt1['V'], ckpt2['U'], ckpt2['V']
        rank = U1.shape[1]
        c1, c2 = (U1 @ V1 / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float(), (U2 @ V2 / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()

        for t in range(5):
            # U = (1 - t * 0.2) * U1 + (t * 0.2) * U2
            # V = (1 - t * 0.2) * V1 + (t * 0.2) * V2
            c = (1 - t * 0.25) * c1 + (t * 0.25) * c2
            randn = (prev_frame * sigma_1 + rand_noise * sigma_2).detach()
            # c = {'crossattn': (U @ V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
            c = {'crossattn': c}

            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
            samples_x = model.decode_first_stage(samples_z)
            samples_x = torch.clamp(samples_x, min=-1.0, max=1.0).detach()
            loss = 0.2 * lpips(samples_x, gt) + 0.8 * loss_fn(samples_x, gt)
            print('noise: {}, loss: {}'.format(t, loss))
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            if filter is not None:
                samples = filter(samples)
                samples = (
                    (255 * samples)
                        .to(dtype=torch.uint8)
                        .permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                )
                img = samples[0][:, :, ::-1]
                cv2.imwrite('tmp/{:05d}.png'.format(t), img)

    return samples

def sample_sd_c(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with (precision_scope("cuda")):
        uc = None
        sigma_1 = torch.Tensor([0.05]).float().cuda()
        sigma_2 = torch.Tensor([0.95]).float().cuda()
        rand_noise = seeded_randn(shape, seed)

        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        # prev_frame = model.encode_first_stage(
        #     load_img(
        #         '/data/wph/N3DV_dataset/frame_coffee_martini/0000/images/cam00.png'))  # torch.load('/home/wjk/workspace/PyProject/generative-models/sky_z_1k.pth')
        loss_mse = torch.nn.MSELoss()
        loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
        # mse = []
        # lpips = []

        for t in range(2, 10):
            for r in range(0, 9):
                prev_frame = model.encode_first_stage(
                    load_img(
                        'tmp/images_2048/{:05d}.png'.format(r)))
                gt = load_img('tmp/images_2048/{:05d}.png'.format(t))
                ckpt = torch.load(
                    'tmp/{:04d}_our_lpips_mse_[-1,1]_0.2_0.8_8bits_rank8_L2_5.0_filter_0.1/ckpt.pth'.format(t))  # torch.load('result/406_rank16/ckpt.pth')
                U, V = ckpt['U'], ckpt['V']
                rank = U.shape[1]
                c = (U @ V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()
                randn = (prev_frame * sigma_1 + rand_noise * sigma_2).detach()
                # c = {'crossattn': (U @ V / torch.sqrt(torch.tensor([rank])).cuda()).unsqueeze(dim=0).float()}
                c = {'crossattn': c}

                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)
                samples_x = torch.clamp(samples_x, min=-1.0, max=1.0).detach()
                # lpips.append(loss_lpips(samples_x, gt).item())
                # mse.append(loss_mse(samples_x, gt).item())
                print('ref:{}, pred:{}, lpips:{}, mse:{}'.format(r,t,loss_lpips(samples_x, gt).item(),loss_mse(samples_x, gt).item()))
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                if filter is not None:
                    samples = filter(samples)
                    samples = (
                        (255 * samples)
                            .to(dtype=torch.uint8)
                            .permute(0, 2, 3, 1)
                            .detach()
                            .cpu()
                            .numpy()
                    )
                    img = samples[0][:, :, ::-1]
                    cv2.imwrite('tmp/ref{:05d}_{:05d}.png'.format(r,t), img)
        # print(mse, ', mean: {}'.format(np.mean(mse)))
        # print(lpips, ', mean: {}'.format(np.mean(lpips)))

    return samples

from quantization import quantize_tensor, dequantize_tensor
def VAE_encoder(
        model,
        sampler,
        H=1024,
        W=1024,
        seed=0,
        filter=None,
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    loss_mse = torch.nn.MSELoss()
    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
    mse_list = []
    lpips_list = []

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        # Quant_P = {}
        Quant_P = torch.load('data/cartoon_2/VAE/Quant_Parm.pth')
        cap = cv2.VideoCapture('data/cartoon_2/VAE/80k_x265.mp4')
        for f_id in range(60):
            img = load_img('data/cartoon_2/{:05d}.png'.format(f_id))
            # img_z = model.encode_first_stage(img)

            # #Quant and save as img
            # Quant_Param = QParam(num_bits=8)
            # Quant_Param.update(img_z)
            # z_q = quantize_tensor(img_z, Quant_Param.scale, Quant_Param.zero_point, Quant_Param.num_bits)
            # Quant_P[f_id] = [Quant_Param.scale, Quant_Param.zero_point, Quant_Param.num_bits]
            # z_q = z_q.to(dtype=torch.uint8).detach().cpu().numpy()
            # z_out = np.ones([128,128],dtype=np.uint8)
            # z_out[:64, :64] = z_q[0, 0, :, :]
            # z_out[:64, 64:] = z_q[0, 1, :, :]
            # z_out[64:, :64] = z_q[0, 2, :, :]
            # z_out[64:, 64:] = z_q[0, 3, :, :]
            # cv2.imwrite('data/uvg_2/VAE/{:05d}.png'.format(f_id), z_out)


            # #Fake Quant
            # img_z = FakeQuantize.apply(img_z, Quant_Param)

            #decoder
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
            ret, z_in = cap.read()
            z_in = z_in[:,:,0]

            # z_in = cv.imread('data/uvg_2/VAE/{:05d}.png'.format(f_id), 0)
            z_q = np.ones([1,4,64,64],dtype=np.uint8)
            z_q[0, 0, :, :] = z_in[:64, :64]
            z_q[0, 1, :, :] = z_in[:64, 64:]
            z_q[0, 2, :, :] = z_in[64:, :64]
            z_q[0, 3, :, :] = z_in[64:, 64:]
            z_q = torch.from_numpy(z_q).to(torch.float32).cuda()
            img_z = dequantize_tensor(z_q, Quant_P[f_id][0], Quant_P[f_id][1])

            samples_x = model.decode_first_stage(img_z)
            samples_x = torch.clamp(samples_x, min=-1.0, max=1.0)

            mse_value, lpips_value = loss_mse(samples_x, img).item(), loss_lpips(samples_x, img).item()
            # print('f_id: {}, lpips:{}, mse:{}'.format(f_id, lpips_value, mse_value))
            print(lpips_value, mse_value)
            mse_list.append(mse_value)
            lpips_list.append(lpips_value)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            if filter is not None:
                samples = filter(samples)
                samples = (
                    (255 * samples)
                    .to(dtype=torch.uint8)
                    .permute(0, 2, 3, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                img = samples[0][:, :, ::-1]
                cv2.imshow('VAE', img)
                cv2.waitKey(1)
                cv2.imwrite('data/cartoon_2/VAE/80k/{:05d}.png'.format(f_id), img)
        print('mean lpips: {}, mse: {}'.format(np.mean(lpips_list), np.mean(mse_list)))
        # torch.save(Quant_P, 'data/uvg_2/VAE/Quant_Parm.pth')


if __name__ == "__main__":
    # version_dict = VERSION2SPECS['SDXL-Turbo']
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)

    n_steps = 1

    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    sampler.n_sample_steps = n_steps
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    seed_ = 88
    while True:
        sampler.noise_sampler = SeededNoise(seed=seed_)
        time_begin = time.time()
        # VAE_encoder(
        #     model, sampler, H=512, W=512, seed=seed_, filter=state.get("filter")
        # )
        # out = sample_sd_c(
        #     model, sampler, H=512, W=512, seed=seed_, prompt=prompt, filter=state.get("filter")
        # )
        # out = sample_sd_interplation(
        #     model, sampler, H=512, W=512, seed=seed_, prompt=prompt, filter=state.get("filter")
        # )
        # out = sample_sd(
        #     model, sampler, H=512, W=512, seed=seed_, prompt=prompt, filter=state.get("filter")
        # )
        # out = sample_inv_sd(
        #     model, sampler, H=512, W=512, seed=seed_, prompt=prompt, filter=state.get("filter")
        # )
        # out = sample_inv_sd_sequence(
        #    model, sampler, H=832, W=832, seed=seed_, filter=state.get("filter")
        # )
        # out = sample_inv_sd_sequence_patch(
        #    model, sampler, H=1024, W=1024, seed=seed_, filter=state.get("filter")
        # )
        # out = sample_inv_sd_sequence_latent(
        #     model, sampler, H=512, W=512, seed=seed_, filter=state.get("filter")
        # )
        # out = sample_inv_sd_sequence_cp(
        #     model, sampler, H=512, W=512, seed=seed_, filter=state.get("filter")
        # )
        # out = sample_inv(
        #     model, sampler, H=512, W=512, seed=seed_, prompt=prompt, filter=state.get("filter")
        # )
        out = sample(
            model, sampler, H=512, W=512, seed=seed_, filter=state.get("filter")
        )
        time_end = time.time()
        print('cost time: ', time_end - time_begin)
        img = out[0][:,:,::-1]
        cv2.imshow('sd', img)
        cv2.waitKey()
        seed_ = seed_ + 1

