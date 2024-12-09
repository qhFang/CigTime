# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from mdm.utils.fixseed import fixseed
import os
import numpy as np
import torch
from mdm.utils.parser_util import edit_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm.data_loaders import humanml_utils
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from omegaconf import OmegaConf
from tqdm import tqdm
import joblib

def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')
    else:
        out_path = os.path.join(os.path.dirname(args.model_path),
                                out_path)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    #args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    args.batch_size = 1024


    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='edit',
                              drop_last=False,
                              shuffle=False,
                              edit_file=args.edit_file)  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    #total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    #iterator = iter(data)
    all_motions = []
    all_input_motions = []
    all_lengths = []
    all_text = []
    

    for i, (input_motions, model_kwargs) in tqdm(enumerate(data)):
        input_motions = input_motions.to(dist_util.dev())

        body_parts = []
        
        if max_frames != input_motions.shape[-1]:
            input_motions = input_motions[:, :, :, :max_frames]
            
        gt_frames_per_sample = {}
        model_kwargs['y']['inpainted_motion'] = input_motions
        model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
                                                            device=input_motions.device)  # True is lower body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])


        for j, text in enumerate(model_kwargs['y']['text']):
            if text.startswith('The person'):
                model_kwargs['y']['text'][j] = 'A' + text[3:] 
            if 'upper body' in text.lower():
                model_kwargs['y']['inpainting_mask'][j] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
                                                            device=input_motions.device).unsqueeze(-1).unsqueeze(-1).repeat(1, input_motions.shape[2], input_motions.shape[3])
                body_parts.append('upper_body')
            elif 'lower body' in text.lower():
                model_kwargs['y']['inpainting_mask'][j] = torch.tensor(humanml_utils.HML_UPPER_BODY_MASK, dtype=torch.bool,
                                                            device=input_motions.device).unsqueeze(-1).unsqueeze(-1).repeat(1, input_motions.shape[2], input_motions.shape[3])
                body_parts.append('lower_body')
            else: 
                model_kwargs['y']['inpainting_mask'][j] = torch.zeros_like(model_kwargs['y']['inpainting_mask'][j])

        model_kwargs['y']['ori_text'] = model_kwargs['y']['text']


        for rep_i in range(1):
            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(input_motions.shape[0], device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (input_motions.shape[0], model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )


            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                all_motions.append(sample.cpu().numpy())
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_text += model_kwargs['y']['ori_text']
            all_input_motions.append(input_motions.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            #print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = (np.concatenate(all_motions, axis=0))
    all_input_motions = (np.concatenate(all_input_motions, axis=0))
    all_lengths = np.concatenate(all_lengths, axis=0)


    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    pkl_path = os.path.join(out_path, 'results.pkl')
    print(f"saving results file to [{pkl_path}]")
    
    joblib.dump({'input_motion': data.dataset.t2m_dataset.inv_transform(torch.from_numpy(all_input_motions).permute(0, 2, 3, 1)).float(), 'motion': all_motions, 'text': all_text, 'lengths': all_lengths}, pkl_path)
    with open(pkl_path.replace('.pkl', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(pkl_path.replace('.pkl', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(torch.from_numpy(all_input_motions).permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

        output_motions = recover_from_ric(torch.from_numpy(all_motions), n_joints)
        output_motions = output_motions.view(-1, *output_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    for sample_i in range(100):
        caption = 'Input Motion'
        length = model_kwargs['y']['lengths'][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = 'input_motion{:02d}.gif'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                    dataset=args.dataset, fps=fps, vis_mode='gt',
                    gt_frames=gt_frames_per_sample.get(sample_i, []))
        #for rep_i in range(args.num_repetitions):
        caption = all_text[rep_i*args.batch_size + sample_i]
        if caption == '':
            caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
        else:
            caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
        length = all_lengths[sample_i]
        motion = output_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = 'sample{:02d}_rep{:02d}.gif'.format(sample_i, rep_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode=body_parts[sample_i],
                            gt_frames=gt_frames_per_sample.get(sample_i, []))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.gif'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
