from omegaconf import OmegaConf
from tqdm import tqdm
import csv
import os
from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check
import numpy as np

def main(args_cli):
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    # sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    if args.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {args.output_path}')
    print('Device:', args.device)

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == 'i3d':
        from models.i3d.extract_i3d import ExtractI3D as Extractor
    elif args.feature_type == 'r21d':
        from models.r21d.extract_r21d import ExtractR21D as Extractor
    elif args.feature_type == 's3d':
        from models.s3d.extract_s3d import ExtractS3D as Extractor
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish as Extractor
    elif args.feature_type == 'resnet':
        from models.resnet.extract_resnet import ExtractResNet as Extractor
    elif args.feature_type == 'raft':
        from models.raft.extract_raft import ExtractRAFT as Extractor
    elif args.feature_type == 'pwc':
        from models.pwc.extract_pwc import ExtractPWC as Extractor
    elif args.feature_type == 'clip':
        from models.clip.extract_clip import ExtractCLIP as Extractor
    else:
        raise NotImplementedError(f'Extractor {args.feature_type} is not implemented.')

    # unifies whatever a user specified as paths into a list of paths
    data_dir = "../dlcv-final-problem1-talking-to-me/student_data/student_data"
    extractor = Extractor(args, data_dir)

    seg_dir = os.path.join(data_dir, 'train', 'seg')
    video_ids = sorted([x.split('_')[0] for x in os.listdir(seg_dir)])
    seg = []
    
    for video_id in video_ids:
        seg_path = os.path.join(seg_dir, video_id + '_seg.csv')
        with open(seg_path, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                row['video_id'] = video_id
                seg.append(row)

    # print(seg[11136])
    # feature = extractor.extract(seg[0])
    # print(feature['rgb'].shape)
    # print(feature['rgb'].shape)

    for segment in tqdm(seg[11136:]):
        feature = extractor.extract(segment)
        save_path = os.path.join(data_dir, "i3d", segment['video_id'] + '_' + segment['person_id'] + '_' + segment['start_frame'] + '_' + segment['end_frame'] + '.npy')
        np.save(save_path, feature['rgb'])
    # for video_path in tqdm(video_paths):
    #     extractor._extract(video_path)  # note the `_` in the method name

    # yep, it is this simple!


if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    main(args_cli)
