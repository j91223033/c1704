from header import *
from dataset import load_dataset
from model import *
from config import *
import torch.distributed as dist


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--scannet_file', type=str, help='pkl file containing the data of Scannet'
                                                                       ' as generated by running XXX')
    parser.add_argument('--referit3D_file', type=str)
    parser.add_argument('--alpaca_path', type=str)
    parser.add_argument('--model', type=str, default='nextgpt')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/')
    parser.add_argument('--log_path', type=str, default='../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--dataset', type=str, default='sr3d')
    
    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--stage', type=int, default=1)  # the training stage
    parser.add_argument('--modality', type=list, default=['image', 'video', 'audio', 'text'])
    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = 6001 #os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    init_method = f'tcp://{args["master_ip"]}:{args["master_port"]}'
    print(f"Initialized DeepSpeed with master IP {args['master_ip']}, port {args['master_port']}, rank {args['local_rank']}, and world size {args['world_size']}")
    deepspeed.init_distributed(dist_backend='nccl', init_method=init_method, rank=args['local_rank'], world_size=args['world_size'])    
    #deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args['root_dir'] = '../'
    # args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)


def main(**args):
    config_env(args)
    print(args)
    args['ds_config_path'] = f'dsconfig/stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data, train_iter, sampler, train_data_alpaca, train_iter_alpaca, sampler_alpaca = load_dataset(args)

    train_num = train_data.__len__() # 65844
    length = args['epochs'] * train_num // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu'] # 1 * 65844 // 4 // 4 == 4415
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size'] # 1 * 65844 // 16 == 4415
    args['total_steps'] = total_steps # 4415
    agent = load_model(args)
    torch.distributed.barrier()
    # begin to train
    pbar = tqdm(total=length) # maximum total number 4415
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        # for train_iter in train_iter_list:
        train_iter2_iter = iter(train_iter_alpaca)
        for i, batch in enumerate(train_iter):
            batch2 = next(train_iter2_iter, None)
            agent.train_model(
                {"dataset1":batch, "dataset2":batch2},
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1
            if current_step % 2000 == 0:
                torch.distributed.barrier()
                agent.save_model(args['save_path'], current_step)
            
    # save at the end of the training
    torch.distributed.barrier()
    agent.save_model(args['save_path'], current_step)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
