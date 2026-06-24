from args_config import get_args
from finetune_and_anomaly_infer import finetune_and_infer
from train import train

combo = [('halo', 'forum', 'novel'), ('halo', 'novel', 'forum'), ('novel', 'forum', 'halo')]
num_ratio_finetuning = [(500, 0.2), (1000, 0.3), (2000, 0.4), (3000, 0.5), (4000, 0.6)]
args = get_args()
for source_sys0, source_sys1, target_sys in combo:
    '''Train'''
    args.source_dataset = [source_sys0, source_sys1]
    args.target_dataset = target_sys
    train(args)
    for num_finetuning, ratio_finetuning in num_ratio_finetuning:
        '''Finetune and Infer'''
        args.num_finetuning = num_finetuning
        args.ratio_finetuning = ratio_finetuning
        finetune_and_infer(args)
