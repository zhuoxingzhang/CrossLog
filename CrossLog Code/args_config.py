import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration")

    # General setting
    parser.add_argument("--source_dataset", type=lambda s: [item for item in s.split(',')], default=['novel', 'halo'], help="datasets for source")
    parser.add_argument("--target_dataset", type=str, default='forum', help="datasets for target")
    parser.add_argument("--num_finetuning", type=int, default=400, help="graph number of test dataset for fine-tuning")
    parser.add_argument("--ratio_finetuning", type=float, default=0.1, help="ratio of test dataset for fine-tuning")
    parser.add_argument("--dataset_id", type=int, default=51, help="dataset id for test")
    parser.add_argument("--sys_agnostic_feature_dim", type=int, default=1024, help="Feature dimension of the event abstraction")
    parser.add_argument("--sys_specific_feature_dim", type=int, default=1024, help="Feature dimension of the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the dataloader")
    parser.add_argument("--device", type=str, default="cuda", help="Device for experiments")
    parser.add_argument("--num_domain_invar_per_abs", type=int, default=25, help="Number of domain invariants")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size of the GNNs/MLPs")
    parser.add_argument("--abs_coupling", type=float, default=0.6, help="Coupling threshold of event abstraction")

    # Model training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--step_size", type=int, default=5, help="Step size of learning rate reduction")
    parser.add_argument("--gamma", type=float, default=0.8, help="Ratio of learning rate reduction")
    parser.add_argument("--gnn_type", type=str, default="Transformer", help="Type of the GNN (GCN GAT GIN Transformer)")
    parser.add_argument("--gnn_num_head", type=int, default=1, help="Number of heads in transformer-based GNN")
    parser.add_argument("--gnn_num_layers", type=int, default=2, help="Number of layers of the GNN")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for predictions")
    parser.add_argument("--alpha_sys", type=float, default=0.3, help="Ratio for system classification loss")
    parser.add_argument("--alpha_mi", type=float, default=0.3, help="Ratio for mutual information loss")
    parser.add_argument("--mlp_num_layers", type=int, default=4, help="Number of layers of the MLP")

    # Model fine-tuning
    parser.add_argument("--ft_lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ft_num_epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--ft_step_size", type=int, default=10, help="Step size of learning rate reduction")
    parser.add_argument("--ft_gamma", type=float, default=0.8, help="Ratio of learning rate reduction")

    args = parser.parse_args()
    return args
