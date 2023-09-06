import chemprop
from chemprop.hyperparameter_optimization import hyperopt

arguments = [
    '--smiles_column', "SMILES",
    '--target_columns', "HLM", "MLM",
    '--data_path', 'data/train_dropna.csv',
    '--dataset_type', 'regression',
    '--metric', 'rmse',
    '--epochs', '50',
    '--save_dir', 'train_result',
    '--split_sizes', '0.8', '0.1' ,'0.1',
    '--ensemble_size', '2',
    # '--config_save_path', 'train_result/config_v0.json',
    '--seed', '77',
    '--quiet',
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
