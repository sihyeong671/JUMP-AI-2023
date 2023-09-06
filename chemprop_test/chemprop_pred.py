import chemprop
from multiprocessing import freeze_support

arguments = [
    '--test_path', 'data/test_dropna.csv',
    '--preds_path', 'test_result/dacon_preds.csv',
    '--checkpoint_dir', 'train_result'
]

if __name__ == "__main__":
    freeze_support()
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
