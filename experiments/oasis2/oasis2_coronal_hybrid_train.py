from oasis2_coronal_experiment import ExperimentConfig, compare_models, parse_args, run_experiment

import torch


def main():
    args = parse_args()
    config = ExperimentConfig(
        model_name="hybrid",
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        trials=args.trials,
        seeds=args.seeds,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        device=torch.device(args.device),
    )
    run_experiment(config)
    compare_models(args.output_dir)


if __name__ == "__main__":
    main()
