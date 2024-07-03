from parti_pytorch import VitVQGanVAE, VQGanVAETrainer, VQGanVAETrainerMGPU
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--data_dir", type=str, default=None, help="the dir of data.")
    parser.add_argument("--patch_size", type=str, default="16", help="the patch size of image.")
    parser.add_argument("--num_layers", type=int, default=4, help="the number of layers.")
    parser.add_argument("--vq_codebook_dim", type=int, default=64, help="the dim of vq codebook.")
    parser.add_argument("--vq_codebook_size", type=int, default=512, help="the size of vq codebook.")
    parser.add_argument("--dim", type=int, default=256, help="the dim of model.")
    parser.add_argument("--image_size", type=int, default=512, help="the size of image.")
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size of model.")
    parser.add_argument("--local_rank", type=int, default=0, help="the rank of gpu.")
    parser.add_argument("--grad_accum_every", type=int, default=1, help="the grad_accum_every of model.")
    args = parser.parse_args()

    # conver str to list
    args.patch_size = eval(args.patch_size)

    vae_config = {"dim": args.dim, "image_size": args.image_size, "patch_size": args.patch_size, "num_layers": args.num_layers, 
                  "vq_codebook_dim":args.vq_codebook_dim, "vq_codebook_size": args.vq_codebook_size}
    
    if args.output_dir is None:
        raise ValueError("output_dir must be specified")
    if args.data_dir is None:
        raise ValueError("data_dir must be specified")
    
    trainer = VQGanVAETrainerMGPU(
        vae_config,
        folder = args.data_dir,
        results_folder = args.output_dir,
        num_train_steps = 500000,
        lr = 3e-4,
        batch_size = args.batch_size,
        grad_accum_every = args.grad_accum_every,
        save_results_every = 1000,
        amp = True
    )

    trainer.train()