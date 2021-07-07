from ppo import PPO
from settings import *
import torch.multiprocessing as mp


def main():
    BipedalWalkerPPO = PPO(per_gpu_workers, gpus, lam, gamma, epsilon, n_episodes)
    BipedalWalkerPPO.train(iterations, ppo_epochs, batch_size, n_batch)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
