from ppo import PPO
from settings import *


def main():
    BipedalWalkerPPO = PPO(per_gpu_workers, gpus, lam, gamma, epsilon)
    BipedalWalkerPPO.train(iterations, ppo_epochs, batch_size, n_batch, n_episodes)


if __name__ == "__main__":
    main()
