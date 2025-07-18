import os
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    os.system("ps aux | egrep 'dexrun|dexsyn' > /tmp/debug.txt")

    with open("/tmp/debug.txt", "r") as f:
        x = f.readlines()

    x = [line for line in x if cfg.debug_name in line]
    print("\n".join(x))

    flag = input("Press 'c' to kill all the above processes\n")
    if flag != "c":
        exit(1)

    for line in x:
        parts = [part for part in line.split(" ") if part]
        pid = parts[1]
        print(f"kill -9 {pid}")
        os.system(f"kill -9 {pid}")


if __name__ == "__main__":
    main()
