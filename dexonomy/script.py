import time
import os
import sys
import hydra
import logging
import subprocess
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor

from dexonomy.util.file_util import get_template_names


def check_finish_init(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_info = f.read()
        if "Finish initialization" in log_info:
            logging.info(f"Initialization has finished! Recorded in {log_path}")
            return True
    return False


def check_finish_grasp(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_info = f.readlines()
        if len(log_info) < 11:
            return False
        for i in range(1, 11):
            if "Find 0 initialization" not in log_info[-i]:
                return False
        return True
    return False


def check_finish_eval(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_info = f.readlines()
        if len(log_info) < 11:
            return False
        for i in range(1, 11):
            if "Find 0 grasp data" not in log_info[-i]:
                return False
        return True
    return False


def run_init(global_cfg_str, init_cfg_str, tmpl_name, device_id, log_id, log_path):
    if "init_gpu=" in global_cfg_str:
        prefix, suffix = global_cfg_str.split("init_gpu=", 1)  # Split at "init_gpu="
        new_global_cfg_str = f"{prefix}init_gpu=[{device_id}]{suffix.split(']')[1]}"
    else:
        new_global_cfg_str = global_cfg_str + f" 'init_gpu=[{device_id}]'"
    init_cmd = f"dexrun op=init tmpl_name={tmpl_name} log_id={log_id} {new_global_cfg_str} {init_cfg_str}"
    logging.info(init_cmd)
    count = 0
    while not check_finish_init(log_path) and count < 10:
        os.system(init_cmd)
        count += 1
    return


def run_grasp(global_cfg_str, grasp_cfg_str, hand_log_path):
    grasp_cmd = f"dexrun op=grasp {global_cfg_str} {grasp_cfg_str}"
    logging.info(grasp_cmd)
    while not check_finish_grasp(hand_log_path):
        os.system(grasp_cmd)
    return


def run_eval(global_cfg_str, eval_cfg_str, log_id, eval_log_path):
    eval_cmd = f"dexrun op=eval log_id={log_id} {global_cfg_str} {eval_cfg_str}"
    logging.info(eval_cmd)
    time.sleep(5)
    while not check_finish_eval(eval_log_path):
        os.system(eval_cmd)
    return


def run_traj(global_cfg_str, traj_cfg_str, eval_log_path):
    traj_cmd = f"dexrun op=traj {global_cfg_str} {traj_cfg_str}"
    logging.info(traj_cmd)
    while not check_finish_eval(eval_log_path):
        os.system(traj_cmd)
    return


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    cmd_cfg = {"global": [], "init": [], "grasp": [], "eval": [], "traj": []}
    for argv in sys.argv[1:]:
        if "+init." in argv:
            cmd_cfg["init"].append(argv.replace("+init.", "op."))
        elif "+grasp." in argv:
            cmd_cfg["grasp"].append(argv.replace("+grasp.", "op."))
        elif "+traj." in argv:
            cmd_cfg["traj"].append(argv.replace("+traj.", "op."))
        elif "+eval." in argv:
            cmd_cfg["eval"].append(argv.replace("+eval.", "op."))
        else:
            if "tmpl_name=" in argv or "log_id=" in argv:
                continue
            cmd_cfg["global"].append(argv)

    if not cfg.skip_grasp_eval:
        cmd_cfg["global_wo_arm"] = []
        for v in cmd_cfg["global"]:
            if "skip_traj" not in v:
                cmd_cfg["global_wo_arm"].append(v)
        cmd_cfg["global_wo_arm"].append("skip_traj=True")

    for k, v in cmd_cfg.items():
        if len(v) == 0:
            cmd_cfg[k] = ""
        else:
            cmd_cfg[k] = "'" + "' '".join(v) + "'"

    # read hand template names
    tmpl_names = get_template_names(cfg.tmpl_name, cfg.init_tmpl_dir)
    n_tmpl, n_gpu = len(tmpl_names), len(cfg.init_gpu)
    assert n_tmpl <= n_gpu, f"Template number: {n_tmpl}; GPU number: {n_gpu}"
    if not cfg.skip_traj:
        assert (
            len(set(cfg.init_gpu).intersection(set(cfg.traj_gpu))) == 0
        ), f"init_gpu should be different with traj_gpu! init: {cfg.init_gpu}; traj: {cfg.traj_gpu}"

    log_dir_dir = os.path.dirname(cfg.log_dir)
    grasp_eval_log = os.path.join(log_dir_dir, "eval_0", "main.log")
    traj_eval_log = os.path.join(log_dir_dir, "eval_1", "main.log")
    grasp_log = os.path.join(log_dir_dir, "grasp_0", "main.log")

    # Check dependency
    if not cfg.skip_traj:
        check_bodex = subprocess.run(
            "pip show nvidia_curobo_bodex", shell=True, capture_output=True
        )
        if check_bodex.returncode != 0:
            logging.error(
                "For trajectory synthesis, please install the third-party library BODex!"
            )
            exit(1)

    # Use ProcessPoolExecutor to run functions in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, tn in enumerate(tmpl_names):
            init_log = os.path.join(
                os.path.dirname(cfg.log_dir), f"init_{i}", "main.log"
            )
            futures.append(
                executor.submit(
                    run_init,
                    cmd_cfg["global"],
                    cmd_cfg["init"],
                    tn,
                    cfg.init_gpu[i],
                    i,
                    init_log,
                )
            )
        futures.append(
            executor.submit(run_grasp, cmd_cfg["global"], cmd_cfg["grasp"], grasp_log)
        )
        if not cfg.skip_grasp_eval:
            futures.append(
                executor.submit(
                    run_eval,
                    cmd_cfg["global_wo_arm"],
                    cmd_cfg["eval"],
                    0,
                    grasp_eval_log,
                )
            )
        if not cfg.skip_traj:
            futures.append(
                executor.submit(
                    run_traj, cmd_cfg["global"], cmd_cfg["traj"], traj_eval_log
                )
            )
            futures.append(
                executor.submit(
                    run_eval, cmd_cfg["global"], cmd_cfg["eval"], 1, traj_eval_log
                )
            )

    # Wait for all functions to complete
    for future in futures:
        future.result()

    return


if __name__ == "__main__":
    main()
