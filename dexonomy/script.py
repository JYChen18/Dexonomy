import time
import os
import sys
import hydra
import logging
import subprocess

from concurrent.futures import ProcessPoolExecutor
from dexonomy.util.file_util import get_template_name_lst


def check_finish_init(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_info = f.read()
        if "Finish op init" in log_info:
            logging.info(f"op init has finished! Recorded in {log_path}")
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


def run_init(
    general_config_str,
    init_config_str,
    template_name,
    device_id,
    log_id,
    obj_log_path,
):
    init_cmd = f"CUDA_VISIBLE_DEVICES={device_id} python -m dexonomy.main op=init template_name={template_name} log_id={log_id} {general_config_str} {init_config_str}"
    logging.info(init_cmd)
    count = 0
    while not check_finish_init(obj_log_path) and count < 10:
        os.system(init_cmd)
        count += 1
    return


def run_grasp(general_config_str, grasp_config_str, hand_log_path):
    grasp_cmd = (
        f"python -m dexonomy.main op=grasp {general_config_str} {grasp_config_str}"
    )
    logging.info(grasp_cmd)
    while not check_finish_grasp(hand_log_path):
        os.system(grasp_cmd)
    return


def run_eval(general_config_str, eval_config_str, log_id, eval_log_path):
    eval_cmd = f"python -m dexonomy.main op=eval log_id={log_id} {general_config_str} {eval_config_str}"
    logging.info(eval_cmd)
    time.sleep(5)
    while not check_finish_eval(eval_log_path):
        os.system(eval_cmd)
    return


def run_straj(grasp_dir, traj_dir, hand_name, gpu_lst, eval_log_path):
    straj_cmd = f"cd third_party/BODex && DISABLE_GRASP_SYN=true python example_grasp/multi_gpu.py \
    -t mogen_dexonomy \
    -c {hand_name}/dexonomy.yml \
    -g {' '.join([str(g) for g in gpu_lst])} \
    -e {os.path.abspath(traj_dir)} \
    -i '{os.path.abspath(grasp_dir)}/**/*.npy' "

    logging.info(straj_cmd)
    while not check_finish_eval(eval_log_path):
        os.system(straj_cmd)
    return


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(configs):

    if os.path.exists(configs.succ_grasp_dir) or os.path.exists(configs.succ_traj_dir):
        logging.error(f"The exp_name {configs.exp_name} is already used!")
        exit(1)

    override_config = {
        "general": [],
        "init": [],
        "grasp": [],
        "eval": [],
    }
    for argv in sys.argv[1:]:
        if "+init." in argv:
            override_config["init"].append(argv.replace("+init.", "op."))
        elif "+grasp." in argv:
            override_config["grasp"].append(argv.replace("+grasp.", "op."))
        elif "+eval." in argv:
            override_config["eval"].append(argv.replace("+eval.", "op."))
        else:
            if "template_name=" in argv or "log_id=" in argv:
                continue
            override_config["general"].append(argv)

    if not configs.skip_eval_grasp:
        override_config["general_no_arm"] = []
        for v in override_config["general"]:
            if "adding_arm" not in v:
                override_config["general_no_arm"].append(v)
        override_config["general_no_arm"].append("adding_arm=False")

    for k, v in override_config.items():
        if len(v) == 0:
            override_config[k] = ""
        else:
            override_config[k] = "'" + "' '".join(v) + "'"

    # read hand template names
    hand_template_names = get_template_name_lst(
        configs.template_name, configs.init_template_dir
    )
    assert len(hand_template_names) <= len(
        configs.init_gpu
    ), f"Template number: {len(hand_template_names)}; GPU number: {len(configs.init_gpu)}"
    assert (
        len(set(configs.init_gpu).intersection(set(configs.straj_gpu))) == 0
    ), f"init GPU should be different with straj GPU! init: {configs.init_gpu}; straj: {configs.straj_gpu}"

    grasp_eval_log_path = os.path.join(
        os.path.dirname(configs.log_dir), "eval_0", "main.log"
    )
    traj_eval_log_path = os.path.join(
        os.path.dirname(configs.log_dir), "eval_1", "main.log"
    )
    grasp_log_path = os.path.join(
        os.path.dirname(configs.log_dir), "grasp_0", "main.log"
    )

    # Check dependency
    if configs.adding_arm:
        if (
            subprocess.run(
                "pip show nvidia_curobo_bodex", shell=True, capture_output=True
            ).returncode
            != 0
        ):
            logging.error(
                "Please install the third-party library BODex for trajectory synthesis!"
            )
            exit(1)

    # Use ProcessPoolExecutor to run functions in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, template_name in enumerate(hand_template_names):
            obj_log_path = os.path.join(
                os.path.dirname(configs.log_dir), f"init_{i}", "main.log"
            )
            futures.append(
                executor.submit(
                    run_init,
                    override_config["general"],
                    override_config["init"],
                    template_name,
                    configs.init_gpu[i],
                    i,
                    obj_log_path,
                )
            )
        futures.append(
            executor.submit(
                run_grasp,
                override_config["general"],
                override_config["grasp"],
                grasp_log_path,
            )
        )
        if not configs.skip_eval_grasp:
            futures.append(
                executor.submit(
                    run_eval,
                    override_config["general_no_arm"],
                    override_config["eval"],
                    0,
                    grasp_eval_log_path,
                )
            )
        if configs.adding_arm:
            futures.append(
                executor.submit(
                    run_straj,
                    (
                        configs.grasp_dir
                        if configs.skip_eval_grasp
                        else configs.succ_grasp_dir
                    ),
                    configs.traj_dir,
                    configs.hand_name,
                    configs.straj_gpu,
                    traj_eval_log_path,
                )
            )
            futures.append(
                executor.submit(
                    run_eval,
                    override_config["general"],
                    override_config["eval"],
                    1,
                    traj_eval_log_path,
                )
            )

    # Wait for all functions to complete
    for future in futures:
        future.result()

    return


if __name__ == "__main__":
    main()
