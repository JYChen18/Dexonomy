import time
import os
import sys
import hydra
import logging
import subprocess

from concurrent.futures import ProcessPoolExecutor
from dexonomy.util.file_util import get_template_name_lst


def check_stop(test_log_path):
    if os.path.exists(test_log_path):
        with open(test_log_path, "r") as f:
            log_info = f.readlines()
        if len(log_info) < 11:
            return False
        for i in range(1, 11):
            if "Find 0 graspdata" not in log_info[-i]:
                return False
        return True
    return False


def check_finish_synobj(obj_log_path):
    if os.path.exists(obj_log_path):
        with open(obj_log_path, "r") as f:
            log_info = f.read()
        if "Finish task syn_obj" in log_info:
            logging.info(
                f"The task of syn_obj has finished! Recorded in {obj_log_path}"
            )
            return True
    return False


def run_syn_obj(
    general_config_str,
    syn_obj_config_str,
    template_name,
    device_id,
    log_id,
    obj_log_path,
):
    syn_obj_cmd = f"CUDA_VISIBLE_DEVICES={device_id} python -m dexonomy.main task=syn_obj template_name={template_name} log_id={log_id} {general_config_str} {syn_obj_config_str}"
    logging.info(syn_obj_cmd)
    count = 0
    while not check_finish_synobj(obj_log_path) and count < 10:
        os.system(syn_obj_cmd)
        count += 1
    return


def run_syn_hand(general_config_str, syn_hand_config_str, test_log_path):
    syn_hand_cmd = f"python -m dexonomy.main task=syn_hand {general_config_str} {syn_hand_config_str}"
    logging.info(syn_hand_cmd)
    while not check_stop(test_log_path):
        os.system(syn_hand_cmd)
    return


def run_mogen(save_dir, grasp_dir, hand_name, gpu_lst, test_log_path):
    mogen_cmd = f"cd third_party/BODex && DISABLE_GRASP_SYN=true python example_grasp/multi_gpu.py \
    -t mogen_dexonomy \
    -c {hand_name}/dexonomy.yml \
    -g {' '.join([str(g) for g in gpu_lst])} \
    -e {os.path.abspath(save_dir)} \
    -i '{os.path.abspath(grasp_dir)}/**/*.npy' "

    logging.info(mogen_cmd)
    while not check_stop(test_log_path):
        os.system(mogen_cmd)
    return


def run_syn_test(general_config_str, syn_test_config_str, test_log_path):
    syn_test_cmd = f"python -m dexonomy.main task=syn_test {general_config_str} {syn_test_config_str}"
    logging.info(syn_test_cmd)
    time.sleep(5)
    while not check_stop(test_log_path):
        os.system(syn_test_cmd)
    return


@hydra.main(config_path="config", config_name="base", version_base=None)
def run_together(configs):

    if os.path.exists(configs.succ_dir):
        logging.error(f"The exp_name {configs.exp_name} is already used!")
        exit(1)

    override_config = {
        "general": [],
        "syn_obj": [],
        "syn_hand": [],
        "syn_test": [],
    }
    for argv in sys.argv[1:]:
        if "+syn_obj." in argv:
            override_config["syn_obj"].append(argv.replace("+syn_obj.", "task."))
        elif "+syn_hand." in argv:
            override_config["syn_hand"].append(argv.replace("+syn_hand.", "task."))
        elif "+syn_test." in argv:
            override_config["syn_test"].append(argv.replace("+syn_test.", "task."))
        else:
            if "template_name=" in argv or "log_id=" in argv:
                continue
            override_config["general"].append(argv)

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
        configs.syn_obj_gpu
    ), f"Template number: {len(hand_template_names)}; GPU number: {len(configs.syn_obj_gpu)}"
    assert (
        len(set(configs.syn_obj_gpu).intersection(set(configs.mogen_gpu))) == 0
    ), f"Syn_obj GPU should be different with mogen GPU! Syn_obj: {configs.syn_obj_gpu}; Mogen: {configs.mogen_gpu}"

    test_log_path = os.path.join(
        configs.log_dir.replace(configs.task_name, "syn_test"), "main.log"
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
        futures = [
            executor.submit(
                run_syn_hand,
                override_config["general"],
                override_config["syn_hand"],
                test_log_path,
            ),
            executor.submit(
                run_syn_test,
                override_config["general"],
                override_config["syn_test"],
                test_log_path,
            ),
        ]
        for i, template_name in enumerate(hand_template_names):
            obj_log_path = os.path.join(
                os.path.dirname(configs.log_dir), f"syn_obj_{i}", "main.log"
            )
            futures.append(
                executor.submit(
                    run_syn_obj,
                    override_config["general"],
                    override_config["syn_obj"],
                    template_name,
                    configs.syn_obj_gpu[i],
                    i,
                    obj_log_path,
                )
            )
        if configs.adding_arm:
            futures.append(
                executor.submit(
                    run_mogen,
                    configs.save_dir,
                    configs.grasp_dir,
                    configs.hand_name,
                    configs.mogen_gpu,
                    test_log_path,
                )
            )

    # Wait for all functions to complete
    for future in futures:
        future.result()

    return


if __name__ == "__main__":
    run_together()
