import time
import os
import sys
import hydra
import omegaconf
import logging

from concurrent.futures import ProcessPoolExecutor


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


def run_csample(
    general_config_str, csample_config_str, template_name, device_id, log_id
):
    csample_cmd = f"CUDA_VISIBLE_DEVICES={device_id} python main.py task=csample debug_template={template_name} log_id={log_id} {general_config_str} {csample_config_str}"
    logging.warning(csample_cmd)
    os.system(csample_cmd)
    return


def run_mjopt(general_config_str, mjopt_config_str, test_log_path):
    mjopt_cmd = f"python main.py task=mjopt {general_config_str} {mjopt_config_str}"
    logging.warning(mjopt_cmd)
    time.sleep(10)
    while not check_stop(test_log_path):
        os.system(mjopt_cmd)
    return


def run_mjtest(general_config_str, mjtest_config_str, test_log_path):
    mjtest_cmd = f"python main.py task=mjtest {general_config_str} {mjtest_config_str}"
    logging.warning(mjtest_cmd)
    time.sleep(30)
    while not check_stop(test_log_path):
        os.system(mjtest_cmd)
    return


@hydra.main(config_path="config", config_name="base", version_base=None)
def run_together(configs):

    if os.path.exists(configs.succ_dir):
        logging.error(f"The exp_name {configs.exp_name} is already used!")
        exit(1)

    override_config = {
        "general": [],
        "csample": [],
        "mjopt": [],
        "mjtest": [],
    }
    for argv in sys.argv[1:]:
        if "+csample." in argv:
            override_config["csample"].append(argv.replace("+csample.", "task."))
        elif "+mjopt." in argv:
            override_config["mjopt"].append(argv.replace("+mjopt.", "task."))
        elif "+mjtest." in argv:
            override_config["mjtest"].append(argv.replace("+mjtest.", "task."))
        else:
            if "debug_template=" in argv or "log_id=" in argv:
                continue
            override_config["general"].append(argv)

    for k, v in override_config.items():
        if len(v) == 0:
            override_config[k] = ""
        else:
            override_config[k] = "'" + "' '".join(v) + "'"

    # read hand template names
    if configs.debug_template == "**" or configs.debug_template is None:
        hand_template_names = [
            f.split(".yaml")[0] for f in os.listdir(configs.hand.template_path)
        ]
    elif isinstance(configs.debug_template, omegaconf.listconfig.ListConfig):
        hand_template_names = list(configs.debug_template)
    elif isinstance(configs.debug_template, str):
        hand_template_names = [configs.debug_template]
    else:
        raise NotImplementedError(f"Undefined debug_template: {configs.debug_template}")

    assert len(hand_template_names) <= len(
        configs.gpu_list
    ), f"Template number: {len(hand_template_names)}; GPU number: {len(configs.gpu_list)}"

    test_log_path = os.path.join(
        configs.log_dir.replace(configs.task_name, "mjtest"), "main.log"
    )

    # Use ProcessPoolExecutor to run functions in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_mjopt,
                override_config["general"],
                override_config["mjopt"],
                test_log_path,
            ),
            executor.submit(
                run_mjtest,
                override_config["general"],
                override_config["mjtest"],
                test_log_path,
            ),
        ]
        for i, template_name in enumerate(hand_template_names):
            futures.append(
                executor.submit(
                    run_csample,
                    override_config["general"],
                    override_config["csample"],
                    template_name,
                    configs.gpu_list[i],
                    i,
                )
            )

    # Wait for all functions to complete
    for future in futures:
        future.result()

    return


if __name__ == "__main__":
    run_together()
