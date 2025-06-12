import mujoco
import os

root_folder = "../../third_party/mujoco_menagerie"
arm_xml_path = "universal_robots_ur10e/ur10e.xml"
hand_xml_path = "shadow_hand/right_with_forearm.xml"

arm_spec = mujoco.MjSpec.from_file(os.path.join(root_folder, arm_xml_path))
arm_model = arm_spec.compile()

hand_spec = mujoco.MjSpec.from_file("shadow/real_with_forearm.xml")
hand_model = hand_spec.compile()

combined_spec = arm_spec
attachment_site = next(s for s in combined_spec.sites if s.name == "attachment_site")
attachment_site.attach_body(hand_spec.worldbody, "hand:", "")

for m in combined_spec.meshes:
    if "hand:" in m.name:
        m.file = os.path.abspath(
            os.path.join(
                root_folder, os.path.dirname(hand_xml_path), hand_spec.meshdir, m.file
            )
        )
    else:
        m.file = os.path.abspath(
            os.path.join(
                root_folder, os.path.dirname(arm_xml_path), arm_spec.meshdir, m.file
            )
        )
print(combined_spec.meshdir)
combined_spec.meshdir = os.path.join("..", root_folder)
print(combined_spec.meshdir)
combined_model = combined_spec.compile()

with open("shadow/real_right_ur10e.xml", "w") as f:
    f.write(combined_spec.to_xml())
