import os
import re

base_dir = os.path.abspath(os.path.dirname(__file__))

def replace_in_file(file_path, replacements):
    with open(file_path, "r") as f:
        content = f.read()

    original = content
    for old, new in replacements:
        content = re.sub(old, new, content)

    if content != original:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"[✓] Patched: {os.path.relpath(file_path, base_dir)}")
    else:
        print(f"[i] Skipped (no changes): {os.path.relpath(file_path, base_dir)}")


def patch_her_imports():
    target_dir = os.path.join(base_dir, "residual-policy-learning", "tensorflow", "experiment", "configs")
    for file in os.listdir(target_dir):
        if file.endswith(".py"):
            replace_in_file(
                os.path.join(target_dir, file),
                [(r"from baselines\.her\.her import make_sample_her_transitions",
                  r"from baselines.her.her_sampler import make_sample_her_transitions")]
            )


def patch_staging_area():
    files = [
        "ddpg_controller.py",
        "ddpg_controller_residual_base.py",
        "train_staged.py",
        "train_residual_base.py",
    ]
    for f in files:
        path = os.path.join(base_dir, "residual-policy-learning", "tensorflow", "experiment", f)
        replace_in_file(path, [
            (r"from tensorflow\.contrib\.staging import StagingArea",
             r"from tensorflow.python.ops.data_flow_ops import StagingArea")
        ])


def patch_tf_compat():
    tf_pattern = [
        (r"\b(tf\.variable_scope)\b", r"tf.compat.v1.variable_scope"),
        (r"\b(tf\.placeholder)\b", r"tf.compat.v1.placeholder"),
        (r"\b(tf\.InteractiveSession)\b", r"tf.compat.v1.InteractiveSession"),
        (r"\b(tf\.get_default_session)\b", r"tf.compat.v1.get_default_session"),
        (r"\b(tf\.get_collection)\b", r"tf.compat.v1.get_collection"),
        (r"\b(tf\.assign)\b", r"tf.compat.v1.assign"),
        (r"\b(tf\.gradients)\b", r"tf.compat.v1.gradients"),
        (r"\b(tf\.train\.AdamOptimizer)\b", r"tf.compat.v1.train.AdamOptimizer"),
        (r"\b(tf\.variables_initializer)\b", r"tf.compat.v1.variables_initializer"),
        (r"\b(tf\.layers\.dense)\b", r"tf.compat.v1.layers.dense"),
        (r"\b(tf\.contrib\.layers\.xavier_initializer)\b", r"tf.compat.v1.keras.initializers.VarianceScaling\(scale=1.0, mode=\"fan_avg\", distribution=\"uniform\""),
    ]
    target_dir = os.path.join(base_dir, "residual-policy-learning", "tensorflow", "experiment")
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                replace_in_file(path, tf_pattern)


def patch_disable_eager_execution():
    path = os.path.join(base_dir, "residual-policy-learning", "tensorflow", "experiment", "train_staged.py")
    with open(path, "r") as f:
        lines = f.readlines()

    if not any("disable_eager_execution" in line for line in lines):
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("import"):
                insert_index = i
        lines.insert(insert_index + 1, "import tensorflow as tf\n")
        lines.insert(insert_index + 2, "tf.compat.v1.disable_eager_execution()\n\n")
        with open(path, "w") as f:
            f.writelines(lines)
        print("[✓] train_staged.py: disabled eager execution")
    else:
        print("[i] train_staged.py already contains eager execution disabling")


if __name__ == "__main__":
    print("🔧 Applying 0003-.patch manually...")
    patch_her_imports()
    patch_staging_area()
    patch_tf_compat()
    patch_disable_eager_execution()
    print("✅ All patches applied.")
