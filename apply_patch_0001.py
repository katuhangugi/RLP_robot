import os

base_dir = os.path.abspath(os.path.dirname(__file__))

def patch_gitignore():
    path = os.path.join(base_dir, "residual-policy-learning", ".gitignore")
    lines = open(path).read().splitlines()
    additions = [
        "",
        "/internal/",
        "/deps/",
        "**/*.DS_Store",
        "**/logs/",
        "**/logdir/",
        "**/*.egg-info"
    ]
    if not any("internal/" in line for line in lines):
        lines.append("")
        lines.extend(additions)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("[✓] .gitignore patched.")
    else:
        print("[i] .gitignore already patched.")


def patch_requirements():
    path = os.path.join(base_dir, "residual-policy-learning", "requirements.txt")
    content = """numpy==1.23.5
gym==0.13.1
matplotlib==3.9.2
pandas==2.2.3
seaborn==0.13.2
mpi4py==4.0.1
Cython<3
tensorflow[and-cuda]==2.12.0
"""
    with open(path, "w") as f:
        f.write(content)
    print("[✓] requirements.txt replaced.")


def patch_mpi_adam():
    path = os.path.join(base_dir, "residual-policy-learning", "src/baselines/baselines/common/mpi_adam.py")
    with open(path, "r") as f:
        lines = f.readlines()

    insert_index = None
    for i, line in enumerate(lines):
        if "except ImportError:" in line:
            insert_index = i + 1
            break

    if insert_index is not None:
        tf_adam_code = """
MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")
"""
        lines.insert(insert_index + 1, tf_adam_code + "\n")
        with open(path, "w") as f:
            f.writelines(lines)
        print("[✓] mpi_adam.py patched.")
    else:
        print("[×] Cannot find insertion point in mpi_adam.py")


def patch_tf_util():
    path = os.path.join(base_dir, "residual-policy-learning", "src/baselines/baselines/common/tf_util.py")
    with open(path, "r") as f:
        content = f.read()

    # Remove ".numpy()" to make TF compatible with session
    content = content.replace(".numpy()", "")
    with open(path, "w") as f:
        f.write(content)
    print("[✓] tf_util.py patched.")


def patch_normalizer():
    path = os.path.join(base_dir, "residual-policy-learning", "src/baselines/baselines/her/normalizer.py")
    with open(path, "r") as f:
        lines = f.readlines()

    # Replace import
    for i, line in enumerate(lines):
        if "import tensorflow as tf" in line:
            lines[i] = "import tensorflow.compat.v1 as tf\n"

    # Add sess param
    for i, line in enumerate(lines):
        if "def __init__(self, size" in line and "sess=None" not in line:
            lines[i] = line.strip().rstrip("):") + ", sess=None):\n"

    # Add sess assignment
    for i, line in enumerate(lines):
        if "self.default_clip_range = default_clip_range" in line:
            lines.insert(i + 1, "        self.sess = sess if sess is not None else tf.compat.v1.get_default_session()\n")
            break

    with open(path, "w") as f:
        f.writelines(lines)
    print("[✓] normalizer.py patched.")


if __name__ == "__main__":
    print("🔧 Applying 0001-baselines.patch manually...")
    patch_gitignore()
    patch_requirements()
    patch_mpi_adam()
    patch_tf_util()
    patch_normalizer()
    print("✅ All patches applied.")
