import os

base_dir = os.path.abspath(os.path.dirname(__file__))

def patch_init_py():
    path = os.path.join(base_dir, "residual-policy-learning", "rpl_environments", "rpl_environments", "__init__.py")
    with open(path, "r") as f:
        lines = f.readlines()

    changed = False
    for i, line in enumerate(lines):
        if "timestep_limit" in line:
            lines[i] = line.replace("timestep_limit", "max_episode_steps")
            changed = True

    if changed:
        with open(path, "w") as f:
            f.writelines(lines)
        print("[✓] __init__.py patched: replaced 'timestep_limit' with 'max_episode_steps'")
    else:
        print("[i] __init__.py already patched or no changes needed")


def patch_complex_hook_env():
    path = os.path.join(base_dir, "residual-policy-learning", "rpl_environments", "rpl_environments", "envs", "complex_hook_env.py")
    with open(path, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if "except:" in lines[i]:
            lines[i] = "            except Exception as e:\n"
            lines[i+1] = "                print(\"FAILED; skipping\", e)\n"
            print("[✓] complex_hook_env.py patched: expanded bare except to Exception as e")
            break
    else:
        print("[i] complex_hook_env.py already patched or no changes found")

    with open(path, "w") as f:
        f.writelines(lines)


def patch_push_database_controller():
    path = os.path.join(base_dir, "residual-policy-learning", "rpl_environments", "rpl_environments", "envs", "push_database_controller.py")
    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "np.load(database_file)" in line:
            lines[i] = "    states, actions = np.load(database_file, allow_pickle=True)\n"
            print("[✓] push_database_controller.py patched: added allow_pickle=True")
            break
    else:
        print("[i] push_database_controller.py already patched or no changes found")

    with open(path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    print("🔧 Applying 0002-rpl.patch manually...")
    patch_init_py()
    patch_complex_hook_env()
    patch_push_database_controller()
    print("✅ All patches applied.")
