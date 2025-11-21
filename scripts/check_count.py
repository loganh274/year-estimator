import os

base_dir = "data/raw/faces_aligned_small_mirrored_co_aligned_cropped_cleaned"
if not os.path.exists(base_dir):
    print("Base directory does not exist yet.")
else:
    f_count = len(os.listdir(os.path.join(base_dir, "F"))) if os.path.exists(os.path.join(base_dir, "F")) else 0
    m_count = len(os.listdir(os.path.join(base_dir, "M"))) if os.path.exists(os.path.join(base_dir, "M")) else 0
    print(f"F: {f_count}")
    print(f"M: {m_count}")
    print(f"Total: {f_count + m_count}")
