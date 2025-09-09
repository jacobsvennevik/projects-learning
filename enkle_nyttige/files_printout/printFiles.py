import os

def load_gitignore(gitignore_path):
    """
    Reads a .gitignore file and returns a set of ignored file patterns.
    """
    ignored_files = set()
    ignored_dirs = set()
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore comments
                    if line.endswith("/"):  
                        ignored_dirs.add(line.rstrip("/"))  # Remove trailing slash
                    else:
                        ignored_files.add(line)

    return ignored_files, ignored_dirs


def write_directory_structure(root_dir, output_file, gitignore_path=".gitignore"):
    """
    Writes the directory structure to a file, excluding files and directories from .gitignore.
    """
    exclude_files, exclude_dirs = load_gitignore(gitignore_path)

    with open(output_file, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Filter out ignored directories
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            level = dirpath.replace(root_dir, "").count(os.sep)
            indent = "   " * level
            f.write(f"{indent}📁 {os.path.basename(dirpath)}/\n")
            for filename in filenames:
                if filename in exclude_files:
                    continue
                f.write(f"{indent}   📄 {filename}\n")

if __name__ == "__main__":
    project_dir = "/Users/jacobhornsvennevik/Documents/GitHub/MaRepo_root/frontend"
    output_file = "directory_structure.txt"
    write_directory_structure(project_dir, output_file)
    print(f"Directory structure written to {output_file}, using .gitignore rules.")
