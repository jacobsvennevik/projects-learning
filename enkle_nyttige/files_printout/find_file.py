#!/usr/bin/env python3
def find_file_in_structure(structure_file, target_filename):
    with open(structure_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Stack to hold tuples of (indentation_level, directory_name)
    stack = []

    for line in lines:
        # Count the number of leading spaces to determine the level
        level = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if not stripped:
            continue

        # If it's a directory (starts with 📁)
        if stripped.startswith("📁"):
            # Remove the icon and any trailing slash
            dir_name = stripped[1:].strip().rstrip("/")
            # If the current line's level is less than or equal to the top of the stack, pop from the stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            # Add the current directory to the stack
            stack.append((level, dir_name))
        
        # If it's a file (starts with 📄)
        elif stripped.startswith("📄"):
            file_name = stripped[1:].strip()
            if file_name == target_filename:
                # Build the full path from the directories in the stack plus the file name
                full_path = "/".join([dir_name for _, dir_name in stack] + [file_name])
                print(full_path)

if __name__ == "__main__":
    structure_file = "directory_structure.txt"
    target_filename = "tailwind.config.js"
    find_file_in_structure(structure_file, target_filename)
