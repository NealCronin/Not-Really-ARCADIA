import os

def print_tree(startpath, max_depth=3):
    for root, dirs, files in os.walk(startpath):
        # Ignore hidden folders and python cache files
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(startpath, '').count(os.sep)
        if level >= max_depth:
            continue
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if not f.startswith('.') and not f.endswith('.pyc'):
                print(f'{subindent}{f}')

print_tree('.', 5)