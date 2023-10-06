import os

def display_directory_structure(directory, indent=0, sibling_count=0):
    """Prints the directory structure recursively."""
    if not os.path.exists(directory):
        print("Directory not found:", directory)
        return

    if sibling_count >= 2:
        return

    items = os.listdir(directory)

    folder_count = 0  # Counter for sibling folders
    for item in items:
        item_path = os.path.join(directory, item)

        if os.path.isdir(item_path):
            print('| ' * indent + '|-- ' + item)
            folder_count += 1
            
            if folder_count > 3:  # If we've displayed 3 siblings, stop further display at this level
                break

            display_directory_structure(item_path, indent + 1)

# Displaying the structure of the current working directory
directory_path = input("Enter the path of the directory (or leave empty for the current directory): ")
directory_path = directory_path or os.getcwd()
display_directory_structure(directory_path)
