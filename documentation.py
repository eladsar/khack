import ast
import pandas as pd
import os
from beam.data import BeamData
from beam import tqdm


def parse_directory(directory):
    # Initialize empty lists for storing the parsed AST objects
    node_index_list = []
    name_list = []
    parent_index_list = []
    type_list = []
    text_content_list = []

    node_index = 1  # Running index for nodes

    # Iterate over all .py files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)

            # Read the file content
            with open(file_path, "r") as file:
                try:
                    file_content = file.read()
                    ast_tree = ast.parse(file_content)
                except SyntaxError:
                    print(f"Syntax error in file: {filename}")
                    continue

            # Function to recursively process the AST nodes
            def process_node(node, parent_index=0, name=None):
                nonlocal node_index

                type_node = type(node).__name__
                if type_node not in ['FunctionDef', 'ClassDef', 'Module', 'AsyncFunctionDef']:
                    return

                # Store the current node's information in the lists
                node_index_list.append(node_index)
                if name is not None:
                    name_list.append(name)
                elif hasattr(node, 'name'):
                    name_list.append(node.name)
                else:
                    name_list.append('NA')
                parent_index_list.append(parent_index)
                type_list.append(type_node)

                # Get the source code text for the current node
                text_content = ast.get_source_segment(file_content, node)
                text_content_list.append(text_content)

                # Increment the node index
                node_index += 1

                # Recursively process child nodes
                for child_node in ast.iter_child_nodes(node):
                    process_node(child_node, node_index_list[-1])

            # Start processing the AST nodes
            process_node(ast_tree, name=filename)

    # Create a dataframe from the collected information
    df = pd.DataFrame({
        'name': name_list,
        'parent_index': parent_index_list,
        'type': type_list,
        'text_content': text_content_list
    }, index=node_index_list)

    return df


if __name__ == '__main__':
    directory_path = '/home/elad/docker/beamds/src/beam'
    store_path = '/home/elad/documentation/beamds'

    ast_dataframe = parse_directory(directory_path)

    bd = BeamData(data=ast_dataframe, path=store_path)
    bd.store()

    print(ast_dataframe.head())