def all_paths(graph):
    """
    Given a graph represented as a dictionary in which each key represents a vertex and
    each value is a list of adjacent vertices, finds all possible paths from all root
    nodes to all leaf nodes of the graph using DFS algorithm and returns a dictionary of
    paths in which each key is the source vertex and each value is a list of destination
    vertices.
    """
    paths = {}
    visited = set()

    # Find all root nodes of the graph
    root_nodes = set(graph.keys()) - set(v for adj in graph.values() for v in adj)

    for node in root_nodes:
        paths[node] = []
        all_paths_helper(graph, node, visited, [], paths[node])

    return paths


def all_paths_helper(graph, node, visited, path, paths):
    """
    A helper function that takes a graph, a node, a set of visited nodes, a current path,
    and a list of paths from the root nodes to the leaf nodes, and finds all possible
    paths from the given node to the leaf nodes.
    """
    visited.add(node)
    path.append(node)

    if not graph[node]:
        # This node has no children, so it's a leaf node
        paths.append(path[:])

    for child in graph[node]:
        if child not in visited:
            all_paths_helper(graph, child, visited, path, paths)

    path.pop()
    visited.remove(node)

def build_graph(edges):
    """
    Given a list of edges representing the connected source and destination vertices,
    constructs a unidirectional graph as a dictionary in which each key represents a vertex
    and each value is a list of adjacent vertices.
    """
    graph_dict = {}

    for src, dest in edges:
        if src not in graph_dict:
            graph_dict[src] = []
        if dest not in graph_dict:
            graph_dict[dest] = []
        graph_dict[src].append(dest)

    return graph_dict
