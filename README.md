# Assignment: Implementation of Prefix Trie and Ford-Fulkerson Algorithm

## Overview
This project implements two major components:
1. **Prefix Trie**: A data structure to store and search sequences efficiently.
2. **Ford-Fulkerson Algorithm**: An algorithm to calculate the maximum flow in a flow network.

The implementation provides modular methods for each functionality, allowing easy integration into other projects or for educational purposes.

---

## Components
### 1. Prefix Trie
**Description:**
The `PrefixTrie` class is designed to store and search DNA sequences efficiently. It supports insertion of sequences along with their starting positions and retrieval of all occurrences of specific prefixes.

**Key Methods:**
- `insert(location, sequence)`: Adds a sequence to the trie, starting from a given location.
- `search(pfx)`: Searches for all occurrences of a prefix and returns their starting positions.

**Additional Class: OrfFinder**
- Initializes a `PrefixTrie` with all suffixes of a given genome.
- `find(start, end)`: Finds substrings that start and end with specified sequences.

### 2. Ford-Fulkerson Algorithm
**Description:**
The implementation calculates the maximum flow in a flow network using the Ford-Fulkerson method. It uses Breadth-First Search (BFS) to find augmenting paths.

**Key Functions:**
- `bfs(graph, source, sink, parent)`: Finds if a path exists between the source and sink.
- `ford_fulkerson(graph, source, sink)`: Computes the maximum flow from source to sink.

---

## How to Use
### Prefix Trie
1. **Initialize a Trie**:
   ```python
   trie = PrefixTrie()
   ```
2. **Insert a Sequence**:
   ```python
   trie.insert(location, sequence)
   ```
3. **Search for a Prefix**:
   ```python
   locations = trie.search(prefix)
   ```
4. **Using OrfFinder**:
   ```python
   finder = OrfFinder(genome)
   results = finder.find(start, end)
   ```

### Ford-Fulkerson Algorithm
1. **Set Up a Graph**:
   - Define the adjacency matrix where `graph[u][v]` is the capacity of the edge from node `u` to `v`.
2. **Calculate Maximum Flow**:
   ```python
   max_flow = ford_fulkerson(graph, source, sink)
   ```

---

## Complexity Analysis
### Prefix Trie
- **Insertion**: O(M), where M is the length of the sequence.
- **Search**: O(P), where P is the length of the prefix.
- **OrfFinder Initialization**: O(N^2), where N is the length of the genome.

### Ford-Fulkerson Algorithm
- **BFS**: O(V + E), where V is the number of vertices and E is the number of edges.
- **Ford-Fulkerson**: O(E * max_flow).

---

## File Structure
- **`prefix_trie.py`**: Contains the `PrefixTrie` and `OrfFinder` classes.
- **`ford_fulkerson.py`**: Contains the BFS and Ford-Fulkerson implementations.
- **`README.md`**: Documentation for the assignment.

---

## Example Usage
### Prefix Trie Example
```python
trie = PrefixTrie()
trie.insert(0, "ABCD")
print(trie.search("AB"))  # Output: [0]

finder = OrfFinder("ACGTACGT")
print(finder.find("AC", "GT"))  # Example output: ['ACGT']
```

### Ford-Fulkerson Example
```python
graph = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
source = 0
sink = 5
print(ford_fulkerson(graph, source, sink))  # Output: Maximum flow value
```

---

## License
This project is licensed under the MIT License. Feel free to use and modify the code.

---

## Author
Srividya Bobba

