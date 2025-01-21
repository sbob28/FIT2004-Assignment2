#task 1
class PrefixTrie:
    def __init__(self):
        """ 
        initializes a prefix trie with four child node placeholders and a location list.
        This trie structure is designed to store sequences made up of characters 'A' to 'D'.

        input:
        there is no input

        output:
        --> creates an empty prefix trie with four child slots (corresponding to letters A-D)
        --> and an empty list to track where sequences start in a larger string.

        complexity analysis:
        time complexity: O(1) because it only involves initializing fixed size attributes.
        space complexity: O(1) because it only allocates space for four pointers and an empty list.
        """

        self.nodes= [None]*4 #intilaises a list of 4 slot for children nodes.
        # a,b,c,d corresponds to each slot 
        self.locations=[]#a list to store locations where each prefix starts in the main string

    def insert(self, location, sequence):

        """
        this method adds a sequence to the trie, starting froma given position.
        for each character in the sequence, it finds the correct position in the trie and creates a new node if needed.
        then it moves to this new node. if a position is provided, it saves that position in the node.

        input:
        location: is an integer representing the starting index of the sequence in the main data.
        sequence: a string of characters to be inserted into the trie.

        output:
        updates the trie by adding nodes corresponding to each character of the sequence.

        complexity analysis
        time complexity- O(M), where M is the length of the sequence. Each character insertion is a constant time operation.
        space complexity: O(M), as in the worst case, a new node might be needed for each character in the sequence.
        """

        current = self #starts from the root node of the trie. iterates over the
        for char in sequence: #iterates over each character in the sequence
            index = ord(char) - ord('A')#calculates the index for the character (assumes only a-d are used)
            if not current.nodes[index]:# checks if there is no trie node or not
                current.nodes[index]=PrefixTrie()#creates a new node if nothing exists 
            current = current.nodes[index] #moves to the node that corresponds to the current character
            if location is not None:#if a valid location is provided then
                current.locations.append(location)#it stores the location in the locations list at this node

    def search(self, pfx):

        """
        this method looks for all occurrences of a given prefix in the trie and returns their starting positions.
        it starts at the root node. then for each character in the prefix, it calculates its position and moves to the corresponding node.
        if a node for the character doesnt exist, retuns empty list.
        if prefix found then it return a list of starting positions for the prefix.

        input:
        pfx: String prefix made of characters 'A' to 'D' to search within the trie.

        output
        returns a list of integers indicating where in the main data the prefix starts.

        complexity
        time complexity: O(P), where P is the length of the prefix. Each step moves to a specific child node.
        space complexity: O(1), uses constant space regardless of input size.
        """

        current = self #start from the root node
        for char in pfx: #iterate over each character in the prefix
            index = ord(char) - ord('A')#calculates the index for the character (assumes only a-d are used)
            # checks if there is no trie node or not
            if not current.nodes[index]:
                return[]#return an empty list if the prefix is not found
            current = current.nodes[index] #move to the next node in the path of the prefix
        return current.locations #return the list of locations where the prefix occurs
    
class OrfFinder:
    def __init__(self,genome):

        """
        intialises an OrfFinder for a given genome sequence by creating a PrefixTrie containing all suffixes of the genome.

        input
        genome: String containing the genome sequence.

        output:
        creates an OrfFinder object with a prefix trie loaded with all possible suffixes of the genome.

        complexity
        time complexity: 
        - inserting suffixes, genome of length N, there are N suffixes
        - the insert method will be called N times
        - each call to insert goes through a suffix of length upto N

        worst case time complexity is O(N^2) because total operations are N +(N-1)+(N-2)...1
        which is (N(N+2))/2 so O(N^2).

        space complexity: O(N^2), as each suffix can potentially create a new path in the trie.
        """

        self.genome = genome #stores the entire genome sequence as string
        self.pfx_t = PrefixTrie()#creates the new PrefixTrie obj
        for i in range (len(genome)):# then it iterates over the len of genome
            self.pfx_t.insert(i, genome[i:])#inserts each suffix of the genome starting from each position
    
    def find(self, start, end):

        """
        finds all substrings in the genome that start with 'start' and end with 'end'.
        it first locates all positions of start and end in the genome, then uses nested loops to find
        the valid substrings that start with start and end with end. each valid
        substring is added to the result list.

        input:
        start: Starting sequence to look for in the genome.
        end: Ending sequence to look for in the genome.

        output:
        Returns a list of substrings from the genome that start with 'start' (prefix) and end with 'end' (suffix).

        complexity:
        time complexity: 
        the time xomplexity for this method is potentially more worse than the required complexity.
        this is because it has nested loops and string slicing involved. this function
        involves finding all start and end location in the genome, which is effecient, but then uses nested 
        loops to compare each pair of start and end locations.
        each comparison involves slicing a substring and checking if it ends with the 
        specified end sequence. 
        this slicing can be costly, id there are long substrings involved which can lead to a complexity of 
        O(N*M*(S+U)). S is the length of the susbtring.
        on the other hand, to achieve O(T+U+V) complexity would require a more effecient 
        approach that processes each character in the genome a minimal number of times. 

        space complexity: O(V), where V is the space needed to store the resulting substrings.
        """

        start_locations = self.pfx_t.search(start)#find all locations where the substring 'start' occurs
        end_locations = self.pfx_t.search(end)#find all locations where the substring 'end' occurs
        result=[]#empty list initialised to store results
        for start_location in start_locations:#loops each start loc
            for end_location in end_locations:#loops each end loc
                if start_location + len(start) <= end_location and self.genome[start_location:end_location + len(end)].endswith(end):
                    # makes sure that the prefix of DNA starts with start sequence and ends with end without any overlapping
                    result.append(self.genome[start_location: end_location+len(end)])
                    #extract the substring from the genome from the start loc to the end of the end sequence
        return result #returns list of all substrings that pass crtieria

#test cases
if __name__ == "__main__":
    def compare(l1, l2):
        return sorted(l1) == sorted(l2)

    genome1 = OrfFinder("ABCABC")

    print("Test 1:", compare(genome1.find("A", "C"), ['ABC', 'ABC', 'ABCABC']))
    print("Test 2:", compare(genome1.find("A", "B"), ['AB', 'AB', 'ABCAB']))
    print("Test 3:", compare(genome1.find("B", "C"), ['BC', 'BC', 'BCABC']))
    print("Test 4:", compare(genome1.find("C", "A"), ['CA']))
    print("Test 5:", compare(genome1.find("AB", "C"), ['ABC', 'ABC', 'ABCABC']))
    print("Test 6:", compare(genome1.find("C", "C"), ['CABC']))
    print("Test 7:", compare(genome1.find("ABCABC", "ABCABC"), []))

    genome2 = OrfFinder("AAA")

    print("Test 8:", compare(genome2.find("A", "A"), ['AA', 'AA', 'AAA']))

genome = OrfFinder('BCBCACCCB')
assert sorted(genome.find('B', 'A')) == sorted(['BCA', 'BCBCA'])
genome = OrfFinder('BADCABDADB')
assert sorted(genome.find('D', 'C')) == sorted(['DC'])
genome = OrfFinder('CCDDBBACABBBBBCDC')
assert sorted(genome.find('B', 'A')) == sorted(['BA', 'BACA', 'BBA', 'BBACA'])
genome = OrfFinder('ACBBCCCBBAABDCCBDDAC')
assert sorted(genome.find('D', 'C')) == sorted(['DAC', 'DC', 'DCC', 'DCCBDDAC', 'DDAC'])
genome = OrfFinder('BACDA')
assert sorted(genome.find('C', 'A')) == sorted(['CDA'])
genome = OrfFinder('CACACBDACCBBBBD')
assert sorted(genome.find('C', 'C')) == sorted(['CAC', 'CAC', 'CACAC', 'CACACBDAC', 'CACACBDACC', 'CACBDAC', 'CACBDACC', 'CBDAC', 'CBDACC', 'CC'])
genome = OrfFinder('ABBDCCCACCB')
assert sorted(genome.find('B', 'A')) == sorted(['BBDCCCA', 'BDCCCA'])
genome = OrfFinder('CACAAAABCDCBCBCCA')
assert sorted(genome.find('A', 'A')) == sorted(['AA', 'AA', 'AA', 'AAA', 'AAA', 'AAAA', 'AAAABCDCBCBCCA', 'AAABCDCBCBCCA', 'AABCDCBCBCCA', 'ABCDCBCBCCA', 'ACA', 'ACAA', 'ACAAA', 'ACAAAA', 'ACAAAABCDCBCBCCA'])



#task 2
from collections import deque

def bfs(graph, source, sink, parent):
    """

    this function performs a breadth first search on a given graph
    to find if there is a path from the source node to the sink node.
    it also updates the parent list so that it can store the path/s.

    it starts from the source node and explores all the neighboruing nodes at the current depth prior to moving on 
    to nodes at the next depth level. the traversal continues until the sink node is reached or all possible paths are explored.

    input:
    graph- is a 2d list representing the matrix of the graph, where graph[u][v] is the capacity of the edge from node u to v.
    source- is an integer that represents the source node
    sink- also an integer that refers to the sink node
    parent- is a list that stores the parent of each node to reconstruct the path from the source to the sink.

    output:
    the function returns either True is there is a path from the source to sink node, or false otherwise. 
    the parent list is updated to represent the path 

    time complexity:
    - to intialise the visited list, it take O(V) time because it needs to set a boolean value for each vertex
    - then in the bfs loop:
        in the worst case, every vertex will be added to the queue exactly once, and edge will be checked once.
        then each vertex is dequeued and processed once, which also involves checking all its different vertices.
        the inner loop iterates through all possible edges of the current vertex. if there are E edges and each vertex is processed once,
        this would make the complexity O(E) over the whole traversal.

    so the total time complexity then would be, O(V+E)

    space complexity:
    - visited list takes O(V) space to because it stores a boolean value for each vertex.
    - queue can hold up to V vertices in the worst case, resulting in O(V) space usage.
    - then the parent list also takes O(V) space to store the parent of each vertex.

    so the overall space complexity is O(V), where v is the number of vertices in the graph.

    """

    visited = [False] * len(graph)#creates a list to keep track of the visited nodes
    queue = deque([source])#this initialises a queue with the source node
    visited[source] = True#this marks a node as visted once it is visted

    while queue: 
        current_node = queue.popleft()#this unadds a vertex from the queue
        for ind, val in enumerate(graph[current_node]):#it iterates through adj vertices
            if not visited[ind] and val > 0:#incase the vertex is not vistsed, but there is still capacity 
                queue.append(ind)#adds the vertex to the queue
                visited[ind] = True#this again marks the vertex as vsisted once it is visited
                parent[ind] = current_node#sets the parent of the vert
                if ind == sink:#incase we reach the sink node
                    return True#then the paths is shown to exist
    return False#otherwise it is not shown to exist (path)

def ford_fulkerson(graph, source, sink):
    """

    this function implements the ford fulkerson algorithm in order to find the max flow in a flow netflow. it uses the BFS function
    to find the augmenting paths from the source to the sink and then updates the capacities of the edges of these paths. 

    input:
    graph- a 2d list representing the adjacency matrix of the graph, where graphu[u][v] is the capacity of the edge from node u to v
    source- an integer representing the source node
    sink- an integer representing the sink node

    output:
    the function returns the max flow from the source node, to the sink node in the graph

    time complexity:
    O(O*F), where E is the number of edges in the graph and F is the max flow in the network
    - each BFS call takes O(V+E) time, where V is the number of vertices and E is the number of edges 
    - in the worst case, the number of times BFS is called is proportional to the maximum flow value.
    this happens when each augmenting path found by bfs adds only one unit of flow to the total flow.

    therefore, the overall time complexity of this algorithm is O(O*F)

    space complexity:
    O(v) is the space complexity, where v is the number of vertices in the graph.
    - the space complexity is dominated by the space that required for the parent list
        - this stores the parent of each vertex in the path and the BFS function's visited list and queue.
    - the parent list takes O(V) as well as the visited list in the BFS method.
    
    so the space complexity is O(V).

    """

    parent = [-1] * len(graph)#this intialises the parent list so i can store the path
    max_flow = 0#initialises the max flow

    while bfs(graph, source, sink, parent):#while there is a path from source to link
        path_flow = float('Inf')#it starts with the infinite path flow
        s = sink

        while s != source: #then this is used to traverse the path that is found by the bfs
            #so that we can find the min capacity
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow 
        #adds the path flow to the overall flow
        v = sink

        while v != source:
            #this part updated the capacities of the edges (including the reverse edges also) along path
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]

    return max_flow #then this returns the overall flow

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    this function allocates shifts to officers based on their preferences using the ford fulkerson algorithm to finds the maximum flow in the network flow.
    it ensures that each officer works a minimum number of shifts and does not exceed the maximum number of shifts allowed.
    the function constructs a capacity graph representing the problem, where nodes correspond to the source,
    sink, officers, office day combinations, and company shift day combinations. at the end if returns an allocation matrix.

    input:
    preferences- a list of lists where preferences[i][k] is 1 if officer i prefers shift k. otherwise it is 0.
    officers_per_org- a list of lists where officers_per_org[j][k] is the number of officers required by company for shift k
    min_shifts- is the minimum number of shifts that each officer should work
    max_shifts- the number of shifts officers can work beyond.

    output:
    the function returns a list (the allocation matrix) indicating which officer is allocated to which company, shift, and day. 
    If the required shifts cannot be assigned, it returns None.
    in my code, the first set of test, the test with max shifts above 15, come out as true but print the wrong allocation.
    the second set of 1000 test cases, all the test cases that are supposed to come out as None are passing,
    and all cases where they are not None, come out as not None, but again, when the allocation is printed or asserted, they are allocated wrong
    so the allocation logic is evidently wrong. 

    Time complexity:
    constructing the graph involves nested loops that iterate through officers, days, comapnies and shifts, which results in O(n*days*m*shifts) time.
    which simplifies to O(m*n) as days and shifts are constants.
    the ford fulkerson algorithm is then applied, which has a complexity of O(V*E*F). given v and e are proptional to O(n*m) because the graph
    includes nodes and connections representing officers, days, comapnies and shifts. and F is proportional to n, so the overall complexity for ford fulkerson
    becomes O(n*m*n)
    
    space complexity:
    the main space is taken up by the graph, which has a size proportional to the total number of nodes which is O(m*n*30*3). this includes
    nodes for the source, sink, officers, office day combinations and company shift day combinations.
    the allocation matric also requires space propotional to the number of officers, days and shifts which results in O(n*30*3).
    the parents list used in BFS takes O(m*n) space.

    so, the overall space complexity is dominated by the capacity graph giving, O(m*n*30*3).
    
    """

    n = len(preferences)  # gets the number of officers
    m = len(officers_per_org)  # gets the no. of companies
    days = 30  # and the number of days in a month
    shifts = 3  #then the no. of shifts per day
    #12am-8am
    # 8am-4pm
    # 4pm to midnight

    node_count = 1 + n + n * days + m * shifts * days + n + 1
    #this calculation is for the total nodes:
    # - 1 for source and sink node
    # - n for officers and min shift nds
    # - n*days for the officer (day nds)
    # - shifts*m*days for company (day nodes)
    source = 0
    sink = node_count - 1

    cap_graph = [[0] * node_count for _ in range(node_count)]
    #starts the graph with zeros

    #sets capacities from source to officer nodes
    #where each officer can work upto max_shifts
    for i in range(n):
        cap_graph[source][1 + i] = max_shifts#set the cap from source to officer i

    #similarly, this sets capacities from officer (day nds)
    #to company shift-day nodes
    #this takes up 1 cap if shift is preferred
    for i in range(n):
        for d in range(days):#and this checks if the officer i prefers shift k
            officer_day_node = 1 + n + i * days + d 
            #this is the calculation for the index of officer, day nds
            cap_graph[1 + i][officer_day_node] = 1 #sets the cap from officer i to officer day node

    #again sets capacities from officer day nodes to company shift
    #this is the number of officers required per shift
    for i in range(n):
        for d in range(days):
            for j in range(m):
                for k in range(shifts):
                    if preferences[i][k] == 1:#this line is used to check if an officer i prefers shift k or no
                        company_shift_day_node = 1 +n +n *days + j* shifts *days+ k *days+ d
                        #calculates the index of the company shift day node
                        cap_graph[officer_day_node][company_shift_day_node] = 1 #sets the cap

    # sets capacities from company shift d noes to sink
    for j in range(m):
        for k in range(shifts):
            for d in range(days):
                company_shift_day_node = 1 +n +n *days + j* shifts *days+ k *days+ d
                #calculate the index of the company shift day node
                cap_graph[company_shift_day_node][sink] = officers_per_org[j][k] #then sets the capacity again

    #sets the capacities from officer nodes to min_shift nodes
    #and from min shift to sink
    for i in range(n):
        officer_min_shifts_node = 1 + n +n*days+m *shifts *days+ i
        #this is the index for the officer min shift node
        cap_graph[1 + i][officer_min_shifts_node] = min_shifts #sets the cap from offcier to min shift
        cap_graph[officer_min_shifts_node][sink] = min_shifts#sets the cap from min shift to sink

    #lets fulk fulkerson algorithm find the max flow
    max_flow = ford_fulkerson(cap_graph, source, sink)

    required_shifts = sum(sum(req) for req in officers_per_org) * days
    if max_flow < required_shifts:
    #calculates the total required shifts for validation
        return None #not all the required shifts can be assigned
    

    # this is the allocation graph from the capacity graph
    allocation = [[[0 for _ in range(shifts)] for _ in range(days)] for _ in range(n)]
    for i in range(n):
        for d in range(days):
            for j in range(m):
                for k in range(shifts):
                    officer_day_node = 1 + n + i * days + d #this is the indec for the officer day node
                    company_shift_day_node = 1 +n +n *days + j* shifts *days+ k *days+ d
                    #calculate the index of the company shift day node
                    #but if there is no cap left then that just means the officer is allocated to that shift
                    if cap_graph[officer_day_node][company_shift_day_node] == 0:
                        allocation[i][d][k] = 1#this allocates officer i to company j on shift k on day d

    return allocation #then finally the allocation matrix is returned


#test cases
#the first set of test, the test with max shifts above 15, come out as true but print the wrong allocation
#the second set of 1000 test cases, all the test cases that are supposed to come out as None are passing,
# and all cases where they are not None, come out as not None, but again, when the allocation is printed or asserted, 
# they are allocated wrong
#allocation logic is evidently wrong
preferences = [
    [1, 0, 0], 
    [1, 0, 0], 
]
officers_per_org = [
    [1, 0, 0],  
]

allocation = allocate(preferences, officers_per_org, min_shifts=15, max_shifts=30)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=14, max_shifts=15)
print("Allocation for min_shifts=14, max_shifts=15:", allocation is not None)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=13, max_shifts=15)
print("Allocation for min_shifts=13, max_shifts=15:", allocation is not None)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=12, max_shifts=15)
print("Allocation for min_shifts=12, max_shifts=15:", allocation is not None)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=15, max_shifts=15)
print("Allocation for min_shifts=15, max_shifts=15:", allocation is not None)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=15, max_shifts=16)
print("Allocation for min_shifts=12, max_shifts=16:", allocation is not None)
print(allocation)

allocation = allocate(preferences, officers_per_org, min_shifts=15, max_shifts=17)
print("Allocation for min_shifts=12, max_shifts=16:", allocation is not None)
print(allocation)

result = allocate([[1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],[[9, 8, 5], [5, 2, 6], [2, 6, 2], [9, 7, 2], [5, 7, 5], [10, 5, 1], [9, 7, 10], [7, 7, 9], [9, 4, 5], [6, 4, 1], [10, 1, 3], [5, 6, 3]],28,28) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]],[[2, 2, 5]],26,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]],[[6, 8, 1]],0,27) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],[[4, 5, 1], [7, 3, 3], [5, 7, 3], [9, 2, 9], [5, 6, 9], [5, 6, 8], [5, 6, 1], [10, 1, 2], [9, 7, 8], [10, 7, 9]],7,18) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 0]],[[6, 4, 7]],15,27) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[2, 5, 5]],17,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[1, 1, 3]],23,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]],[[6, 3, 5]],19,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[2, 3, 5]],2,25) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],[[2, 1, 4]],0,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1]],[[4, 6, 2]],27,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],[[6, 4, 2]],2,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]],[[3, 2, 4]],19,20) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],[[5, 4, 4]],0,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]],[[3, 1, 2]],21,25) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0]],[[1, 1, 1]],13,26) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]],[[2, 3, 1]],25,26) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[3, 2, 5]],11,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]],[[1, 4, 4]],5,23) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0]],[[3, 4, 4]],18,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],[[3, 2, 1]],0,22) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0]],[[10, 9, 8], [4, 4, 2]],12,14) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 1]],[[3, 2, 1]],8,26) 
assert result is None
