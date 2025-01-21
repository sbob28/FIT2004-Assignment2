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
result = allocate([[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0]],[[2, 8, 3]],3,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]],[[3, 5, 1]],1,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[3, 2, 2]],6,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],[[4, 2, 5]],15,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]],[[3, 5, 10], [4, 10, 9], [3, 6, 6], [2, 6, 6], [4, 8, 3], [5, 10, 4], [8, 1, 9], [3, 3, 1], [7, 3, 10], [4, 2, 6], [2, 6, 4], [8, 10, 7], [2, 6, 10], [4, 6, 10], [3, 8, 9], [7, 1, 1], [2, 8, 7], [9, 9, 9]],0,15) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1]],[[9, 4, 3]],18,26) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]],[[3, 9, 2], [1, 5, 8], [4, 2, 5], [4, 1, 1], [3, 1, 2], [2, 4, 9]],25,25) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]],[[2, 1, 1]],0,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1]],[[10, 8, 5], [5, 3, 8], [6, 2, 9], [7, 3, 10]],12,23) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1]],[[7, 5, 3]],26,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]],[[2, 3, 1]],21,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]],[[5, 9, 3], [10, 10, 4], [9, 5, 5], [10, 9, 4]],3,20) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],[[1, 3, 2]],10,27) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],[[9, 4, 1]],18,27) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]],[[5, 1, 8]],10,26) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]],[[4, 1, 3]],19,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]],[[1, 4, 4]],24,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[5, 4, 5]],0,28) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],[[4, 1, 8]],14,26) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 0]],[[9, 1, 9]],7,20) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1]],[[4, 1, 2], [7, 5, 7]],3,7) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]],[[3, 2, 2]],13,25) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1]],[[7, 5, 10], [3, 2, 1], [5, 4, 2], [2, 5, 8], [2, 5, 5], [5, 6, 5], [8, 9, 6]],18,23) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0]],[[9, 1, 5], [6, 7, 6], [10, 1, 5], [2, 2, 4], [9, 7, 10]],19,28) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],[[2, 1, 4]],14,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]],[[1, 5, 4]],21,27) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]],[[1, 1, 1]],0,16) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0]],[[6, 2, 1], [10, 10, 2], [7, 7, 10], [9, 6, 4], [8, 7, 1], [6, 7, 5], [6, 5, 10], [8, 10, 5], [5, 7, 10], [6, 10, 1], [9, 1, 7], [4, 8, 4], [10, 6, 1], [6, 7, 8]],10,21) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]],[[4, 4, 1]],2,27) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0]],[[4, 5, 10]],22,27) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[2, 2, 1]],30,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]],[[1, 4, 5]],11,28) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[1, 1, 8]],14,21) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 3, 1]],9,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1]],[[2, 10, 10], [10, 7, 5], [4, 2, 2], [5, 4, 7], [7, 10, 6]],8,15) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[2, 7, 1]],4,25) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1]],[[2, 2, 1]],0,19) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],[[7, 9, 4]],25,26) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1]],[[2, 4, 7]],16,23) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0]],[[7, 7, 10], [8, 7, 2]],21,22) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]],[[2, 2, 6]],18,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]],[[1, 3, 1]],6,21) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0]],[[5, 3, 1]],7,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1]],[[6, 7, 4]],13,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1]],[[4, 9, 2]],20,27) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0]],[[3, 9, 2], [2, 2, 9]],3,27) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1]],[[1, 3, 1]],9,27) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],[[7, 6, 7]],20,26) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]],[[4, 4, 9]],0,27) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],[[1, 5, 3]],1,23) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1]],[[2, 4, 8]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],[[5, 2, 2]],0,17) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0]],[[5, 2, 1]],3,28) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0]],[[1, 4, 8]],6,27) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]],[[4, 1, 4]],30,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],[[1, 3, 1]],4,16) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1]],[[4, 3, 1]],4,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]],[[1, 1, 2]],29,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]],[[5, 2, 6]],23,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],[[7, 4, 7]],15,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1]],[[7, 5, 1], [4, 7, 1]],12,23) 
assert result is None
result = allocate([[0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],[[10, 1, 2], [2, 2, 1], [7, 7, 2], [10, 10, 3], [5, 2, 6], [10, 9, 9], [6, 6, 4], [5, 8, 5], [5, 9, 4]],16,25) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 0, 0]],[[2, 6, 4]],0,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]],[[1, 2, 4]],23,26) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0]],[[2, 5, 5], [5, 9, 10], [8, 5, 7], [2, 10, 5], [8, 7, 4], [10, 8, 1]],11,14) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0]],[[9, 4, 1], [2, 3, 2], [3, 10, 4], [1, 10, 10], [7, 1, 4], [5, 2, 3]],8,16) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[4, 4, 2]],17,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],[[2, 4, 3]],17,22) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0]],[[6, 2, 8], [7, 4, 5], [9, 5, 3], [9, 5, 1], [1, 4, 9]],17,20) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0]],[[1, 2, 2]],0,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]],[[5, 3, 2]],0,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 0, 0]],[[5, 2, 1], [7, 3, 7], [4, 3, 5], [3, 7, 1], [3, 5, 7]],15,28) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]],[[4, 3, 1]],15,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]],[[1, 2, 3]],17,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0]],[[5, 10, 8], [10, 1, 6], [9, 5, 8], [3, 4, 8], [8, 6, 10], [8, 7, 2], [3, 3, 3]],27,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1]],[[8, 7, 3], [3, 8, 10]],14,19) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1]],[[4, 2, 3]],26,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1]],[[5, 1, 7], [2, 4, 9], [8, 4, 7], [8, 9, 6], [7, 9, 8], [2, 8, 6], [2, 4, 3], [4, 2, 8]],7,23) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]],[[4, 9, 4]],1,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]],[[5, 10, 3], [9, 2, 1], [10, 9, 1], [4, 3, 9], [6, 3, 9], [5, 5, 10], [7, 3, 1], [7, 4, 9], [4, 9, 8]],16,18) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],[[2, 1, 1]],12,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]],[[6, 7, 5], [3, 6, 9], [9, 10, 1], [2, 10, 6], [3, 4, 6], [7, 5, 3], [6, 2, 3], [9, 5, 5], [4, 2, 8]],26,30) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],[[3, 1, 4]],0,21) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]],[[2, 1, 2]],19,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1]],[[5, 1, 1]],0,28) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],[[3, 4, 4]],5,24) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]],[[3, 6, 9]],15,25) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]],[[10, 5, 1], [10, 3, 2], [10, 6, 6], [2, 8, 8], [1, 1, 10], [2, 1, 10], [1, 8, 10]],25,27) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 3, 5]],8,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]],[[3, 2, 1]],21,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]],[[5, 5, 3]],13,21) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[4, 1, 2]],20,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 0], [0, 0, 1]],[[7, 7, 8], [9, 3, 9]],16,21) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],[[1, 1, 1]],6,17) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0]],[[10, 1, 10], [6, 10, 7], [8, 6, 2], [2, 2, 3]],16,16) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],[[1, 1, 1]],1,24) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],[[2, 1, 3]],2,24) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0]],[[8, 9, 2], [2, 8, 7], [2, 5, 10], [2, 6, 7], [5, 4, 9]],16,24) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1]],[[1, 2, 5]],8,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1]],[[3, 2, 7]],15,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]],[[4, 2, 3]],14,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[2, 1, 2]],4,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],[[3, 1, 7]],13,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1]],[[1, 6, 10]],0,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 2, 1]],13,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]],[[2, 1, 2]],1,22) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[2, 5, 2]],12,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0]],[[8, 3, 2]],22,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1]],[[5, 9, 9], [6, 8, 1]],23,25) 
assert result is None
result = allocate([[1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]],[[3, 10, 10], [9, 5, 4]],24,27) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]],[[2, 1, 1]],8,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]],[[4, 4, 2]],11,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]],[[2, 3, 2]],11,26) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]],[[4, 6, 2]],19,25) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]],[[6, 4, 4], [1, 9, 1], [9, 1, 3], [6, 6, 10], [2, 6, 4], [6, 4, 10], [1, 4, 10], [10, 3, 3], [3, 9, 5]],13,17) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],[[6, 1, 2]],2,24) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],[[1, 2, 1]],16,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[1, 5, 1]],19,28) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]],[[3, 5, 2], [10, 9, 1], [8, 3, 6], [1, 3, 8], [7, 10, 6], [10, 1, 3], [4, 3, 4]],26,28) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0]],[[2, 6, 4]],10,25) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 0]],[[7, 7, 10]],28,28) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],[[4, 2, 4]],3,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0]],[[3, 1, 2]],0,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0]],[[4, 9, 5]],19,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[2, 1, 5]],6,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]],[[4, 2, 3]],11,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],[[1, 9, 2], [8, 7, 10], [6, 3, 2], [10, 8, 6], [3, 10, 2], [5, 4, 2], [1, 9, 3], [7, 4, 6], [1, 9, 6], [4, 1, 2], [9, 10, 3], [1, 7, 9], [7, 3, 6], [10, 8, 9], [7, 6, 9], [2, 4, 2]],27,28) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]],[[4, 2, 9]],12,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1]],[[2, 5, 3]],0,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[1, 2, 1]],11,21) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],[[1, 1, 1]],11,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[2, 3, 4]],19,25) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]],[[6, 5, 1]],24,29) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]],[[2, 2, 1]],6,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0]],[[2, 4, 2]],0,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],[[4, 2, 1]],6,18) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]],[[2, 3, 1]],22,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[6, 2, 2]],6,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],[[1, 1, 8]],6,29) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[4, 2, 3]],16,21) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1]],[[4, 7, 6]],21,27) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[1, 4, 1], [9, 9, 6], [10, 1, 5], [10, 9, 4]],22,22) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[9, 7, 4], [3, 6, 4], [6, 9, 5], [1, 6, 3], [8, 9, 5], [2, 7, 8]],4,6) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]],[[3, 1, 1]],11,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]],[[3, 3, 1]],18,21) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]],[[1, 1, 2]],12,21) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0]],[[1, 1, 2]],6,10) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1]],[[9, 10, 9], [10, 7, 8], [9, 4, 7], [6, 5, 10]],22,22) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]],[[4, 3, 4]],19,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],[[6, 10, 1], [1, 10, 4], [3, 9, 5]],4,8) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],[[1, 3, 1]],0,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]],[[1, 3, 3]],23,26) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0]],[[6, 10, 2], [9, 3, 1], [10, 4, 8], [4, 7, 2], [3, 6, 8]],13,29) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],[[7, 3, 8], [5, 4, 8], [3, 3, 4]],6,25) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]],[[2, 1, 8], [3, 1, 5], [4, 4, 9], [9, 8, 6], [6, 10, 9], [8, 7, 5], [2, 6, 2], [7, 6, 2], [4, 8, 1], [9, 5, 10], [8, 2, 9], [8, 2, 10]],26,27) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1]],[[1, 1, 2]],0,24) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1], [1, 0, 0]],[[3, 9, 3], [2, 7, 1], [5, 3, 7], [6, 5, 1], [4, 7, 6], [1, 8, 5], [2, 7, 8], [2, 2, 1]],6,23) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]],[[6, 7, 10], [2, 7, 10]],10,17) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[1, 8, 2], [2, 10, 5], [3, 2, 4], [4, 1, 9], [5, 8, 2], [9, 1, 4], [8, 9, 9], [7, 9, 10], [2, 4, 2], [7, 2, 10]],14,29) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]],[[1, 1, 1]],14,20) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[2, 2, 10], [10, 9, 5], [7, 5, 10]],9,20) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]],[[1, 3, 4]],9,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],[[1, 3, 1]],13,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1]],[[1, 3, 2], [4, 1, 1]],16,24) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0]],[[3, 7, 6], [4, 10, 5], [10, 4, 10], [2, 7, 8]],23,28) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1]],[[4, 5, 2]],23,28) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1]],[[7, 3, 1]],0,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0]],[[4, 2, 1]],22,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1]],[[1, 2, 1]],10,26) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[4, 3, 5]],21,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0]],[[7, 6, 3]],10,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]],[[1, 1, 4]],0,17) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]],[[4, 4, 2]],2,24) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],[[8, 1, 2]],12,28) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],[[2, 1, 2]],3,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 1, 1]],6,22) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]],[[6, 4, 9], [8, 7, 7], [2, 8, 1], [5, 9, 1], [1, 6, 8], [8, 1, 2], [1, 8, 3], [8, 7, 6], [3, 3, 8], [5, 8, 3], [6, 10, 9]],25,28) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0]],[[2, 1, 2], [4, 3, 2]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1]],[[3, 3, 1]],7,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]],[[1, 6, 4]],10,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]],[[4, 7, 3]],28,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]],[[1, 3, 7]],10,26) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]],[[1, 1, 5]],26,28) 
assert result is not None
result = allocate([[0, 0, 1]],[[6, 6, 1]],25,26) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]],[[7, 5, 4]],19,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],[[1, 3, 1]],19,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]],[[2, 1, 2]],7,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]],[[5, 1, 1]],3,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[1, 6, 6]],29,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]],[[5, 4, 3]],23,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[4, 4, 7]],22,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]],[[4, 4, 5]],0,24) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]],[[3, 2, 5]],21,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]],[[6, 2, 2]],4,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]],[[2, 2, 3]],0,22) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]],[[1, 1, 1]],7,9) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]],[[5, 1, 2]],4,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1]],[[2, 3, 2]],22,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[3, 2, 1]],13,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]],[[1, 2, 2]],0,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0]],[[2, 1, 5]],24,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]],[[5, 2, 9], [6, 6, 6], [10, 10, 6], [6, 3, 4], [1, 10, 3], [3, 9, 9]],28,29) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1]],[[6, 4, 4]],19,28) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[3, 2, 3]],1,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0]],[[6, 4, 8], [4, 5, 1], [5, 2, 4], [8, 9, 8], [2, 8, 2], [2, 6, 7], [5, 7, 3], [9, 1, 3], [2, 10, 2], [9, 9, 3], [4, 10, 2], [2, 10, 8], [5, 9, 2], [10, 8, 4], [2, 4, 1]],4,30) 
assert result is None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],[[3, 8, 2], [6, 10, 4], [1, 9, 8], [7, 3, 5], [8, 10, 2], [6, 4, 8], [7, 1, 5], [7, 9, 1], [8, 10, 7], [7, 7, 1]],5,5) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]],[[3, 4, 3]],27,29) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]],[[2, 2, 7]],19,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 0]],[[1, 4, 3]],0,20) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]],[[9, 3, 4]],6,24) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]],[[2, 1, 4]],7,24) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]],[[5, 3, 5], [7, 7, 1]],9,10) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]],[[5, 7, 1], [5, 10, 7], [10, 8, 1], [10, 1, 3], [4, 7, 4], [9, 4, 4]],15,27) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]],[[4, 1, 1]],0,26) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1]],[[7, 1, 6], [2, 8, 4], [10, 9, 10], [5, 3, 3], [1, 5, 2], [9, 5, 5], [1, 4, 5]],3,22) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]],[[10, 9, 5], [2, 10, 7], [2, 6, 9], [6, 9, 6], [8, 2, 7], [4, 7, 4], [5, 9, 5], [5, 2, 5], [6, 1, 2], [10, 5, 5], [9, 7, 5], [1, 5, 4], [7, 4, 5], [4, 9, 6], [10, 10, 1], [10, 2, 2], [9, 7, 7], [10, 3, 5], [8, 1, 5]],22,28) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]],[[1, 4, 2]],7,20) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]],[[1, 2, 6]],7,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],[[1, 4, 2]],28,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]],[[3, 10, 2]],14,25) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1]],[[1, 1, 1]],6,20) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1]],[[4, 3, 6]],27,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 0]],[[2, 7, 1], [3, 10, 6]],9,19) 
assert result is None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[1, 2, 4]],14,27) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[2, 1, 1]],10,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0]],[[2, 4, 6]],7,16) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0]],[[1, 7, 3]],2,28) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]],[[7, 2, 4]],8,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]],[[1, 6, 2]],1,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]],[[2, 5, 2]],21,28) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[2, 1, 3]],20,23) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]],[[3, 2, 9]],22,25) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1]],[[1, 1, 5]],18,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]],[[4, 2, 1]],11,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 7, 1]],2,27) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]],[[3, 1, 3]],19,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]],[[3, 1, 7], [8, 8, 3]],27,27) 
assert result is None
result = allocate([[0, 0, 0], [1, 1, 0]],[[7, 1, 7], [3, 8, 3]],2,29) 
assert result is None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0]],[[1, 1, 2]],0,14) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1]],[[3, 1, 2]],4,23) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1]],[[4, 7, 9], [4, 7, 1], [7, 2, 10], [1, 6, 2], [5, 5, 2], [9, 6, 8], [6, 3, 7]],24,28) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1]],[[1, 3, 5]],8,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]],[[1, 3, 6]],18,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],[[5, 6, 8]],7,8) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]],[[6, 2, 7]],6,19) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]],[[1, 5, 2]],6,28) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[4, 7, 1]],15,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],[[5, 4, 3]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0]],[[3, 1, 1]],0,15) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1]],[[5, 2, 1], [9, 3, 8]],30,30) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0]],[[5, 3, 6]],18,22) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 1]],[[10, 6, 5]],3,25) 
assert result is None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[2, 2, 4]],0,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],[[9, 6, 8]],22,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1]],[[1, 9, 10], [9, 8, 7], [5, 6, 5]],6,24) 
assert result is None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],[[2, 1, 2]],0,25) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0]],[[2, 5, 6]],6,19) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]],[[1, 1, 1]],12,24) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0]],[[1, 1, 5]],5,22) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],[[8, 1, 4]],7,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0]],[[1, 3, 2]],30,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]],[[1, 2, 2]],10,26) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0]],[[1, 2, 3]],10,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[1, 1, 2]],14,22) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[3, 6, 4]],29,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0]],[[6, 3, 2]],25,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0]],[[1, 3, 1]],7,25) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]],[[1, 1, 1]],1,12) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[4, 1, 1]],11,26) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0]],[[4, 3, 1]],26,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],[[5, 3, 2]],18,25) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],[[5, 3, 2]],9,25) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0]],[[2, 5, 1]],7,27) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0]],[[1, 1, 1]],6,16) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],[[5, 4, 6]],13,26) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]],[[2, 1, 2]],30,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]],[[1, 1, 3]],0,20) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],[[4, 3, 10], [2, 6, 8]],7,30) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]],[[3, 4, 7]],0,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],[[7, 2, 5]],8,23) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]],[[3, 1, 1]],13,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],[[1, 2, 6], [4, 8, 4], [8, 1, 6], [1, 5, 6], [10, 6, 8], [4, 6, 4]],19,20) 
assert result is None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1]],[[2, 3, 5], [9, 5, 10], [6, 1, 4]],7,15) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],[[6, 8, 1]],24,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],[[2, 3, 4]],6,22) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[3, 1, 2]],12,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]],[[1, 10, 6], [10, 5, 9], [9, 10, 6], [5, 4, 8], [6, 5, 1], [3, 1, 5], [9, 7, 6], [9, 8, 9], [2, 2, 9]],27,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1]],[[2, 5, 8]],22,24) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1]],[[6, 4, 4]],0,23) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],[[7, 2, 3]],4,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]],[[4, 4, 7]],19,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]],[[4, 5, 8], [1, 10, 1], [10, 9, 2], [2, 4, 5], [8, 4, 9], [7, 4, 1], [6, 5, 10], [1, 4, 3], [5, 2, 9]],15,24) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]],[[7, 7, 2]],21,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1]],[[3, 3, 1]],0,23) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[2, 1, 4]],4,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0]],[[3, 1, 1]],15,26) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]],[[7, 9, 10], [10, 1, 9], [6, 8, 5], [5, 2, 10], [5, 10, 4], [3, 6, 6], [9, 2, 5]],19,25) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1]],[[2, 6, 3]],20,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]],[[8, 4, 6]],14,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1]],[[8, 2, 4]],8,27) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]],[[2, 5, 4]],23,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1]],[[7, 3, 2]],13,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],[[1, 2, 8], [5, 2, 4]],15,24) 
assert result is None
result = allocate([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],[[2, 1, 7]],0,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[1, 2, 1]],22,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],[[2, 2, 1]],10,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]],[[4, 8, 3]],20,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1]],[[4, 7, 5]],1,29) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]],[[3, 1, 4]],7,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1]],[[2, 9, 2]],0,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1]],[[2, 1, 2]],11,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [0, 0, 1]],[[2, 5, 5], [8, 8, 2]],11,23) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 0, 1]],[[8, 1, 8], [8, 2, 9], [2, 1, 4]],28,28) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]],[[2, 3, 3]],18,27) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0]],[[9, 9, 1], [10, 10, 1], [4, 2, 9], [10, 7, 6], [5, 5, 10], [7, 6, 1], [10, 1, 10], [3, 7, 9], [3, 7, 10], [5, 5, 6], [7, 6, 10], [2, 10, 6], [3, 8, 8], [3, 10, 10]],3,14) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 1]],[[10, 2, 10], [7, 4, 2]],19,30) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0]],[[6, 2, 8], [9, 3, 6], [6, 2, 5], [5, 3, 5], [4, 3, 5], [1, 2, 3], [1, 4, 9]],30,30) 
assert result is None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1]],[[4, 2, 1]],30,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]],[[2, 3, 7]],9,20) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]],[[2, 1, 1]],18,24) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]],[[10, 8, 1], [2, 3, 7], [7, 4, 4], [3, 1, 8], [6, 2, 5], [5, 4, 4], [1, 1, 1], [2, 10, 2], [1, 2, 3], [3, 8, 3], [4, 5, 3]],2,3) 
assert result is None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]],[[1, 3, 1], [6, 5, 3]],20,24) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]],[[3, 2, 2]],17,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]],[[1, 5, 1]],2,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]],[[2, 2, 2]],9,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 2, 3]],16,23) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 1, 5]],10,24) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]],[[3, 1, 8]],1,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0]],[[2, 5, 7], [1, 7, 3]],23,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]],[[2, 1, 1]],30,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[3, 1, 3]],29,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]],[[4, 4, 2]],26,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0]],[[3, 3, 8]],6,23) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0]],[[9, 3, 10]],9,23) 
assert result is None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]],[[9, 3, 5]],12,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]],[[6, 1, 1]],0,29) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]],[[3, 2, 7]],13,25) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1]],[[2, 5, 6]],19,23) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]],[[2, 3, 2]],24,27) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]],[[1, 6, 6]],7,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],[[9, 6, 5]],1,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1]],[[6, 5, 6]],1,26) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]],[[2, 2, 1]],7,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]],[[2, 1, 1]],8,27) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1]],[[8, 2, 1], [3, 8, 1], [7, 10, 5], [6, 9, 3], [10, 6, 8]],12,13) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],[[4, 1, 1]],15,25) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1]],[[3, 2, 2]],30,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1]],[[4, 3, 3], [6, 9, 9], [1, 9, 5], [5, 6, 9], [10, 10, 7], [7, 3, 8], [4, 5, 10], [2, 9, 6], [5, 8, 3]],24,27) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]],[[7, 5, 1]],5,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],[[1, 7, 2], [9, 4, 3], [6, 7, 8], [2, 2, 10]],20,28) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0]],[[6, 4, 3]],19,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0]],[[6, 8, 1]],4,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],[[3, 8, 1]],3,28) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]],[[1, 3, 3]],2,26) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0]],[[3, 9, 1], [4, 9, 1]],20,24) 
assert result is None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1]],[[6, 6, 7], [4, 8, 3], [4, 8, 9], [5, 6, 2], [4, 6, 7], [7, 10, 2], [4, 10, 9], [2, 9, 8], [8, 1, 2], [9, 10, 9], [6, 9, 3], [7, 10, 4], [3, 2, 3], [9, 2, 2], [9, 8, 1], [5, 9, 2], [7, 7, 3], [7, 1, 2], [10, 6, 5], [8, 8, 3], [2, 5, 1], [10, 7, 5]],8,27) 
assert result is None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0]],[[1, 4, 4]],3,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]],[[2, 3, 2]],25,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]],[[8, 4, 9]],7,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]],[[1, 3, 8]],13,24) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0]],[[9, 8, 9], [2, 5, 8], [1, 2, 5], [3, 2, 7]],14,21) 
assert result is None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]],[[4, 7, 2]],9,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1]],[[9, 1, 4]],8,26) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1]],[[8, 2, 2]],21,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[2, 2, 10]],22,26) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],[[5, 6, 5]],8,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],[[5, 3, 6], [5, 3, 2], [10, 5, 10], [7, 3, 8], [10, 6, 8], [8, 3, 5], [4, 8, 4]],1,13) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0]],[[1, 1, 10], [6, 5, 7], [10, 6, 3], [2, 2, 1], [6, 9, 7], [8, 3, 6], [3, 5, 7], [4, 6, 3], [6, 4, 1], [4, 3, 1], [2, 2, 1]],7,9) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0]],[[5, 3, 1]],13,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]],[[5, 1, 7]],11,28) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],[[4, 1, 5]],25,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]],[[4, 6, 10]],23,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]],[[1, 1, 3]],2,29) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]],[[6, 8, 2], [1, 8, 8], [2, 4, 7], [9, 8, 1]],7,24) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1]],[[1, 7, 3], [8, 10, 10], [6, 7, 6], [5, 7, 8], [6, 7, 6], [4, 9, 9], [7, 9, 5], [8, 2, 1], [3, 7, 7], [4, 8, 8], [5, 4, 8]],21,26) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]],[[9, 9, 6], [4, 3, 3], [7, 9, 5], [8, 1, 10], [5, 9, 6]],15,26) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0]],[[9, 2, 3]],0,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]],[[2, 3, 1]],17,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]],[[1, 3, 1]],0,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]],[[3, 1, 2]],10,21) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],[[5, 2, 1]],8,18) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]],[[3, 4, 1]],1,28) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],[[1, 1, 3]],30,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]],[[3, 4, 1]],12,22) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]],[[4, 1, 2]],12,21) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[10, 3, 9], [5, 5, 3], [1, 9, 9], [6, 8, 2], [2, 7, 6], [9, 9, 4], [2, 1, 4], [2, 9, 5], [4, 1, 1]],17,17) 
assert result is None
result = allocate([[0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0]],[[3, 5, 1]],0,20) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]],[[6, 5, 5]],5,18) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]],[[5, 8, 6]],4,4) 
assert result is None
result = allocate([[1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]],[[2, 1, 10], [9, 10, 8], [5, 3, 8], [8, 5, 2], [6, 7, 3], [10, 9, 10], [10, 1, 6], [8, 7, 5], [8, 3, 7], [3, 6, 10]],24,28) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1]],[[2, 2, 1]],10,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1]],[[4, 2, 2]],23,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0]],[[7, 7, 2]],2,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]],[[6, 2, 4]],12,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],[[4, 7, 1]],29,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[3, 5, 7]],3,5) 
assert result is None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 0, 1]],[[1, 1, 1]],11,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0]],[[7, 7, 1]],11,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],[[2, 1, 2]],19,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 1, 1]],9,25) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[3, 4, 9]],0,27) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]],[[3, 5, 5], [1, 10, 7], [2, 9, 7], [6, 6, 10], [10, 8, 10], [10, 5, 9], [6, 6, 6], [10, 2, 1]],24,24) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1]],[[5, 4, 6]],18,26) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]],[[1, 3, 1]],20,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],[[2, 1, 1]],20,28) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],[[4, 4, 3]],24,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]],[[1, 6, 5]],10,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[2, 4, 1]],22,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],[[3, 1, 8]],29,29) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],[[2, 2, 2]],18,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]],[[3, 7, 1], [10, 4, 7], [7, 7, 6], [3, 4, 6], [7, 9, 9], [8, 5, 8]],11,28) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[1, 2, 3]],13,23) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]],[[1, 2, 2]],13,28) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1]],[[1, 7, 1]],4,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],[[2, 1, 2]],7,25) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],[[2, 4, 1]],2,22) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1]],[[1, 7, 7]],0,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[3, 1, 3]],0,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],[[1, 6, 7]],5,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[6, 2, 5]],0,25) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]],[[5, 6, 3], [10, 9, 10], [8, 4, 3], [3, 8, 10], [3, 3, 7], [4, 9, 4], [7, 9, 6], [6, 10, 5], [8, 4, 2], [9, 9, 7], [6, 2, 6]],29,29) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]],[[9, 10, 6], [1, 5, 4], [3, 2, 9]],24,28) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1]],[[3, 4, 2]],4,27) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]],[[3, 2, 3]],0,11) 
assert result is not None
result = allocate([[1, 0, 0]],[[8, 3, 6]],23,27) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]],[[3, 2, 5]],11,20) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],[[2, 9, 6]],12,19) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1]],[[3, 2, 1]],16,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]],[[2, 5, 1]],18,26) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1]],[[7, 2, 8]],1,27) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1]],[[3, 3, 3]],30,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],[[1, 2, 9]],10,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[2, 2, 2]],0,24) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],[[3, 1, 3]],9,20) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],[[4, 1, 2]],17,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],[[10, 7, 3]],25,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]],[[2, 5, 2]],14,23) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 3, 1]],17,25) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]],[[3, 2, 3]],10,26) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1]],[[3, 1, 8]],13,25) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0]],[[2, 1, 1], [3, 5, 10]],30,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],[[1, 1, 3]],23,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1]],[[9, 2, 3], [4, 9, 4], [6, 9, 3], [5, 1, 8], [5, 1, 4], [7, 6, 6], [7, 8, 8], [3, 4, 6], [3, 3, 10], [10, 2, 8], [5, 8, 2], [10, 2, 5], [3, 10, 9], [7, 10, 7]],25,30) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0]],[[6, 2, 3]],28,30) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]],[[1, 4, 3]],0,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0]],[[3, 4, 2]],18,24) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0]],[[2, 2, 4]],21,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]],[[1, 1, 4]],6,23) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0]],[[3, 4, 9]],24,27) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]],[[1, 7, 1]],15,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]],[[2, 6, 4]],0,20) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1]],[[1, 1, 1]],0,27) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1]],[[2, 1, 7]],0,29) 
assert result is not None
result = allocate([[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]],[[10, 1, 2]],14,22) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]],[[3, 6, 1]],6,27) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]],[[2, 2, 2]],9,24) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]],[[9, 4, 3]],21,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1]],[[4, 6, 7]],27,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],[[5, 1, 10], [5, 3, 4], [1, 7, 1], [4, 4, 8], [7, 6, 10], [9, 9, 4], [5, 3, 6], [1, 9, 7], [9, 3, 5], [6, 6, 10], [7, 4, 6]],19,29) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0]],[[2, 6, 9]],8,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1]],[[5, 4, 2]],0,27) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1]],[[3, 4, 1]],2,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1]],[[1, 1, 1]],6,23) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[3, 2, 1]],9,17) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]],[[1, 2, 2]],0,24) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1]],[[3, 6, 9]],0,30) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0]],[[1, 4, 4]],0,23) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]],[[6, 1, 4]],8,20) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],[[6, 6, 2]],0,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 0]],[[2, 7, 2], [5, 6, 6], [6, 5, 10], [8, 8, 4], [5, 8, 8], [10, 2, 5], [2, 7, 8], [8, 1, 8], [8, 3, 1], [5, 9, 8], [6, 8, 8], [7, 6, 9], [10, 4, 10], [10, 7, 2], [7, 7, 6], [3, 6, 10], [6, 7, 3], [8, 10, 9], [5, 8, 2], [4, 7, 6]],8,10) 
assert result is None
result = allocate([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0]],[[1, 3, 4], [1, 6, 7], [5, 3, 9], [10, 9, 10], [7, 10, 9], [10, 1, 3], [5, 5, 3], [3, 1, 7], [6, 7, 3], [7, 2, 10], [10, 9, 7], [4, 8, 6], [2, 10, 10], [6, 9, 5], [9, 4, 3], [9, 7, 7], [7, 10, 6], [9, 4, 5]],28,29) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1]],[[4, 1, 2]],16,18) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [0, 1, 0]],[[4, 2, 10], [10, 7, 4], [5, 1, 9]],4,23) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0]],[[5, 3, 6]],20,24) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 1]],[[9, 2, 4], [8, 9, 10], [7, 1, 7]],3,21) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]],[[4, 6, 8]],0,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1]],[[3, 2, 4]],17,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]],[[10, 3, 9]],2,24) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],[[1, 2, 1]],23,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]],[[3, 1, 1]],2,26) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[1, 4, 2]],27,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]],[[4, 2, 1]],8,27) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0]],[[5, 3, 4]],20,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]],[[1, 2, 1]],2,27) 
assert result is not None
result = allocate([[0, 1, 0]],[[1, 3, 9]],20,26) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1]],[[1, 2, 3]],22,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 0, 0]],[[9, 6, 1], [10, 3, 6], [10, 3, 7]],19,19) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0]],[[6, 1, 6]],10,28) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0]],[[3, 4, 7]],28,30) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0]],[[3, 7, 3], [5, 6, 2], [8, 8, 10], [6, 2, 9]],9,18) 
assert result is None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0]],[[2, 1, 3]],18,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[2, 3, 1]],13,23) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0]],[[6, 5, 3]],3,24) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[1, 1, 1]],6,12) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1]],[[4, 2, 3]],9,28) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[1, 2, 6]],7,16) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]],[[1, 6, 4]],4,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]],[[4, 1, 3]],0,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1]],[[3, 8, 1]],14,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]],[[3, 1, 2]],2,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]],[[1, 1, 3]],6,23) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0]],[[6, 5, 5]],5,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1]],[[1, 1, 5]],20,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[4, 1, 2]],13,27) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0]],[[1, 4, 1]],2,22) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]],[[1, 2, 1]],7,24) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]],[[3, 2, 1]],11,19) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],[[4, 1, 2], [1, 9, 4], [7, 8, 9], [3, 10, 1], [3, 2, 7], [1, 8, 2], [8, 10, 6]],4,16) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],[[6, 2, 6]],15,28) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]],[[2, 1, 4]],0,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],[[1, 3, 3], [3, 7, 4], [10, 7, 9], [5, 5, 4], [3, 8, 5], [4, 8, 5], [3, 4, 1], [5, 5, 5], [4, 5, 3]],0,24) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [1, 1, 0]],[[4, 8, 4], [3, 4, 1], [2, 7, 3], [2, 4, 2]],1,26) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]],[[1, 1, 1]],11,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1]],[[4, 1, 7]],0,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0]],[[4, 6, 8], [7, 4, 10], [4, 2, 8], [10, 5, 4]],22,30) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],[[5, 1, 1]],1,27) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],[[2, 1, 5]],10,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1]],[[2, 1, 3]],16,20) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 0]],[[3, 1, 7]],25,28) 
assert result is None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]],[[1, 1, 2]],12,18) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1]],[[2, 1, 3]],10,17) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]],[[1, 9, 3], [1, 1, 8], [5, 4, 2], [1, 6, 3], [1, 5, 6], [10, 9, 8], [3, 8, 1], [3, 6, 9], [6, 5, 6], [5, 5, 3], [7, 9, 1], [6, 3, 5], [1, 3, 10], [2, 10, 2]],1,30) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0]],[[10, 6, 8], [6, 10, 10], [7, 6, 10]],2,23) 
assert result is None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]],[[2, 3, 2]],2,23) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]],[[3, 5, 7], [1, 3, 3], [4, 8, 2], [8, 2, 7], [2, 1, 9], [1, 10, 4], [5, 9, 3], [3, 10, 6]],22,26) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]],[[9, 4, 7], [2, 3, 9]],6,25) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1]],[[2, 3, 4]],7,29) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],[[2, 4, 9]],0,28) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]],[[7, 8, 8], [6, 10, 2], [5, 10, 8], [9, 6, 1]],1,9) 
assert result is None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]],[[2, 2, 2]],4,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0]],[[2, 4, 6]],14,28) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],[[3, 3, 2]],23,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]],[[3, 4, 1]],20,26) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0]],[[4, 5, 9]],9,19) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[5, 3, 2]],21,28) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0]],[[1, 3, 7], [8, 2, 4]],20,23) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]],[[3, 6, 4]],18,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1]],[[4, 3, 3]],9,20) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]],[[1, 3, 4]],0,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1]],[[3, 1, 4]],21,26) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1]],[[4, 3, 4]],6,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],[[9, 10, 8], [5, 1, 9], [1, 1, 4]],20,22) 
assert result is None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0]],[[3, 2, 4]],0,26) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]],[[6, 1, 9], [6, 3, 1], [1, 6, 1], [6, 8, 2], [9, 2, 5]],11,17) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0]],[[1, 9, 5]],13,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],[[2, 5, 2]],8,17) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[1, 3, 2]],11,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[2, 3, 9]],17,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]],[[6, 3, 1]],7,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[5, 1, 1]],9,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0]],[[4, 10, 1], [7, 8, 2], [9, 4, 2], [9, 7, 7]],27,28) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]],[[1, 4, 2]],11,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1]],[[9, 9, 6]],15,21) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0]],[[2, 5, 6], [6, 2, 1], [3, 9, 9], [2, 10, 3], [3, 1, 8], [10, 9, 9], [2, 8, 9], [7, 3, 3], [1, 2, 5], [8, 1, 4], [8, 9, 6], [8, 4, 4], [7, 1, 6], [2, 1, 6], [9, 6, 9], [9, 1, 3], [1, 8, 5]],5,12) 
assert result is None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],[[2, 2, 5]],8,22) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1]],[[9, 6, 3]],6,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1]],[[1, 1, 1]],5,24) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [0, 0, 1]],[[4, 10, 5], [3, 5, 4], [5, 7, 3]],24,29) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0]],[[2, 7, 4]],0,22) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0]],[[8, 1, 4]],23,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],[[1, 5, 4]],15,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]],[[1, 3, 5]],0,18) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]],[[2, 2, 4]],3,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]],[[1, 1, 6]],6,27) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1]],[[1, 1, 5]],2,17) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]],[[2, 1, 5]],1,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1]],[[4, 3, 7], [9, 2, 7], [9, 5, 8], [3, 3, 8], [1, 3, 8], [9, 5, 8], [6, 9, 8]],26,29) 
assert result is None
result = allocate([[0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1]],[[5, 1, 9], [2, 5, 9], [9, 7, 9], [4, 10, 10], [7, 7, 8]],3,28) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1]],[[5, 5, 2], [4, 5, 7]],14,24) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1]],[[2, 1, 5]],12,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1]],[[3, 8, 2]],0,20) 
assert result is not None
result = allocate([[1, 0, 1]],[[2, 1, 3]],30,30) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1]],[[6, 5, 6], [5, 9, 2]],3,25) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0]],[[1, 1, 1]],0,19) 
assert result is not None
result = allocate([[1, 0, 0]],[[10, 8, 10]],13,24) 
assert result is None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[2, 4, 6]],17,27) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0]],[[2, 3, 6], [10, 10, 10], [4, 3, 1], [9, 5, 9], [9, 2, 4], [5, 4, 7]],28,28) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0]],[[5, 4, 4]],4,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[2, 4, 4]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]],[[8, 1, 2]],24,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[4, 3, 2], [6, 5, 10]],17,26) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],[[1, 3, 6]],12,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0]],[[1, 1, 1]],17,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0]],[[3, 6, 2]],7,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],[[5, 1, 3]],13,19) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[4, 6, 2]],18,29) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]],[[2, 5, 2]],0,21) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1]],[[3, 4, 2]],29,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],[[10, 2, 2]],21,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[7, 7, 10], [8, 4, 9]],22,30) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]],[[3, 9, 5], [5, 1, 10], [4, 8, 5], [5, 8, 4], [4, 10, 7], [3, 6, 1]],5,13) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 0]],[[3, 5, 4]],0,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1]],[[1, 2, 3]],11,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0]],[[3, 3, 1]],0,18) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0]],[[3, 3, 5]],16,25) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]],[[4, 5, 4]],9,19) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],[[3, 4, 1]],25,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1]],[[3, 1, 2]],7,26) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1]],[[1, 2, 6]],8,16) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]],[[5, 10, 5]],16,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1]],[[2, 5, 1], [3, 6, 8], [9, 9, 4]],11,20) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],[[3, 1, 2]],4,22) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]],[[2, 3, 1]],0,19) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]],[[2, 4, 1]],11,17) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]],[[3, 7, 1], [7, 3, 9]],8,13) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0]],[[3, 2, 2]],26,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0]],[[4, 4, 2]],3,23) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]],[[6, 3, 7]],28,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0]],[[1, 8, 10], [1, 1, 9]],20,22) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]],[[1, 1, 2]],5,20) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],[[2, 2, 4]],20,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1]],[[4, 5, 1]],27,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]],[[2, 4, 5]],9,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1]],[[3, 10, 7], [6, 2, 10], [1, 4, 5], [7, 8, 5], [10, 6, 5]],15,28) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]],[[4, 5, 3]],0,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1]],[[2, 5, 2]],16,29) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]],[[4, 7, 1], [6, 9, 4], [10, 5, 4], [4, 6, 6], [4, 10, 6], [1, 8, 5], [9, 2, 6], [6, 4, 4], [1, 9, 3]],22,28) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],[[8, 5, 3]],10,23) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0]],[[2, 1, 4]],26,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0]],[[5, 1, 4], [5, 7, 7], [6, 1, 2], [3, 8, 5], [5, 4, 9], [6, 6, 7]],8,15) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]],[[1, 1, 10], [5, 4, 9], [4, 4, 3], [2, 1, 3], [10, 10, 3]],12,22) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0]],[[5, 1, 2]],8,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],[[4, 6, 10]],30,30) 
assert result is None
result = allocate([[0, 0, 0], [1, 0, 1]],[[8, 2, 4], [5, 4, 3]],21,24) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1]],[[4, 1, 5], [8, 10, 9], [5, 1, 6]],1,5) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0]],[[1, 7, 4]],17,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[5, 2, 4]],16,22) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 1, 1]],9,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],[[4, 1, 2]],10,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0]],[[1, 1, 2]],9,21) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]],[[1, 7, 3]],10,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1]],[[3, 3, 5]],2,27) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]],[[2, 3, 1]],17,25) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]],[[8, 8, 6]],18,28) 
assert result is None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0]],[[2, 2, 1]],16,25) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],[[8, 4, 3]],7,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],[[4, 5, 3]],1,30) 
assert result is not None
result = allocate([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0]],[[7, 7, 7], [10, 4, 5], [3, 1, 2], [2, 4, 2], [5, 9, 4], [9, 4, 3], [5, 2, 1]],2,25) 
assert result is None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]],[[1, 1, 5]],15,30) 
assert result is not None
result = allocate([[1, 0, 0]],[[7, 6, 7]],8,23) 
assert result is None
result = allocate([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1]],[[5, 4, 6]],27,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]],[[1, 5, 4]],0,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0]],[[7, 3, 9], [2, 10, 5], [2, 1, 7]],16,22) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]],[[1, 1, 1]],8,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]],[[3, 4, 7]],24,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],[[2, 1, 4]],26,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]],[[1, 2, 1]],28,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1]],[[1, 3, 3]],19,27) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],[[4, 2, 4]],20,27) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1]],[[2, 3, 1]],21,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]],[[3, 2, 5]],12,25) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]],[[4, 5, 6]],20,26) 
assert result is None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1]],[[2, 4, 10]],20,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]],[[2, 2, 6]],11,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]],[[3, 4, 6]],0,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[7, 3, 5]],26,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1]],[[7, 6, 3], [7, 2, 9], [4, 2, 5]],16,29) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]],[[3, 4, 2]],25,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0]],[[9, 1, 1], [10, 4, 9], [9, 7, 8], [8, 6, 2], [3, 3, 1], [5, 5, 4], [2, 7, 9], [10, 1, 1]],11,15) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],[[3, 3, 4], [8, 5, 8], [8, 10, 1], [10, 7, 4]],19,21) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1]],[[2, 4, 3]],0,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1]],[[9, 4, 9]],20,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0]],[[2, 1, 5]],1,21) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1]],[[3, 3, 2]],0,26) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]],[[2, 4, 3]],15,23) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]],[[3, 2, 1]],9,27) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]],[[1, 1, 7]],24,27) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]],[[3, 1, 5]],0,20) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]],[[7, 6, 5]],23,30) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]],[[3, 3, 3]],0,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1]],[[1, 1, 6]],5,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]],[[3, 2, 4]],21,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1]],[[2, 2, 3]],0,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],[[4, 2, 4]],21,23) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]],[[5, 4, 3]],1,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]],[[1, 1, 3]],4,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],[[3, 1, 6]],4,28) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[1, 1, 3]],25,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]],[[1, 2, 5]],3,26) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0]],[[3, 2, 2]],3,27) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]],[[3, 2, 2]],27,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]],[[6, 4, 3]],23,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]],[[2, 2, 1]],20,29) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[4, 4, 2]],22,30) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[4, 9, 7]],20,26) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1]],[[2, 4, 3]],9,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],[[4, 2, 2]],5,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]],[[10, 7, 2]],9,25) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]],[[4, 2, 5]],19,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0]],[[1, 1, 6]],12,15) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]],[[4, 3, 5]],11,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[3, 6, 3], [2, 1, 5], [3, 1, 8]],30,30) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[4, 3, 5]],14,29) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]],[[2, 2, 7]],16,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1]],[[2, 3, 3]],0,21) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]],[[2, 1, 2]],27,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]],[[2, 3, 4]],9,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1]],[[2, 4, 2]],5,13) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]],[[4, 8, 5]],13,28) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]],[[4, 2, 3]],9,18) 
assert result is not None
result = allocate([[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]],[[1, 1, 7], [10, 5, 10], [8, 3, 3], [10, 2, 9], [5, 1, 4], [4, 8, 6], [5, 3, 7], [1, 6, 5], [5, 6, 8], [1, 7, 4], [7, 6, 8], [7, 7, 6], [1, 4, 8]],15,24) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]],[[2, 2, 1]],10,28) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1]],[[3, 2, 7]],17,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0]],[[1, 4, 2]],20,26) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]],[[4, 3, 5]],12,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]],[[5, 2, 2]],0,23) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1]],[[4, 1, 5]],0,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]],[[5, 5, 3]],29,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1]],[[5, 4, 8]],21,28) 
assert result is None
result = allocate([[1, 0, 0]],[[1, 3, 1]],26,30) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]],[[2, 2, 2]],29,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1]],[[2, 9, 3], [3, 2, 8]],30,30) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],[[2, 2, 1]],0,14) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]],[[1, 2, 3]],8,20) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[4, 8, 10], [10, 7, 6]],30,30) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]],[[1, 1, 5]],30,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[7, 6, 5], [8, 3, 9], [6, 10, 4], [1, 1, 8], [7, 8, 10], [8, 10, 4], [9, 10, 3], [9, 7, 9], [2, 9, 3], [3, 4, 2]],19,30) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0]],[[6, 2, 1]],16,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1]],[[3, 2, 2]],11,24) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],[[5, 3, 3]],17,26) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0]],[[5, 6, 2]],19,21) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1]],[[4, 10, 10], [8, 2, 3]],2,27) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]],[[1, 3, 1]],2,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1]],[[10, 1, 7], [6, 6, 1], [2, 7, 4], [1, 5, 2], [9, 8, 3], [10, 1, 1], [5, 2, 4]],30,30) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]],[[6, 2, 1]],17,22) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1]],[[2, 5, 3]],15,25) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[1, 9, 2]],22,27) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0]],[[2, 4, 5]],0,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]],[[1, 2, 2]],21,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1]],[[6, 7, 5], [6, 10, 5]],24,25) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],[[2, 4, 2]],0,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]],[[1, 2, 6]],23,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],[[2, 5, 2]],16,24) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],[[8, 2, 1]],18,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0]],[[1, 5, 7], [9, 7, 7], [10, 6, 3], [6, 8, 2], [9, 4, 7], [7, 2, 4], [4, 7, 9], [3, 8, 4], [4, 7, 10]],29,29) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]],[[2, 5, 5]],11,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[4, 2, 7]],20,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]],[[3, 2, 3]],0,27) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]],[[1, 1, 1]],2,21) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[6, 5, 6], [10, 8, 4]],20,26) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1]],[[7, 3, 5]],0,27) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]],[[7, 10, 3], [10, 4, 7], [6, 8, 3], [7, 7, 4], [5, 9, 7], [7, 1, 9], [9, 8, 10], [6, 4, 4], [2, 3, 4]],29,30) 
assert result is None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],[[7, 2, 4]],0,26) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1]],[[3, 4, 5], [10, 2, 3], [9, 9, 4], [10, 7, 5], [7, 1, 9]],30,30) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]],[[3, 1, 2]],17,25) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],[[1, 1, 2]],18,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]],[[2, 4, 3]],19,29) 
assert result is not None
result = allocate([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]],[[3, 9, 10], [1, 8, 3], [3, 2, 2], [8, 6, 8], [10, 3, 4], [4, 6, 3], [4, 8, 9], [1, 1, 7], [9, 4, 4], [3, 3, 1], [10, 3, 3], [3, 5, 3]],26,30) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]],[[1, 5, 3]],3,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0]],[[6, 6, 3]],12,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],[[4, 1, 1]],6,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],[[2, 3, 1]],18,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],[[7, 4, 2]],13,17) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0]],[[1, 3, 3]],25,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[1, 3, 4]],12,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]],[[6, 4, 10]],24,27) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]],[[2, 2, 8]],27,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],[[4, 10, 3]],17,29) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]],[[6, 1, 3]],0,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]],[[1, 2, 2]],28,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[3, 2, 2]],22,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[3, 1, 1]],7,20) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0]],[[9, 6, 4]],0,29) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]],[[2, 1, 2]],9,29) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0]],[[1, 2, 1]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1]],[[3, 10, 5]],23,25) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],[[8, 3, 7], [1, 1, 6], [8, 1, 8]],19,24) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]],[[2, 8, 1]],0,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[2, 1, 2]],11,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0]],[[6, 1, 3]],13,13) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0]],[[2, 8, 7], [2, 2, 9], [6, 5, 4], [10, 3, 1]],15,25) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1]],[[7, 9, 9], [4, 3, 3], [8, 7, 5]],28,28) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 0]],[[6, 5, 7], [4, 9, 9], [7, 6, 9]],30,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]],[[2, 4, 6], [5, 8, 3]],16,16) 
assert result is None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]],[[5, 6, 2]],16,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[2, 1, 4]],0,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0]],[[1, 8, 3]],24,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],[[1, 3, 3]],12,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]],[[8, 1, 5]],26,26) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]],[[3, 3, 6]],5,30) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1]],[[8, 4, 6]],22,26) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1]],[[9, 3, 3]],14,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],[[1, 7, 2]],2,26) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1]],[[8, 2, 4]],20,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]],[[4, 1, 9], [2, 7, 6], [4, 4, 9], [1, 4, 10]],8,8) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]],[[1, 7, 1]],0,17) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1]],[[1, 2, 2]],17,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]],[[5, 2, 7]],25,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],[[5, 4, 5]],8,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]],[[7, 1, 1]],8,25) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],[[2, 6, 10], [1, 9, 3], [4, 2, 2], [9, 2, 10], [7, 3, 5], [9, 6, 5], [6, 4, 9], [4, 10, 5], [7, 9, 4], [2, 9, 4], [5, 1, 7], [10, 10, 1], [5, 8, 8], [1, 7, 2], [5, 6, 8], [8, 5, 3], [9, 4, 7]],29,29) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 0]],[[2, 2, 5], [9, 9, 7]],4,4) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]],[[1, 2, 4], [1, 2, 3], [10, 7, 9], [2, 4, 5], [9, 7, 7], [5, 6, 10]],4,5) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1]],[[1, 4, 4]],0,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1]],[[4, 9, 6]],4,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1]],[[4, 4, 4], [1, 1, 10], [8, 7, 3], [3, 5, 10], [2, 10, 9]],20,29) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],[[3, 4, 1]],13,24) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]],[[7, 2, 6]],17,29) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[1, 3, 2]],2,16) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]],[[2, 1, 8], [3, 10, 8], [8, 6, 5], [1, 6, 7], [4, 2, 8], [6, 6, 8], [10, 10, 7], [8, 5, 10], [2, 1, 3], [8, 3, 2], [10, 1, 8], [3, 3, 3], [1, 8, 8], [5, 1, 7], [10, 1, 1], [10, 10, 9]],21,29) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],[[1, 1, 1]],22,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]],[[2, 5, 1]],13,21) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1]],[[9, 3, 7]],0,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]],[[3, 2, 4]],16,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0]],[[8, 4, 1]],8,29) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],[[3, 7, 8]],15,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]],[[3, 2, 9]],0,29) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0]],[[3, 3, 6], [5, 3, 7], [2, 2, 5], [3, 5, 8], [5, 5, 4]],8,29) 
assert result is None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[6, 4, 6]],12,27) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 0]],[[1, 1, 1]],30,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0]],[[5, 7, 7]],8,23) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[1, 1, 2]],24,27) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],[[6, 4, 2]],10,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1]],[[5, 7, 3]],11,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]],[[1, 1, 2]],9,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]],[[5, 1, 2]],3,27) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0]],[[6, 6, 3], [5, 1, 1], [4, 4, 8], [8, 3, 1], [4, 5, 1], [1, 10, 8], [4, 5, 3], [4, 1, 10], [10, 8, 3], [3, 5, 4], [7, 7, 4]],8,18) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]],[[3, 9, 4], [8, 10, 9], [8, 2, 9], [2, 10, 9], [8, 10, 7], [1, 9, 8]],12,12) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1]],[[3, 3, 4]],30,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],[[6, 1, 8]],19,24) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]],[[3, 2, 3]],11,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0]],[[2, 2, 3]],8,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0]],[[3, 2, 3]],30,30) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]],[[1, 9, 2]],9,26) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1]],[[5, 8, 5]],10,22) 
assert result is None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1]],[[6, 3, 2]],0,28) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]],[[5, 4, 1]],0,14) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]],[[1, 4, 3]],0,26) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0]],[[9, 4, 5]],0,27) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]],[[2, 4, 4]],25,26) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0]],[[5, 4, 3]],0,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[1, 2, 8], [4, 9, 4], [4, 4, 6], [9, 1, 10]],28,30) 
assert result is None
result = allocate([[0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1]],[[4, 2, 2]],11,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]],[[1, 4, 4]],11,17) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]],[[3, 3, 1]],14,22) 
assert result is not None
result = allocate([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1]],[[1, 6, 2]],8,23) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1]],[[1, 1, 1]],3,24) 
assert result is not None
result = allocate([[0, 0, 0]],[[3, 1, 6]],7,14) 
assert result is None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]],[[4, 3, 6]],30,30) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1]],[[4, 2, 4]],23,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[1, 1, 8]],10,22) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],[[3, 1, 2]],20,26) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0]],[[4, 10, 4]],4,24) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]],[[7, 1, 4]],3,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]],[[3, 4, 3]],0,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0]],[[2, 2, 3]],26,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]],[[3, 2, 1]],20,24) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1]],[[3, 5, 1]],15,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]],[[4, 4, 4]],17,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],[[2, 1, 1]],8,21) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1]],[[3, 6, 5], [6, 5, 1], [5, 10, 10], [1, 8, 5], [10, 3, 5], [10, 3, 8], [3, 6, 7], [4, 9, 4], [9, 2, 4], [3, 1, 1], [3, 10, 7], [1, 3, 3], [6, 6, 1], [7, 7, 4]],7,23) 
assert result is None
result = allocate([[0, 0, 0], [0, 0, 0], [0, 1, 0]],[[7, 10, 10], [8, 4, 7]],29,30) 
assert result is None
result = allocate([[0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]],[[5, 1, 5]],0,29) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]],[[5, 10, 1]],0,25) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0]],[[1, 5, 2]],0,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]],[[1, 1, 6]],18,21) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1]],[[7, 2, 10], [10, 2, 6], [5, 6, 8], [5, 1, 3], [10, 8, 5], [7, 5, 4]],4,17) 
assert result is None
result = allocate([[0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0]],[[4, 4, 2]],13,24) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0]],[[2, 7, 3]],3,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]],[[9, 7, 7]],30,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0]],[[10, 5, 4]],8,14) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]],[[1, 2, 2], [3, 4, 1]],30,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1]],[[2, 1, 2]],20,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1]],[[2, 4, 1]],3,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1]],[[1, 1, 4]],1,25) 
assert result is not None
result = allocate([[1, 1, 1]],[[9, 10, 7]],7,14) 
assert result is None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]],[[2, 2, 6]],9,23) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]],[[4, 3, 1]],12,25) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0]],[[2, 1, 3]],1,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],[[5, 2, 2]],13,29) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1]],[[5, 5, 3]],20,30) 
assert result is not None
result = allocate([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]],[[1, 1, 2]],30,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1]],[[3, 3, 5]],29,30) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1]],[[4, 3, 3]],8,22) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]],[[4, 1, 9], [7, 1, 4], [1, 7, 1], [9, 10, 3], [5, 3, 7], [9, 9, 2]],15,19) 
assert result is None
result = allocate([[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]],[[3, 9, 6]],28,30) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1]],[[4, 4, 5], [4, 5, 3]],0,9) 
assert result is None
result = allocate([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0]],[[5, 6, 5], [8, 4, 9], [1, 3, 6], [7, 6, 10]],3,22) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1]],[[9, 3, 2]],13,27) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0]],[[1, 3, 2]],21,28) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]],[[2, 2, 6]],16,28) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 1]],[[1, 6, 8], [8, 3, 4], [6, 10, 10], [9, 4, 1]],26,27) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1]],[[1, 1, 8]],20,30) 
assert result is not None
result = allocate([[1, 1, 0]],[[9, 2, 5]],15,18) 
assert result is None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1]],[[6, 4, 2]],18,20) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],[[9, 6, 7], [2, 7, 7], [10, 9, 3], [2, 8, 5], [2, 3, 7]],15,30) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]],[[1, 1, 3]],0,25) 
assert result is not None
result = allocate([[0, 1, 0]],[[4, 8, 4]],27,29) 
assert result is None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]],[[6, 2, 1]],0,24) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]],[[1, 3, 1]],5,25) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]],[[6, 7, 4], [7, 3, 7], [7, 1, 3]],27,30) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]],[[6, 2, 9]],7,28) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],[[8, 6, 5], [4, 2, 7], [2, 8, 6], [9, 4, 5], [10, 9, 8], [4, 1, 1], [4, 4, 2], [2, 2, 8], [7, 7, 7], [8, 9, 7], [10, 3, 10], [7, 7, 5], [7, 7, 4], [6, 10, 9], [1, 7, 9], [3, 4, 6], [5, 1, 9], [4, 4, 4], [1, 8, 8], [1, 7, 7], [5, 4, 8], [6, 5, 1], [7, 10, 7]],14,28) 
assert result is None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1]],[[9, 2, 4], [2, 4, 1], [10, 3, 7], [5, 7, 1]],14,17) 
assert result is None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1]],[[8, 5, 6]],16,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]],[[4, 4, 1]],14,20) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1]],[[6, 5, 10]],0,29) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]],[[3, 1, 3]],4,24) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 1], [1, 0, 0]],[[2, 8, 1], [10, 8, 9], [7, 6, 1], [8, 9, 8], [2, 7, 1], [4, 7, 1]],30,30) 
assert result is None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0]],[[3, 2, 3]],13,23) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]],[[6, 1, 2]],4,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0]],[[3, 3, 1]],2,23) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1]],[[6, 4, 9], [1, 6, 10], [2, 2, 2], [5, 4, 3]],19,26) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]],[[2, 3, 8]],16,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1]],[[1, 1, 3]],13,30) 
assert result is not None
result = allocate([[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]],[[1, 3, 4]],15,27) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0]],[[1, 6, 3]],1,19) 
assert result is not None
result = allocate([[1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],[[6, 2, 4]],0,22) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0]],[[1, 3, 1]],23,27) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1]],[[5, 2, 3]],11,28) 
assert result is not None
result = allocate([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0]],[[2, 3, 1]],19,30) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],[[2, 1, 2]],0,13) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]],[[2, 3, 2]],19,24) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0]],[[4, 1, 5]],11,26) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],[[6, 1, 3]],5,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1]],[[7, 2, 8]],22,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]],[[2, 1, 3]],4,29) 
assert result is not None
result = allocate([[1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]],[[4, 1, 2]],15,26) 
assert result is not None
result = allocate([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0]],[[3, 9, 3]],16,26) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],[[2, 2, 1]],20,25) 
assert result is not None
result = allocate([[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0]],[[10, 2, 7], [5, 6, 4], [7, 7, 5], [8, 10, 10], [4, 6, 6], [4, 2, 9], [5, 9, 2]],11,25) 
assert result is None
result = allocate([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]],[[2, 6, 2]],13,26) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1]],[[7, 1, 5], [4, 2, 1]],21,22) 
assert result is None
result = allocate([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]],[[8, 7, 5], [5, 6, 8], [3, 1, 3], [7, 6, 6], [10, 5, 5], [6, 6, 5], [4, 3, 6], [6, 1, 5], [6, 5, 10], [5, 1, 8], [3, 8, 8], [9, 1, 3], [4, 6, 3], [5, 2, 5]],4,19) 
assert result is None
result = allocate([[1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],[[4, 3, 8]],5,23) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]],[[7, 1, 5], [1, 10, 9], [9, 4, 6], [6, 6, 7]],0,29) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0]],[[2, 3, 1]],4,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]],[[5, 3, 6], [10, 2, 2], [10, 3, 6], [4, 9, 10], [1, 7, 9]],28,30) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0]],[[5, 3, 4], [2, 1, 1]],25,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1]],[[1, 1, 8]],1,30) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]],[[1, 1, 1]],6,19) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]],[[1, 7, 8]],0,25) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]],[[1, 2, 4]],2,27) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1]],[[4, 1, 1]],30,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],[[8, 3, 8]],26,29) 
assert result is not None
result = allocate([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]],[[4, 9, 4], [10, 7, 1], [8, 3, 3], [9, 2, 9], [7, 3, 6]],6,7) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]],[[2, 3, 5]],12,22) 
assert result is not None
result = allocate([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[5, 2, 6]],9,21) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]],[[2, 1, 1]],2,29) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]],[[7, 3, 10]],10,11) 
assert result is None
result = allocate([[1, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],[[7, 7, 4]],30,30) 
assert result is not None
result = allocate([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]],[[2, 2, 1]],12,21) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]],[[1, 5, 1]],0,18) 
assert result is not None
result = allocate([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]],[[9, 8, 8], [9, 7, 4], [10, 4, 3], [4, 9, 2], [9, 9, 8], [2, 8, 10], [3, 2, 1]],1,4) 
assert result is None
result = allocate([[0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]],[[1, 1, 4]],17,27) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],[[5, 1, 2]],15,26) 
assert result is not None
result = allocate([[0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[1, 1, 1]],0,15) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],[[10, 6, 4]],11,29) 
assert result is not None
result = allocate([[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 1]],[[2, 1, 1]],2,13) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]],[[2, 7, 5]],30,30) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],[[1, 3, 10], [6, 7, 2]],19,27) 
assert result is None
result = allocate([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],[[2, 6, 1]],7,28) 
assert result is not None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]],[[2, 3, 2]],20,27) 
assert result is not None
result = allocate([[0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0]],[[1, 3, 4]],9,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],[[2, 4, 1]],25,28) 
assert result is not None
result = allocate([[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0]],[[2, 10, 3], [3, 4, 1], [6, 4, 5], [5, 1, 9], [6, 8, 2]],1,27) 
assert result is None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1]],[[2, 2, 1]],0,29) 
assert result is not None
result = allocate([[0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]],[[2, 7, 5]],14,30) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]],[[1, 2, 1]],4,18) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]],[[2, 5, 1]],20,26) 
assert result is not None
result = allocate([[1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]],[[1, 3, 3]],10,27) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]],[[5, 3, 5]],14,28) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],[[1, 1, 9], [2, 2, 2], [1, 1, 6], [7, 6, 5], [1, 3, 4], [8, 9, 4], [4, 4, 1], [1, 2, 1], [8, 6, 8], [2, 6, 6]],8,11) 
assert result is None
result = allocate([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1]],[[1, 1, 2]],6,22) 
assert result is not None
result = allocate([[0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]],[[5, 2, 5]],19,28) 
assert result is not None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0]],[[4, 1, 4]],12,27) 
assert result is not None
result = allocate([[0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]],[[7, 3, 8]],27,30) 
assert result is None
result = allocate([[0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0]],[[2, 2, 1]],5,9) 
assert result is not None
result = allocate([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1]],[[5, 10, 8], [10, 3, 5], [8, 1, 10], [1, 7, 5], [4, 3, 3], [3, 4, 10]],1,24) 
assert result is None
result = allocate([[0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]],[[7, 6, 1], [5, 5, 6], [1, 8, 2], [4, 6, 6], [3, 6, 9], [1, 8, 9], [1, 8, 4], [7, 4, 7], [6, 1, 2]],5,30) 
assert result is None
result = allocate([[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]],[[1, 7, 1]],14,30) 
assert result is not None
result = allocate([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]],[[3, 2, 1]],6,30) 
assert result is not None
result = allocate([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1]],[[5, 10, 2]],0,25) 
assert result is not None
result = allocate([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]],[[1, 5, 5]],10,25) 
assert result is not None
result = allocate([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]],[[2, 4, 1]],12,24) 
assert result is not None
result = allocate([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1]],[[2, 1, 4]],4,18) 
assert result is not None
result = allocate([[1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0]],[[7, 1, 4]],0,28) 
assert result is not None