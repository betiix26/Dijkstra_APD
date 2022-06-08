
//Implementation of Dijkstra's algorithm in C++ which finds 
//the shortest path from a start node to every other node in a weighted graph.
//Time complexity: O(n^2)
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

//#define MAXV 1000

int noNodes;

fstream in_file;
fstream out_file;

class EdgeNode {
public:
    int key;
    int weight;
    EdgeNode* next;
    EdgeNode(int, int);
};

EdgeNode::EdgeNode(int key, int weight) {
    this->key = key;
    this->weight = weight;
    this->next = NULL;
}

class Graph {
    bool directed;
public:
    vector<EdgeNode*> edges; // { new EdgeNode[noNodes]{} };
    Graph(bool);
    ~Graph();
    void insert_edge(int, int, int, bool);
    void print();
};

Graph::Graph(bool directed) {
    this->directed = directed;
    for (int i = 0; i < (noNodes); i++) {
        edges.push_back(NULL);
    }
}

Graph::~Graph() {/*Not necesary*/ }

void Graph::insert_edge(int x, int y, int weight, bool directed) {
    if (x > 0 && x < (noNodes) && y > 0 && y < (noNodes)) {
        EdgeNode* edge = new EdgeNode(y, weight);
        edge->next = this->edges[x];
        this->edges[x] = edge;
        if (!directed) {
            insert_edge(y, x, weight, true);
        }
    }
}

void Graph::print() {
    for (int v = 0; v < (noNodes); v++) {
        if (this->edges[v] != NULL) {
            cout << "Vertex " << v << " has neighbors: " << endl;
            EdgeNode* curr = this->edges[v];
            while (curr != NULL) {
                cout << curr->key << endl;
                curr = curr->next;
            }
        }
    }
}

void init_vars(vector<bool> &discovered, vector<int> &distance, vector<int> &parent) {
    for (int i = 0; i < (noNodes); i++) {
        discovered.push_back(false);
        distance.push_back(std::numeric_limits<int>::max());
        parent.push_back(-1);
    }
}

void dijkstra_shortest_path(Graph* g, vector<int> &parent, vector<int> &distance, int start) {

    vector<bool> discovered;
    EdgeNode* curr;
    int v_curr;
    int v_neighbor;
    int weight;
    int smallest_dist;

    init_vars(discovered, distance, parent);

    distance[start] = 0;
    v_curr = start;

    while (discovered[v_curr] == false) {

        discovered[v_curr] = true;
        curr = g->edges[v_curr];

        while (curr != NULL) {

            v_neighbor = curr->key;
            weight = curr->weight;

            if ((distance[v_curr] + weight) < distance[v_neighbor]) {
                distance[v_neighbor] = distance[v_curr] + weight;
                parent[v_neighbor] = v_curr;
            }
            curr = curr->next;
        }

        //set the next current vertex to the vertex with the smallest distance
        smallest_dist = std::numeric_limits<int>::max();
        for (int i = 0; i < (noNodes); i++) {
            if (!discovered[i] && (distance[i] < smallest_dist)) {
                v_curr = i;
                smallest_dist = distance[i];
            }
        }
    }
}

void print_shortest_path(int v, vector<int> &parent) {
    if (v >= 0 && v < (noNodes) && parent[v] != -1) {
        cout << parent[v] << " ";
        print_shortest_path(parent[v], parent);
    }
}

void print_distances(int start, vector<int> &distance) {
    for (int i = 0; i < (noNodes); i++) {
            cout << "Shortest distance from " << start << " to " << i << " is: " << distance[i] << endl;
        
    }
}

string int_to_string(int x) {
    stringstream ss;
    ss << x;
    return ss.str();
}

int main() {

    string testFile = "test{i}.in";
    for (int i = 0; i <= 9; i++) {

        in_file.open("test" + int_to_string(i + 1));

        int nrNoduri, nrMuchii;
        in_file >> nrNoduri >> nrMuchii;

        noNodes = nrNoduri;

        Graph* g = new Graph(false);
        vector<int> parent;
        vector<int> distance;

        int start = 1;

        int a, b, c;
        for (int muc = 0; muc < nrMuchii; muc++) {
            in_file >> a >> b >> c;
            g->insert_edge(a, b, c, false);
        }

        dijkstra_shortest_path(g, parent, distance, start);
        
        print_distances(start, distance);

        cout << endl << endl << "-------------------------------- TEST: " << i + 1 << " finished-----------------------" << endl << endl;
        delete g;
        in_file.close();
    }

    return 0;
}