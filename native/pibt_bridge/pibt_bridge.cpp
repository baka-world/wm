// Thin input/output adapter around the unmodified upstream PIBT class.
// No priority inheritance or backtracking logic is implemented here.
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <pibt.hpp>
#include <problem.hpp>

namespace {
// Present exactly the currently legal directed adjacency to the upstream
// Graph interface. Locked vertices are removed before this function is called.
class ObservableGraph final : public Graph {
 public:
  ObservableGraph(int n, const int* xy, const int* offsets, const int* edges) {
    V.resize(n, nullptr);
    for (int i = 0; i < n; ++i) V[i] = new Node(i, xy[2*i], xy[2*i+1]);
    for (int i = 0; i < n; ++i)
      for (int j = offsets[i]; j < offsets[i+1]; ++j) V[i]->neighbor.push_back(V[edges[j]]);
  }
  bool existNode(int id) const override { return id >= 0 && id < static_cast<int>(V.size()); }
  Node* getNode(int id) const override { return existNode(id) ? V[id] : nullptr; }
  int dist(const Node* const, const Node* const) const override { return 0; }
};

// The upstream copy constructor is the public route for a new MAPF instance,
// but inherits graph and agent count from its parent. Override protected
// Problem fields only; the original upstream class and solver are unchanged.
class CurrentInstance final : public MAPF_Instance {
 public:
  CurrentInstance(MAPF_Instance* parent, Graph* graph, std::mt19937* rng,
                  Config starts, Config goals)
      : MAPF_Instance(parent, starts, goals, 500, 1) {
    G=graph; MT=rng; num_agents=static_cast<int>(starts.size());
  }
};
}

extern "C" int wm_pibt_step(int nodes, int edges_count, const int* xy,
                            const int* offsets, const int* edges,
                            int agents, const int* starts, const int* goals,
                            const int* distances, unsigned int seed, int* next) {
  if (nodes <= 0 || edges_count < 0 || agents <= 0 || agents > nodes ||
      !xy || !offsets || !edges || !starts || !goals || !distances || !next ||
      offsets[0] != 0 || offsets[nodes] != edges_count) return 1;
  for (int i=0;i<nodes;++i)
    if (offsets[i] > offsets[i+1] || offsets[i] < 0) return 2;
  for (int i=0;i<edges_count;++i) if (edges[i]<0 || edges[i]>=nodes) return 3;
  std::vector<int> used(nodes,0);
  for (int i=0;i<agents;++i) {
    if (starts[i]<0 || starts[i]>=nodes || goals[i]<0 || goals[i]>=nodes || used[starts[i]]++) return 4;
  }
  try {
    // The original MAPF instance parser initializes the parent. The adapter
    // copy below uses the exact live graph and starts/goals instead.
    static MAPF_Instance parent(WM_SAMPLE_INSTANCE);
    ObservableGraph graph(nodes,xy,offsets,edges);
    std::mt19937 rng(seed);
    Config initial,goal;
    for (int i=0;i<agents;++i) {
      initial.push_back(graph.getNode(starts[i]));
      goal.push_back(graph.getNode(goals[i]));
    }
    CurrentInstance instance(&parent,&graph,&rng,initial,goal);
    // Official PIBT initializes its distance table using max_timestep. For a
    // single requested step, provide exact current-graph distances through the
    // upstream public API; otherwise its horizon=1 would truncate distances.
    std::vector<std::vector<int>> table(agents,std::vector<int>(nodes));
    for (int i=0;i<agents;++i)
      for (int j=0;j<nodes;++j) table[i][j]=distances[i*nodes+j];
    PIBT solver(&instance);
    solver.setDistanceTable(&table);
    solver.solve();
    Plan plan=solver.getSolution();
    if (plan.size()!=2 || !plan.validate(initial)) return 5;
    for (int i=0;i<agents;++i) next[i]=plan.get(1,i)->id;
    return 0;
  } catch (...) { return 6; }
}
