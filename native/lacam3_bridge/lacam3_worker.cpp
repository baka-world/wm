// Thin process adapter: construct the upstream Graph/Instance and call solve().
// The official joint-search implementation is not copied or changed.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <vector>

#include <lacam.hpp>

int main() {
  int nodes=0,arcs=0,agents=0,seed=0,budget_ms=0;
  if (!(std::cin>>nodes>>arcs>>agents>>seed>>budget_ms) || nodes<=0 || arcs<0 ||
      agents<=0 || agents>nodes || budget_ms<=0) return 2;
  Graph graph;
  graph.width=nodes; graph.height=1;
  for(int i=0;i<nodes;++i) {
    auto* vertex=new Vertex(i,i,i,0);
    graph.V.push_back(vertex); graph.U.push_back(vertex);
  }
  std::set<std::pair<int,int>> edges;
  for(int i=0;i<arcs;++i) {
    int a,b;if(!(std::cin>>a>>b) || a<0 || b<0 || a>=nodes || b>=nodes || a==b) return 3;
    edges.insert({a,b});
  }
  for(auto [a,b]:edges) if(!edges.count({b,a})) return 4;
  for(auto [a,b]:edges) graph.V[a]->neighbor.push_back(graph.V[b]);
  Config starts,goals;std::set<int> used_starts,used_goals;
  for(int i=0;i<agents;++i) {
    int s,g;if(!(std::cin>>s>>g) || s<0 || g<0 || s>=nodes || g>=nodes ||
                !used_starts.insert(s).second || !used_goals.insert(g).second) return 5;
    starts.push_back(graph.V[s]); goals.push_back(graph.V[g]);
  }
  Instance instance(&graph,starts,goals,agents);
  // Fixed upstream public settings: first complete solution, single-threaded;
  // no optimization/refiner and no custom search algorithm.
  Planner::FLG_STAR=false;
  Planner::FLG_MULTI_THREAD=false;
  Planner::FLG_REFINER=false;
  DistTable::FLG_MULTI_THREAD=false;
  Deadline deadline(budget_ms);
  auto solution=solve(instance,0,&deadline,seed);
  struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
  double cpu_ms=(usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)*1000.0+
                 (usage.ru_utime.tv_usec+usage.ru_stime.tv_usec)/1000.0;
  long hwm_kb=0;std::ifstream proc("/proc/self/status");std::string line;
  while(std::getline(proc,line)) {
    if(line.rfind("VmHWM:",0)==0) {std::istringstream stream(line.substr(6));stream>>hwm_kb;break;}
  }
  if(solution.empty()) {std::cout<<"NO_SOLUTION "<<hwm_kb<<" "<<cpu_ms<<"\n";return 0;}
  if(!is_feasible_solution(instance,solution)) {std::cout<<"INVALID\n";return 6;}
  std::cout<<"OK "<<solution.size()<<" "<<agents<<" "<<hwm_kb<<" "<<cpu_ms<<"\n";
  for(const auto& config:solution) {
    for(auto* v:config) std::cout<<v->id<<" ";
    std::cout<<"\n";
  }
  return 0;
}
