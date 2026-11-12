"""Static cut-vertex diagnosis is descriptive, never proof of causal failure."""
import unittest
from experiments.world_model_integration.topology_pr56.pr56_readonly_stall import articulation_cells


class PublicMapReadonlyStallTests(unittest.TestCase):
    def test_known_single_gateway(self):
        graph={"walkable_cells":[[0,0],[1,0],[2,0]],
               "directed_map_edges":[[0,1],[1,0],[1,2],[2,1]]}
        self.assertEqual(articulation_cells(graph),[[1,0]])

    def test_alternate_route_removes_single_gateway(self):
        graph={"walkable_cells":[[0,0],[1,0],[2,0]],
               "directed_map_edges":[[0,1],[1,0],[1,2],[2,1],[2,0],[0,2]]}
        self.assertEqual(articulation_cells(graph),[])

    def test_bottleneck_is_not_equivalent_to_true_episode_timeout(self):
        # The logic only analyzes a static public grid, NEVER infers
        # a time-limited factual episode or an unobserved external outage.
        graph={"walkable_cells":[[0,0],[1,0]],
               "directed_map_edges":[[0,1],[1,0]]}
        self.assertEqual(articulation_cells(graph),[])


if __name__=="__main__":
    unittest.main()
