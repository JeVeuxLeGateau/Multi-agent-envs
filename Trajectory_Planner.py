import graph_scene as gs
import Probability_field as pf


class Trajectory_Planner:
    def __init__(self):
        self.buildings = 3
        self.initial_scene = gs.graph_scene()
        self.p_f = pf.Probability_field()


t_p = Trajectory_Planner()

# Create Shapes
t_p.initial_scene.create_sphere((500, 500, 500), 100, "red")
shape1 = t_p.initial_scene.shapes
values = shape1[0][0]

t_p.p_f.add_radar(values, 1, (500, 500, 500), 100)

t_p.initial_scene.create_rectangle((0, 0, 0), 200, 200, 200, "blue")
t_p.initial_scene.create_line((1000, 1000, 1000), (0, 0, 0))
t_p.initial_scene.create_rectangle((800, 800, 800), 200, 200, 200, "blue")

# Render Shapes
t_p.initial_scene.render_shapes()
