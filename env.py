from geoclidean_framework.geoclidean_env_euclid import *
concept = [
    'l1 = line(p1(), p2())',
    'c1* = circle(p1(), p2())',
    'c2* = circle(p2(), p1())',
    'l2 = line(p1(), p3(c1, c2))',
    'l3 = line(p2(), p3()))'
]


for i in range(3):
    generate_concept(concept, mark_points=False, show_plots=True)