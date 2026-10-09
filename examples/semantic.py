"""PRIMs model converted from semantic.prims by pyprims.convert."""
from pyprims import Task

task = Task('semantic',
            initial_skills=['semantic'],
            ol=True,
            rt=-2.0,
            lf=0.2,
            default_operator_self_assoc=0.0,
            default_activation=1.0)

skill_semantic = task.skill('semantic')
skill_semantic.operator('start-semantic', '''
           V1 <> nil
           WM1 = nil
        ==>
           V1 -> WM1
           property -> RT1
           V1 -> RT2
           subvocalize -> AC1
           V1 -> AC2
    ''')
skill_semantic.operator('move-up-tree', '''
        RT2 = WM1
        V2 <> WM1
        ==>
        RT3 -> WM1
        property -> RT1
        RT3 -> RT2
        subvocalize -> AC1
        RT3 -> AC2
    ''')
skill_semantic.operator('say-yes', '''
        V2 = WM1
        ==>
        say -> AC1
        yes -> AC2
        stop -> G1
    ''')
skill_semantic.operator('say-no', '''
        V2 <> WM1
        RT1 = error
        ==>
        say -> AC1
        no -> AC2
        nil -> G1
    ''')

task.action('subvocalize', latency=0.2, noise=0.08, distribution='uniform', output='Subvocalizing')
task.action('say', latency=0.3, noise=0.1, distribution='uniform', output='Saying')
task.facts([
    ('sem1', 'property', 'lion', 'mammal'),
    ('sem2', 'property', 'mammal', 'animal'),
    ('sem3', 'property', 'animal', 'living-thing'),
    ('sem4', 'property', 'plant', 'living-thing'),
    ('sem5', 'property', 'tulip', 'plant'),
    ('sem6', 'property', 'bird', 'animal'),
    ('sem7', 'property', 'tweety', 'bird'),
    ('sem8', 'property', 'robin', 'bird'),
])


@task.script
def script(m):
    m.set_graph_title("Reaction time")
    pairs = [["lion", "living-thing", "yes"], ["tulip", "animal", "no"], ["mammal", "living-thing", "yes"], ["lion", "plant", "no"]]
    pairno = m.random(m.length(pairs))
    pair = pairs[pairno]
    m.print("Is", pair[0], "a", pair[1], "?")
    m.screen(pair[0], pair[1])
    m.run_until_action("say")
    m.issue_reward()
    m.trial_end()
