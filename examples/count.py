"""PRIMs model converted from count.prims by pyprims.convert."""
from pyprims import Task

task = Task('count',
            initial_skills=['count'],
            default_activation=1.0,
            ol=True,
            rt=-2.0,
            lf=0.2,
            default_operator_self_assoc=0.0,
            egs=0.05,
            retrieval_reinforces=True)

skill_count = task.skill('count')
skill_count.operator('start-count', '''
         V1 <> nil  // There has to be an input with the starting number
         WM1 = nil  // Working Memory should be empty
        ==>
         V1 -> WM1 // Copy the start number to working memory
         count-fact -> RT1 // Start retrieving the next number
         V1 -> RT2
         say -> AC1 // Say the current number
         V1 -> AC2
    ''')
skill_count.operator('iterate', '''
         RT2 = WM1
         V2 <> WM1
        ==>
         RT3 -> WM1
         count-fact -> RT1
         RT3 -> RT2
         say -> AC1
         RT3 ->AC2
    ''')
skill_count.operator('final', '''
         V2 = WM1
        ==>
         say -> AC1
         stop -> AC2
         nil -> G1
    ''')

task.facts([
    ('cf1', 'count-fact', 'one', 'two'),
    ('cf2', 'count-fact', 'two', 'three'),
    ('cf3', 'count-fact', 'three', 'four'),
    ('cf4', 'count-fact', 'four', 'five'),
    ('cf5', 'count-fact', 'five', 'six'),
    ('cf6', 'count-fact', 'six', 'seven'),
    ('cf7', 'count-fact', 'seven', 'eight'),
    ('cf8', 'count-fact', 'eight', 'nine'),
    ('cg9', 'count-fact', 'nine', 'ten'),
])

task.action('say', latency=0.3, noise=0.1, distribution='uniform', output='Saying')

@task.script
def script(m):
    m.set_graph_title("Reaction time")
    digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    start = m.random(3)
    end = start + (1 + m.random(3))
    m.print("Counting from", digits[start], "to", digits[end])
    m.screen(digits[start], digits[end])
    m.run_until_action("say", "stop")
    m.issue_reward()
    m.trial_end()
