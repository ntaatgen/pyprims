"""PRIMs model converted from count-rev4.prims by pyprims.convert."""
from pyprims import Task

task = Task('count',
            initial_skills=['control'],
            default_activation=1.0,
            ol=True,
            rt=-2.0,
            lf=0.2,
            egs=0.05,
            retrieval_reinforces=True)

skill_control = task.skill('control')
skill_control.operator('get-first-step', '''
        	G2 = nil
        	RT1 = nil
        ==>
        	step -> RT1
        	one -> RT2
    ''', activation=0.0)
skill_control.operator('implement-step', '''
        	G2 = nil
        	RT1 = step
        ==>
        	RT3 -> G2
        	RT4 -> *step
    ''')
skill_control.operator('next-step', '''
        	G2 = nil
        	RT1 = nil
        ==>
        	step -> RT1
        	*step -> RT2
    ''')
skill_control.operator('last-step', '''
        	G2 = done
        ==>
        	done -> AC1
        	nil -> G1
    ''')
skill_control.operator('fail-step', '''
        	G2 = fail
        	RT1 = nil
        ==>
        	fail-step -> RT1
        	*step -> RT2
    ''')

task.facts([
    ('st1', 'step', 'one', 'read', 'two'),
    ('st2', 'step', 'two', 'iterate', 'three'),
    ('st3', 'step', 'three', 'done', 'four'),
])

skill_read = task.skill('read')
skill_read.operator('read-for-two-items', '''
        	V1 <> nil
        	V2 <> nil
        	WM1 = nil
        ==>
        	V1 -> WM1
        	V2 -> WM2
    ''')
skill_read.operator('switch-to-next-skill', '''
        	V1 = WM1
        ==>
        	nil -> G2
    ''')

skill_iterate = task.skill('iterate')
skill_iterate.operator('start-iteration', '''
         WM3 = nil  // Slot we use for iterator is empty
        ==>
         WM1 -> WM3	
         *sub-skill -> G2
    ''')
skill_iterate.operator('do-sub-skill', '''
        RT1 = *fact-type
        ==>
        RT3 -> WM3
        *sub-skill -> G2
    ''')
skill_iterate.operator('retrieve-next', '''
         RT1 = nil
         WM3 <> nil
         WM2 <> WM3
        ==>
         *fact-type -> RT1
         WM3 -> RT2
    ''')
skill_iterate.operator('final', '''
         WM2 = WM3
         RT1 = nil
        ==>
         nil -> G2
    ''')
skill_iterate.operator('final-fail', '''
        WM2 <> WM3
        RT1 = error
        ==>
        fail -> G2
    ''')

skill_do_action = task.skill('do-action')
skill_do_action.operator('do-action-on-WM3', '''
        	WM3 <> nil
        ==>
        	*action -> AC1
        	WM3 -> AC2
        	*main-skill -> G2
    ''')

skill_respond = task.skill('respond')
skill_respond.operator('carry-out-action', '''
        G1 <> nil // Operators require at least one condition
        ==>
        *action -> AC1
        *arg -> AC2
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
    m.add_binding("sub-skill", "do-action")
    m.add_binding("fact-type", "count-fact")
    m.add_binding("main-skill", "iterate")
    m.add_binding("action", "say")
    m.instantiate_skill("read", "read-one", ["next-skill", "iterate-one"])
    m.instantiate_skill("iterate", "iterate-one", ["sub-skill", "do-action-say"], ["fact-type", "count-fact"], ["final-action-skill", "respond-one"])
    m.instantiate_skill("do-action", "do-action-say", ["action", "say"], ["main-skill", "iterate-one"])
    m.instantiate_skill("respond", "respond-one", ["action", "say"], ["arg", "stop"])
    digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    start = m.random(3)
    end = start + (1 + m.random(3))
    m.print("Counting from", digits[start], "to", digits[end])
    m.screen(digits[start], digits[end])
    m.run_until_action("done")
    m.issue_reward()
    m.trial_end()
